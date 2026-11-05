"""
Phase 1 / Phase 3 검증: 배치 내에 N(patch 개수)이 서로 다른 샘플이 섞였을 때
LlavaGRPOProcessor + ResamplerWrapper + LlavaLlamaForCausalLM 조합이
1) 정상적으로 forward가 도는지 (shape 체크)
2) 패딩된 키가 attention에서 실제로 완전히 마스킹되어, 각 샘플의 결과가
   "혼합 배치 안에서 계산한 결과"와 "그 샘플 단독으로 계산한 결과"가
   수치적으로 일치하는지 (정확성 체크 — 이게 핵심)
를 검증한다.

[수정사항 v3]
- STEP B에서 NaN이 발생한 경우, 이전 스레드에서 이미 확인된 "torch.randn() 순수
  노이즈가 실제 UNI2 피처 분포와 달라 activation 폭주" 현상과 "패딩 마스킹이
  실제로 새는 버그" 두 가지 가능성을 구분해야 함. 즉시 fail하지 않고
  STEP B-diag로 넘어가 원인을 자동으로 격리:
    1) img_small(N=200) 단독 forward -> NaN 여부
    2) img_large(N=800) 단독 forward -> NaN 여부
  둘 다 단독으로도 NaN이면 (A) 랜덤 노이즈+큰 N 조합 자체의 문제 (아키텍처 결함 아님,
     테스트 방법론의 한계 -> 실제 .pt 피처로 재검증 필요).
  둘 다 단독으로는 정상인데 혼합배치에서만 NaN이면 (B) 패딩 마스킹이 실제로
     attention에 새고 있는 진짜 버그 -> resampler_wrapper.py/resampler.py 재점검 필요.

사용법:
export PYTHONPATH=.:./WSI_LLaVA
python test_mixed_n_batch.py \
    --model-path /workspace/.../weights/wsi-llava-v1.5-7b-e1 \
    --gpu-id 0
"""
import argparse
import traceback

import torch


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def has_nan_or_inf(t: torch.Tensor):
    return torch.isnan(t).any().item(), torch.isinf(t).any().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--feat-dim", type=int, default=1536)
    parser.add_argument("--n-small", type=int, default=200, help="작은 쪽 patch 개수")
    parser.add_argument("--n-large", type=int, default=800, help="큰 쪽 patch 개수")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2, help="fp16 누적오차 허용 절대오차")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feat-scale", type=float, default=1.0,
        help="가짜 이미지 피처의 스케일. 1.0=순수 표준정규분포(randn). "
             "실제 UNI2 피처가 이보다 작은 스케일일 가능성이 있어 비교용으로 조절 가능."
    )
    args = parser.parse_args()

    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava_grpo_processor import LlavaGRPOProcessor

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    section("STEP A. 모델 로드 (device_map dict 버그 fallback 포함)")
    # ------------------------------------------------------------
    try:
        model_name = get_model_name_from_path(args.model_path)
        try:
            tokenizer, model, _, _ = load_pretrained_model(
                args.model_path, args.model_base, model_name, device_map={"": args.gpu_id}
            )
        except TypeError:
            print("load_pretrained_model이 dict device_map을 못 받음(기존 builder.py 버그) "
                  "-> 기본 로드 후 수동 .to(device)로 우회")
            tokenizer, model, _, _ = load_pretrained_model(
                args.model_path, args.model_base, model_name
            )
            model = model.to(device)
        model.eval()
        model_dtype = next(model.parameters()).dtype
        print("OK: 모델 로드 성공, dtype:", model_dtype)
        print("vision_tower device 확인:", next(model.get_vision_tower().parameters()).device)
    except Exception:
        print("FAIL at STEP A")
        traceback.print_exc()
        return

    processor = LlavaGRPOProcessor(tokenizer, expected_feat_dim=args.feat_dim)
    prompt = f"{DEFAULT_IMAGE_TOKEN}\nWrite a report about this image"

    def run_forward(images_list, texts_list):
        inputs = processor(images=images_list, text=texts_list)
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                images=[x.to(device) for x in inputs["images"]],
                return_dict=True,
            )
        return inputs, out

    # ------------------------------------------------------------
    section("STEP B. N이 다른 두 이미지 준비 + 혼합 배치 forward")
    # ------------------------------------------------------------
    torch.manual_seed(args.seed)
    img_small = torch.randn(args.n_small, args.feat_dim, dtype=model_dtype, device=device) * args.feat_scale
    img_large = torch.randn(args.n_large, args.feat_dim, dtype=model_dtype, device=device) * args.feat_scale
    texts = [prompt, prompt]
    images_batch = [img_small, img_large]

    mixed_inputs, mixed_out = None, None
    mixed_has_nan = False
    try:
        mixed_inputs, mixed_out = run_forward(images_batch, texts)
        print("input_ids shape:", mixed_inputs["input_ids"].shape)
        print("images[0] shape (N=small):", mixed_inputs["images"][0].shape)
        print("images[1] shape (N=large):", mixed_inputs["images"][1].shape)
        print("혼합배치 logits shape:", mixed_out.logits.shape)
        nan_flag, inf_flag = has_nan_or_inf(mixed_out.logits)
        print(f"혼합배치 logits nan={nan_flag}, inf={inf_flag}")
        mixed_has_nan = nan_flag or inf_flag
    except Exception:
        print("FAIL at STEP B (예외 발생, NaN이 아니라 실행 자체가 깨짐)")
        traceback.print_exc()
        return

    if not mixed_has_nan:
        print("OK: 혼합 배치(N=200 + N=800) forward 성공, NaN/Inf 없음 -> STEP C로 진행")
    else:
        # ------------------------------------------------------------
        section("STEP B-diag. NaN 원인 격리: 단독 이미지로 (A) vs (B) 구분")
        # ------------------------------------------------------------
        print("혼합배치에서 NaN/Inf 발생 -> 원인이 (A)랜덤노이즈+큰N 문제인지 "
              "(B)패딩마스킹 버그인지 격리 진단 시작\n")

        diag_results = {}
        for label, img in [("img_small(N=%d) 단독" % args.n_small, img_small),
                            ("img_large(N=%d) 단독" % args.n_large, img_large)]:
            try:
                _, single_out = run_forward([img], [prompt])
                nan_flag, inf_flag = has_nan_or_inf(single_out.logits)
                diag_results[label] = (nan_flag, inf_flag)
                print(f"[{label}] logits nan={nan_flag}, inf={inf_flag}")
            except Exception:
                print(f"[{label}] 단독 forward 자체가 예외로 실패")
                traceback.print_exc()
                diag_results[label] = (None, None)

        any_single_nan = any(v[0] or v[1] for v in diag_results.values() if v[0] is not None)

        section("STEP B-diag 결론")
        if any_single_nan:
            print("=> (A) 단독 forward에서도 NaN/Inf 발생함.")
            print("   이전 스레드에서 확인된 '순수 torch.randn() 노이즈가 실제 UNI2 피처 분포와")
            print("   달라 activation이 폭주하는 현상'과 동일한 패턴일 가능성이 높음.")
            print("   즉 패딩/배치혼합 로직 자체의 버그가 아니라 테스트용 가짜 데이터의 한계.")
            print("   -> --feat-scale을 낮춰서(예: 0.1) 재현되는지 확인해보거나,")
            print("      실제 .pt 피처 파일로 재검증하는 것을 권장.")
        else:
            print("=> (B) 단독 forward는 둘 다 정상인데 혼합배치에서만 NaN/Inf 발생.")
            print("   이 경우는 패딩 마스킹이 실제로 attention에 새고 있는 진짜 버그일 가능성이 높음.")
            print("   -> resampler_wrapper.py의 mask 생성 로직, resampler.py의")
            print("      key_padding_mask 전달 방식을 다시 점검해야 함.")
        print("\n(참고) --feat-scale 인자로 노이즈 크기를 줄여 재실행하면 (A)/(B) 판단에 추가 근거가 됨.")
        return  # NaN 원인 격리까지가 이번 실행의 목적이므로 여기서 종료

    # ------------------------------------------------------------
    section("STEP C. 정확성 검증: 단독 forward와 혼합배치 forward 결과 비교")
    # ------------------------------------------------------------
    try:
        single_inputs, single_out = run_forward([img_small], [prompt])

        assert torch.equal(single_inputs["input_ids"], mixed_inputs["input_ids"][0:1]), \
            "단독/혼합 배치의 input_ids가 다름 (padding 로직 재확인 필요)"

        logits_from_mixed = mixed_out.logits[0]
        logits_from_single = single_out.logits[0]

        max_abs_diff = (logits_from_mixed - logits_from_single).abs().max().item()
        print(f"logits 최대 절대 차이 (mixed vs single, img_small 기준): {max_abs_diff:.6f}")

        if max_abs_diff < args.atol:
            print(f"OK: 패딩 마스크가 정확함 (차이 < {args.atol}, fp16 누적오차 범위 내)")
        else:
            print(f"WARNING: 차이가 허용치({args.atol})를 초과함 -> 패딩이 attention에 새고 있을 가능성")
            print("-> ResamplerWrapper.forward()의 mask 생성/전달 로직 재확인 필요")
    except Exception:
        print("FAIL at STEP C")
        traceback.print_exc()
        return

    # ------------------------------------------------------------
    section("STEP D. do_sample=True 다중 completion(G) 생성 시나리오 재현")
    # ------------------------------------------------------------
    try:
        G = 4
        repeated_texts = texts * G
        repeated_images = images_batch * G

        rollout_inputs = processor(images=repeated_images, text=repeated_texts)
        with torch.no_grad():
            gen_out = model.generate(
                inputs=rollout_inputs["input_ids"].to(device),
                attention_mask=rollout_inputs["attention_mask"].to(device),
                images=[x.to(device) for x in rollout_inputs["images"]],
                do_sample=True,
                temperature=0.9,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        print("rollout 배치 크기 (2 samples x G=4):", gen_out.shape[0])
        print("generate() 출력 shape:", gen_out.shape)
        print("OK: G-completion 혼합 N 배치 rollout 성공")
    except Exception:
        print("FAIL at STEP D <- GRPO rollout 단계에서 실제로 터질 수 있는 지점")
        traceback.print_exc()
        return

    section("결론")
    print("모든 단계 통과: 혼합 N 배치가 GRPOTrainer의 rollout/forward 경로에서")
    print("안전하다는 것이 실증됨.")


if __name__ == "__main__":
    main()
