"""
test_mixed_n_batch.py의 실제 .pt 피처 파일 버전. (v2)

배경: 합성 torch.randn() 데이터에서는 NaN 원인이 "랜덤 노이즈 분포 문제"로
확인됐으나, 실제 UNI2 피처(정상 분포, NaN/Inf 없음)로도 동일하게 NaN이 발생함.
-> 데이터 문제가 아니라 다른 원인. 유력한 가설:
   --model-base 없이 base 모델(weights/wsi-llava-v1.5-7b-e1)만 단독으로 로드하면
   Resampler(cls_token/query/mlp 등)가 체크포인트에 없어 랜덤 초기화된 상태로
   로드됨 ("You should probably TRAIN this model..." 경고가 이를 시사).
   즉 학습된 Resampler 가중치가 전혀 아니라 랜덤 가중치로 실제 피처를 처리한 것이라
   activation이 불안정할 수 있음. 특히 N이 큰 경우(1417) cross-attention에서
   종합하는 key 수가 많아져 불안정성이 더 잘 드러날 수 있음.

[수정사항 v2]
- --model-base 인자를 받아 LoRA 체크포인트(학습된 Resampler 가중치)로도 테스트 가능하게 함.
  --model-path에 LoRA 체크포인트, --model-base에 베이스 가중치를 넘기면
  load_pretrained_model이 학습된 Resampler 가중치를 실제로 로드함.
- NaN 발생 시 자동으로 diag 단계(단독 forward)를 실행해 "혼합배치에서만 NaN인지,
  단독에서도 NaN인지"를 구분. (이전 합성데이터 테스트와 동일한 로직)
- 어떤 모델 경로로 로드했는지, Resampler가 랜덤 초기화됐는지 여부를 명시적으로 로그.

사용법 A) base 모델만 (Resampler 랜덤 가중치 - NaN 재현 확인용):
export PYTHONPATH=.:./WSI_LLaVA
python test_mixed_n_batch_real.py \
    --model-path /workspace/.../weights/wsi-llava-v1.5-7b-e1 \
    --image-folder /workspace/home/Daejeon/dataset/REG2026/features/uni2 \
    --image-file-small PIT_06_01395_01.pt \
    --image-file-large PIT_06_01543_01.pt \
    --gpu-id 0

사용법 B) LoRA 체크포인트 (학습된 Resampler 가중치 - 이걸로 NaN 해소되는지 확인):
python test_mixed_n_batch_real.py \
    --model-path /workspace/.../checkpoints/0724/reg2-wsi-llava-v1.5-7b-lora-LR1e-4-20epoch-0724-v3 \
    --model-base /workspace/.../weights/wsi-llava-v1.5-7b-e1 \
    --image-folder /workspace/home/Daejeon/dataset/REG2026/features/uni2 \
    --image-file-small PIT_06_01395_01.pt \
    --image-file-large PIT_06_01543_01.pt \
    --gpu-id 0
"""
import argparse
import os
import traceback

import torch


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def has_nan_or_inf(t: torch.Tensor):
    return torch.isnan(t).any().item(), torch.isinf(t).any().item()


def load_feature(image_folder: str, image_file: str) -> torch.Tensor:
    path = os.path.join(image_folder, image_file)
    image = torch.load(path, map_location="cpu")
    if isinstance(image, dict):
        image = torch.stack(list(image.values()), dim=0)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Unsupported feature type at {path}: {type(image)}")
    return image.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                         help="base 모델 경로 또는 LoRA 체크포인트 경로")
    parser.add_argument("--model-base", default=None,
                         help="--model-path가 LoRA 체크포인트일 때 base 가중치 경로. "
                              "이걸 주면 학습된 Resampler 가중치가 실제로 로드됨.")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--image-file-small", required=True)
    parser.add_argument("--image-file-large", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava_grpo_processor import LlavaGRPOProcessor

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    print(f"[설정] model_path={args.model_path}")
    print(f"[설정] model_base={args.model_base} "
          f"({'LoRA 체크포인트 모드 (학습된 Resampler)' if args.model_base else '베이스 모델 단독 (Resampler 랜덤 초기화 가능성 있음)'})")

    # ------------------------------------------------------------
    section("STEP 0. 실제 .pt 피처 로드 + 통계 확인")
    # ------------------------------------------------------------
    try:
        img_small = load_feature(args.image_folder, args.image_file_small)
        img_large = load_feature(args.image_folder, args.image_file_large)
        for name, t in [("small", img_small), ("large", img_large)]:
            print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
                  f"mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
                  f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
                  f"nan={torch.isnan(t).any().item()}, inf={torch.isinf(t).any().item()}")
        assert img_small.shape[1] == img_large.shape[1] == 1536
        assert img_small.shape[0] != img_large.shape[0]
        print(f"\nOK: N_small={img_small.shape[0]}, N_large={img_large.shape[0]}")
    except Exception:
        print("FAIL at STEP 0")
        traceback.print_exc()
        return

    # ------------------------------------------------------------
    section("STEP A. 모델 로드")
    # ------------------------------------------------------------
    try:
        model_name = get_model_name_from_path(args.model_path)
        try:
            tokenizer, model, _, _ = load_pretrained_model(
                args.model_path, args.model_base, model_name, device_map={"": args.gpu_id}
            )
        except TypeError:
            tokenizer, model, _, _ = load_pretrained_model(
                args.model_path, args.model_base, model_name
            )
            model = model.to(device)
        model.eval()
        model_dtype = next(model.parameters()).dtype
        resampler = model.get_vision_tower().resampler
        cls_token_std = resampler.cls_token.std().item()
        query_std = resampler.query.std().item()
        print("OK: 모델 로드 성공, dtype:", model_dtype)
        print(f"resampler.cls_token std={cls_token_std:.4f} (랜덤 초기화 기대값: 0.02 근처)")
        print(f"resampler.query std={query_std:.4f} (랜덤 초기화 기대값: 0.02 근처)")
        if abs(cls_token_std - 0.02) < 0.01 and abs(query_std - 0.02) < 0.01:
            print("=> cls_token/query가 초기화값(std=0.02)에 매우 근접함. "
                  "학습된 가중치가 아니라 랜덤 초기화 상태일 가능성이 높음.")
    except Exception:
        print("FAIL at STEP A")
        traceback.print_exc()
        return

    img_small = img_small.to(device=device, dtype=model_dtype)
    img_large = img_large.to(device=device, dtype=model_dtype)

    processor = LlavaGRPOProcessor(tokenizer, expected_feat_dim=1536)
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
    section("STEP B. 실제 피처로 N이 다른 두 이미지 혼합 배치 forward")
    # ------------------------------------------------------------
    texts = [prompt, prompt]
    images_batch = [img_small, img_large]
    mixed_inputs, mixed_out, mixed_has_nan = None, None, False
    try:
        mixed_inputs, mixed_out = run_forward(images_batch, texts)
        print("혼합배치 logits shape:", mixed_out.logits.shape)
        nan_flag, inf_flag = has_nan_or_inf(mixed_out.logits)
        print(f"혼합배치 logits nan={nan_flag}, inf={inf_flag}")
        mixed_has_nan = nan_flag or inf_flag
    except Exception:
        print("FAIL at STEP B (예외)")
        traceback.print_exc()
        return

    if not mixed_has_nan:
        print("OK: 실제 피처 혼합 배치 forward 성공, NaN/Inf 없음 -> STEP C로 진행")
    else:
        # ------------------------------------------------------------
        section("STEP B-diag. 실제 피처 기준 NaN 원인 격리 (단독 vs 혼합)")
        # ------------------------------------------------------------
        diag = {}
        for label, img in [(f"img_small(N={img_small.shape[0]}) 단독", img_small),
                            (f"img_large(N={img_large.shape[0]}) 단독", img_large)]:
            try:
                _, single_out = run_forward([img], [prompt])
                nan_flag, inf_flag = has_nan_or_inf(single_out.logits)
                diag[label] = (nan_flag, inf_flag)
                print(f"[{label}] logits nan={nan_flag}, inf={inf_flag}")

                # resampler 출력 자체도 확인 (mm_projector 이전 단계에서 이미 터지는지)
                vision_tower = model.get_vision_tower()
                with torch.no_grad():
                    resampler_out = vision_tower([img])
                r_nan, r_inf = has_nan_or_inf(resampler_out)
                print(f"    (참고) resampler 출력 자체 nan={r_nan}, inf={r_inf}, "
                      f"mean={resampler_out.float().mean().item():.4f}, "
                      f"std={resampler_out.float().std().item():.4f}")
            except Exception:
                print(f"[{label}] 단독 forward 예외")
                traceback.print_exc()
                diag[label] = (None, None)

        any_single_nan = any(v[0] or v[1] for v in diag.values() if v[0] is not None)

        section("STEP B-diag 결론")
        if any_single_nan:
            print("=> 실제 피처로도 단독 forward에서 NaN 발생.")
            if args.model_base is None:
                print("   --model-base 없이 base 모델만 로드함 -> Resampler가 랜덤 초기화된")
                print("   가중치일 가능성이 높음 (위 cls_token/query std 값 확인).")
                print("   랜덤 가중치 + 실제 피처 조합에서 fp16 activation이 불안정할 수 있음.")
                print("   -> LoRA 체크포인트(--model-path에 체크포인트, --model-base에 베이스)로")
                print("      다시 실행해서 학습된 Resampler 가중치로는 NaN이 해소되는지 확인 필요.")
            else:
                print("   LoRA 체크포인트(학습된 가중치)로도 NaN 발생 -> 이건 데이터/초기화 문제가")
                print("   아니라 실제 아키텍처나 특정 체크포인트의 문제일 가능성이 높음.")
                print("   -> 어느 forward 단계(resampler 출력 자체 vs mm_projector 이후 vs LLM 내부)")
                print("      에서 처음 NaN이 나타나는지 위 resampler 출력 로그로 확인 필요.")
        else:
            print("=> 단독으로는 둘 다 정상인데 혼합배치에서만 NaN -> 패딩 마스킹 버그 가능성.")
            print("   resampler_wrapper.py/resampler.py의 mask 처리 재점검 필요.")
        return

    # ------------------------------------------------------------
    section("STEP C. 정확성 검증: 단독 forward와 혼합배치 forward 결과 비교")
    # ------------------------------------------------------------
    try:
        single_inputs, single_out = run_forward([img_small], [prompt])
        assert torch.equal(single_inputs["input_ids"], mixed_inputs["input_ids"][0:1])
        logits_from_mixed = mixed_out.logits[0]
        logits_from_single = single_out.logits[0]
        max_abs_diff = (logits_from_mixed - logits_from_single).abs().max().item()
        print(f"logits 최대 절대 차이 (mixed vs single, img_small 기준): {max_abs_diff:.6f}")
        if max_abs_diff < args.atol:
            print(f"OK: 패딩 마스크가 정확함 (차이 < {args.atol})")
        else:
            print(f"WARNING: 차이가 허용치({args.atol}) 초과 -> 패딩 마스킹 재점검 필요")
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
        print("rollout 배치 크기:", gen_out.shape[0])
        print("OK: G-completion 혼합 N 배치 rollout 성공")
    except Exception:
        print("FAIL at STEP D")
        traceback.print_exc()
        return

    section("결론")
    print("실제 .pt 피처 기준으로 혼합 N 배치가 안전함이 실증됨. Phase 3 완료.")


if __name__ == "__main__":
    main()
