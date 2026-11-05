"""
STEP C에서 발견된 "mixed vs single logits 최대 차이 = 0.126"이
(A) 패딩 마스킹이 실제로 attention에 새는 진짜 버그인지, 아니면
(B) fp16 수치오차가 32개 LLM 레이어를 거치며 증폭된 것인지 구분한다.

방법:
1. resampler 출력(mm_projector, LLM 통과 전) 단계에서 단독 vs 혼합배치 결과를 직접 비교.
   -> 이 단계에서 이미 차이가 크면 (A), 여기서는 작은데 최종 logits에서만 커지면 (B) 쪽으로 기움.
2. 동일 비교를 float32로도 수행.
   -> fp16과 fp32 결과를 나란히 보면 (A)/(B) 판단이 더 명확해짐.
   (fp32는 메모리가 더 필요하므로 7B 모델 fp32 로드가 버거우면 --skip-fp32로 건너뛸 수 있게 함)

사용법:
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.:./WSI_LLaVA python test_padding_mask_precision.py \
    --model-path <LoRA 체크포인트 경로> \
    --model-base <베이스 경로> \
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


def load_feature(image_folder: str, image_file: str) -> torch.Tensor:
    path = os.path.join(image_folder, image_file)
    image = torch.load(path, map_location="cpu")
    if isinstance(image, dict):
        image = torch.stack(list(image.values()), dim=0)
    return image.float()


def compare_resampler_level(model, img_small, img_large, device, dtype, label):
    """resampler(+mm_projector) 출력 단계에서 단독 vs 혼합배치 비교."""
    vision_tower = model.get_vision_tower()
    mm_projector = model.get_model().mm_projector

    img_small_d = img_small.to(device=device, dtype=dtype)
    img_large_d = img_large.to(device=device, dtype=dtype)

    with torch.no_grad():
        # 단독
        single_resampler_out = vision_tower([img_small_d])
        single_proj_out = mm_projector(single_resampler_out)

        # 혼합배치 (N=169 vs N=1417)
        mixed_resampler_out = vision_tower([img_small_d, img_large_d])
        mixed_proj_out = mm_projector(mixed_resampler_out)

    resampler_diff = (single_resampler_out[0] - mixed_resampler_out[0]).abs().max().item()
    proj_diff = (single_proj_out[0] - mixed_proj_out[0]).abs().max().item()

    print(f"[{label}] resampler 출력 shape: {single_resampler_out.shape}")
    print(f"[{label}] resampler 출력 최대 차이 (single vs mixed, img_small 기준): {resampler_diff:.8f}")
    print(f"[{label}] mm_projector 출력 최대 차이 (single vs mixed, img_small 기준): {proj_diff:.8f}")
    return resampler_diff, proj_diff


def compare_final_logits(model, tokenizer, img_small, img_large, device, dtype, label):
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava_grpo_processor import LlavaGRPOProcessor

    processor = LlavaGRPOProcessor(tokenizer, expected_feat_dim=1536)
    prompt = f"{DEFAULT_IMAGE_TOKEN}\nWrite a report about this image"

    img_small_d = img_small.to(device=device, dtype=dtype)
    img_large_d = img_large.to(device=device, dtype=dtype)

    def run_forward(images_list, texts_list):
        inputs = processor(images=images_list, text=texts_list)
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                images=[x.to(device) for x in inputs["images"]],
                return_dict=True,
            )
        return out

    single_out = run_forward([img_small_d], [prompt])
    mixed_out = run_forward([img_small_d, img_large_d], [prompt, prompt])

    logits_diff = (single_out.logits[0] - mixed_out.logits[0]).abs().max().item()
    print(f"[{label}] 최종 logits 최대 차이 (single vs mixed, img_small 기준): {logits_diff:.8f}")
    return logits_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--image-file-small", required=True)
    parser.add_argument("--image-file-large", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--skip-fp32", action="store_true",
                         help="fp32 재검증 생략 (메모리 부족 시)")
    args = parser.parse_args()

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    section("STEP 0. 실제 피처 로드")
    img_small = load_feature(args.image_folder, args.image_file_small)
    img_large = load_feature(args.image_folder, args.image_file_large)
    print(f"N_small={img_small.shape[0]}, N_large={img_large.shape[0]}")

    section("STEP 1. 모델 로드 (fp16, 기존과 동일)")
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
        print("OK: fp16 모델 로드 성공")
    except Exception:
        print("FAIL at STEP 1")
        traceback.print_exc()
        return

    section("STEP 2. [fp16] resampler/projector 단계 vs 최종 logits 단계 비교")
    try:
        r_diff_fp16, p_diff_fp16 = compare_resampler_level(
            model, img_small, img_large, device, torch.float16, "fp16"
        )
        logits_diff_fp16 = compare_final_logits(
            model, tokenizer, img_small, img_large, device, torch.float16, "fp16"
        )
    except Exception:
        print("FAIL at STEP 2")
        traceback.print_exc()
        return

    section("STEP 2 결론 (fp16 기준, 어느 단계에서 차이가 커지는지)")
    print(f"resampler 단계 차이   : {r_diff_fp16:.8f}")
    print(f"mm_projector 단계 차이: {p_diff_fp16:.8f}")
    print(f"최종 logits 단계 차이 : {logits_diff_fp16:.8f}")
    if r_diff_fp16 > 1e-2:
        print("=> resampler 출력 자체에서부터 이미 큰 차이. LLM 진입 전부터 문제 -> "
              "패딩 마스킹이 실제로 새는 버그일 가능성이 높음 (진짜 버그 쪽).")
    elif logits_diff_fp16 > 1e-1 and r_diff_fp16 < 1e-3:
        print("=> resampler 단계는 거의 동일한데 최종 logits에서만 크게 벌어짐.")
        print("   -> 32개 LLM 레이어를 거치며 fp16 수치오차가 증폭된 것일 가능성이 높음")
        print("   (패딩 마스킹 자체는 정상, fp16 정밀도 한계 쪽에 더 가까움)")
    else:
        print("=> 중간 정도 패턴. STEP 3(fp32 비교)로 추가 확인 필요.")

    if args.skip_fp32:
        print("\n--skip-fp32 지정됨 -> fp32 재검증 생략, 여기서 종료")
        return

    section("STEP 3. [fp32] 동일 비교 재실행 (fp16 vs fp32 대조)")
    try:
        del model
        torch.cuda.empty_cache()
        model_name = get_model_name_from_path(args.model_path)
        # fp32로 재로드: builder.py의 load_pretrained_model은 기본적으로 torch_dtype=float16을
        # 강제 지정하므로, 로드 후 모델 전체를 .float()로 캐스팅해서 fp32로 만든다.
        try:
            tokenizer, model, _, _ = load_pretrained_model(
                args.model_path, args.model_base, model_name, device_map={"": args.gpu_id}
            )
        except TypeError:
            tokenizer, model, _, _ = load_pretrained_model(
                args.model_path, args.model_base, model_name
            )
            model = model.to(device)
        model = model.float()
        model.eval()
        print("OK: fp32로 캐스팅 완료")

        r_diff_fp32, p_diff_fp32 = compare_resampler_level(
            model, img_small, img_large, device, torch.float32, "fp32"
        )
        logits_diff_fp32 = compare_final_logits(
            model, tokenizer, img_small, img_large, device, torch.float32, "fp32"
        )
    except torch.cuda.OutOfMemoryError:
        print("FAIL: fp32 로드 중 OOM 발생. --skip-fp32로 재실행하거나 더 큰 GPU 필요.")
        return
    except Exception:
        print("FAIL at STEP 3")
        traceback.print_exc()
        return

    section("최종 결론 (fp16 vs fp32 대조)")
    print(f"{'단계':<20}{'fp16 차이':<15}{'fp32 차이':<15}")
    print(f"{'resampler':<20}{r_diff_fp16:<15.8f}{r_diff_fp32:<15.8f}")
    print(f"{'mm_projector':<20}{p_diff_fp16:<15.8f}{p_diff_fp32:<15.8f}")
    print(f"{'최종 logits':<20}{logits_diff_fp16:<15.8f}{logits_diff_fp32:<15.8f}")

    if logits_diff_fp32 < 1e-3 and logits_diff_fp16 > 1e-1:
        print("\n=> fp32에서는 차이가 거의 사라짐 (수치오차 증폭이 원인, 진짜 버그 아님).")
        print("   패딩 마스킹 로직은 정상. fp16 사용에 따른 자연스러운 현상으로 결론.")
    elif logits_diff_fp32 > 1e-2:
        print("\n=> fp32에서도 차이가 크게 남음. fp16 정밀도 문제가 아니라 구조적 버그일 가능성이 높음.")
        print("   resampler_wrapper.py의 mask 생성/전달 로직을 다시 점검해야 함.")
    else:
        print("\n=> 판단이 모호함. resampler/projector 단계별 수치를 다시 뜯어봐야 함.")


if __name__ == "__main__":
    main()
