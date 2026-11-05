"""
GRPOTrainer 통합 전, 우리 커스텀 LlavaLlamaForCausalLM + ResamplerWrapper 조합이
forward()와 generate() 양쪽에서 image 피처를 정상적으로 소비하는지 단독 검증하는 스크립트.

[수정사항] LoRA 체크포인트 + 베이스 가중치 분리 구조를 반영.
model_vqa.py / inference 스크립트와 동일하게 llava.model.builder.load_pretrained_model()을
그대로 사용한다. 즉:
    --model-path : 학습 돌려서 생긴 LoRA 체크포인트 (output_dir 아래 저장된 디렉토리)
    --model-base : 학습 전 베이스 가중치 (--model_name_or_path로 썼던 경로)
베이스 모델만 단독으로 forward/generate 테스트하고 싶으면 --model-base를 생략하면 된다
(이 경우 --model-path에 베이스 가중치 경로를 직접 넣는다).

실행 위치: llava 패키지(llava_llama.py, llava_arch.py, mm_utils.py, constants.py, builder.py)가
import 가능한 디렉토리 (예: WSI-LLaVA-dev 루트, PYTHONPATH=. 설정).

사용법 예시 (LoRA 체크포인트 검증):
    python test_llava_forward_generate.py \\
        --model-path /workspace/.../checkpoints/0724/reg2-wsi-llava-v1.5-7b-lora-LR1e-4-20epoch-0724-v3 \\
        --model-base /workspace/.../weights/wsi-llava-v1.5-7b-e1

사용법 예시 (베이스 모델만 검증):
    python test_llava_forward_generate.py \\
        --model-path /workspace/.../weights/wsi-llava-v1.5-7b-e1
"""
import argparse
import traceback

import torch


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", required=True,
        help="LoRA 체크포인트 경로 (학습 output_dir) 또는 베이스 모델 경로(단독 검증 시)",
    )
    parser.add_argument(
        "--model-base", default=None,
        help="베이스 가중치 경로 (--model-path가 LoRA 체크포인트일 때 필수). "
             "학습 스크립트의 --model_name_or_path와 동일한 경로.",
    )
    parser.add_argument("--feat-dim", type=int, default=1536, help="tile encoder 출력 차원 (ResamplerWrapper input_dim)")
    parser.add_argument("--num-patches", type=int, default=200, help="가짜 이미지의 patch(N) 개수")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
    from llava.model.builder import load_pretrained_model  # model_vqa.py와 동일한 로더 사용

    device = args.device

    # ---------------------------------------------------------------
    section("STEP 0. 모델 / 토크나이저 로드 (load_pretrained_model 사용)")
    # ---------------------------------------------------------------
    try:
        model_name = get_model_name_from_path(args.model_path)
        print("model_name:", model_name)
        print("model_path:", args.model_path)
        print("model_base:", args.model_base, "(None이면 베이스 단독 로드)")

        # model_vqa.py와 동일한 시그니처: (model_path, model_base, model_name)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name
        )
        model = model.to(device)
        model.eval()
        print("OK: 모델/토크나이저 로드 성공")
        print("vision_tower:", type(model.get_vision_tower()))
        print("is_resampler:", getattr(model.get_vision_tower(), "is_resampler", None))
        print("context_len:", context_len)
    except Exception:
        print("FAIL at STEP 0")
        traceback.print_exc()
        return

    # ---------------------------------------------------------------
    section("STEP 1. 가짜 이미지 피처 + 프롬프트 토크나이즈")
    # ---------------------------------------------------------------
    try:
        # 실제로는 tile encoder가 뽑아준 (N, feat_dim) 텐서를 여기 넣게 됨.
        # (예: .pt 파일 하나를 torch.load()해서 그대로 넣어도 됨)
        fake_image_feat = torch.randn(args.num_patches, args.feat_dim, dtype=torch.bfloat16, device=device)

        prompt = f"{DEFAULT_IMAGE_TOKEN}\n이 이미지에 대해 설명해줘."
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        n_image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
        print("input_ids shape:", input_ids.shape)
        print("IMAGE_TOKEN_INDEX 개수 (반드시 1이어야 함):", n_image_tokens)
        assert n_image_tokens == 1, "우리 아키텍처는 <image> 플레이스홀더가 정확히 1개여야 함"

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        print("OK: 토크나이즈 성공")
    except Exception:
        print("FAIL at STEP 1")
        traceback.print_exc()
        return

    # ---------------------------------------------------------------
    section("STEP 2. model.forward() 단독 호출 (loss/logits 경로)")
    # ---------------------------------------------------------------
    try:
        images_arg = [fake_image_feat]  # encode_images가 list[Tensor]를 기대 (ResamplerWrapper 분기)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images_arg,
                return_dict=True,
            )
        print("logits shape:", out.logits.shape)
        print("logits nan/inf:", torch.isnan(out.logits).any().item(), torch.isinf(out.logits).any().item())
        print("OK: forward() 정상 동작")
    except Exception:
        print("FAIL at STEP 2 (forward)")
        traceback.print_exc()
        return

    # ---------------------------------------------------------------
    section("STEP 3. model.generate() 단독 호출 (rollout 경로, GRPO가 실제로 타는 경로)")
    # ---------------------------------------------------------------
    try:
        with torch.no_grad():
            gen_out = model.generate(
                inputs=input_ids,
                images=images_arg,
                attention_mask=attention_mask,
                image_sizes=None,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        print("generate() 출력 shape:", gen_out.shape)
        print("디코딩 결과:", decoded)
        print("OK: generate() 정상 동작")
    except Exception:
        print("FAIL at STEP 3 (generate) <- GRPOTrainer가 여기서 깨질 가능성이 가장 큰 지점")
        traceback.print_exc()
        return

    # ---------------------------------------------------------------
    section("STEP 4. GRPOTrainer가 쓰는 'processing_class(images=, text=)' 흉내내기")
    # ---------------------------------------------------------------
    # TRL 원문: prompt_inputs = self.processing_class(images=images, text=prompts_text,
    #                                                  padding=True, return_tensors="pt")
    #           forward_kwargs = {k: v for k, v in prompt_inputs.items()
    #                              if k not in ("input_ids", "attention_mask")}
    try:
        class FakeLlavaProcessor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, images, text, padding=True, return_tensors="pt"):
                assert isinstance(text, list)
                input_ids_list = []
                for t in text:
                    ids = tokenizer_image_token(t, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    input_ids_list.append(ids)
                max_len = max(x.shape[0] for x in input_ids_list)
                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                padded_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
                attn = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
                for i, ids in enumerate(input_ids_list):
                    padded_ids[i, : ids.shape[0]] = ids
                    attn[i, : ids.shape[0]] = 1
                flat_images = [img_list[0] for img_list in images]  # 샘플당 이미지 1개 가정
                return {
                    "input_ids": padded_ids,
                    "attention_mask": attn,
                    "images": flat_images,
                }

        fake_processor = FakeLlavaProcessor(tokenizer)
        prompts_text = [prompt, prompt]  # batch size 2 흉내
        images_batch = [[fake_image_feat], [fake_image_feat.clone()]]

        prompt_inputs = fake_processor(images=images_batch, text=prompts_text, padding=True, return_tensors="pt")
        forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
        print("forward_kwargs 키:", list(forward_kwargs.keys()))

        with torch.no_grad():
            out2 = model(
                input_ids=prompt_inputs["input_ids"].to(device),
                attention_mask=prompt_inputs["attention_mask"].to(device),
                **{k: (v if k != "images" else [x.to(device) for x in v]) for k, v in forward_kwargs.items()},
            )
        print("배치 forward logits shape:", out2.logits.shape)
        print("OK: GRPOTrainer가 기대하는 processing_class(images=, text=) 흐름 재현 성공")
        print("\n결론: 위 FakeLlavaProcessor와 같은 형태의 커스텀 processing_class를 만들어")
        print("GRPOConfig(processing_class=...)에 넘기면 GRPOTrainer 통합이 가능할 가능성이 높다.")
    except Exception:
        print("FAIL at STEP 4 (GRPO 흉내 wrapper)")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
