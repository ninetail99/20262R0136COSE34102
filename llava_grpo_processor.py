"""
GRPOTrainer용 실전 processing_class 구현.

배경 (test_llava_forward_generate.py STEP 4의 FakeLlavaProcessor 후속작):
- TRL GRPOTrainer는 processing_class(images=, text=, return_tensors="pt")를 호출해서
  forward_kwargs를 얻고, input_ids/attention_mask를 제외한 나머지 키를 그대로
  model(**forward_kwargs)에 넘긴다.
- 우리 아키텍처(ResamplerWrapper)는 vision tower가 CLIP이 아니라 이미 추출된
  (N, 1536) tile feature 텐서를 list[Tensor] 형태로 받는다. N은 샘플마다 다를 수 있다.
- FakeLlavaProcessor 프로토타입은 배치 내 모든 이미지의 N이 동일한 경우만 테스트됨.
  이 버전은 배치 내에 N이 서로 다른 샘플이 섞여 있는 상황(GRPO의 RepeatSampler가
  N 그룹핑 없이 인덱스를 그대로 뽑기 때문에 실제로 발생 가능)을 명시적으로 지원한다.

주의 (미검증 항목, 실제 TRL 소스로 재확인 필요):
- GRPOTrainer가 processing_class에 decode/batch_decode 같은 추가 메서드를 요구하는지는
  grpo_trainer.py 원문에서 확인되지 않음. 필요 시 아래 __getattr__ passthrough로 대응.
- padding_side를 GRPOTrainer가 강제하는지(left/right) 미확인. LLaVA 학습 관례상
  right padding을 기본값으로 두되, 인자로 노출해 둠.
"""
from typing import List, Optional, Sequence, Union

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token


class LlavaGRPOProcessor:
    """
    ResamplerWrapper 기반 WSI-LLaVA 아키텍처용 GRPOConfig(processing_class=...) 구현체.

    __call__(images=, text=, ...) 반환값:
        {
            "input_ids": LongTensor (B, L),
            "attention_mask": LongTensor (B, L),
            "images": List[Tensor],  # 길이 B, 각 원소 shape (N_i, feat_dim), N_i는 샘플마다 다를 수 있음
        }

    ResamplerWrapper.forward()가 이 list[Tensor]를 받아 배치 내부적으로
    zero-padding + key_padding_mask를 적용하므로, 여기서는 N을 맞춰줄 필요가 없다.
    (패딩/마스크 책임은 전적으로 ResamplerWrapper 쪽에 있음 — 이중 패딩 금지)
    """

    def __init__(
        self,
        tokenizer,
        image_token_index: int = IMAGE_TOKEN_INDEX,
        padding_side: str = "right",
        expected_feat_dim: Optional[int] = 1536,
    ):
        self.tokenizer = tokenizer
        self.image_token_index = image_token_index
        self.padding_side = padding_side
        self.expected_feat_dim = expected_feat_dim

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer에 pad_token_id/eos_token_id가 모두 없음")
        self.pad_token_id = pad_id

    # ------------------------------------------------------------------
    def __call__(
        self,
        images: Optional[Sequence[Union[torch.Tensor, List[torch.Tensor]]]] = None,
        text: Optional[Union[str, Sequence[str]]] = None,
        padding: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ):
        if text is None:
            raise ValueError("`text`는 필수 인자입니다.")
        if isinstance(text, str):
            text = [text]
        if images is None:
            raise ValueError("`images`는 필수 인자입니다. (이 아키텍처는 텍스트 전용 입력을 지원하지 않음)")
        if len(text) != len(images):
            raise ValueError(
                f"text 배치 크기({len(text)})와 images 배치 크기({len(images)})가 다릅니다."
            )

        input_ids_list = self._tokenize_batch(text)
        flat_images = self._normalize_images(images)

        if return_tensors != "pt":
            raise ValueError(f"return_tensors='{return_tensors}'는 지원하지 않음 (pt만 지원)")

        padded_ids, attention_mask = self._pad_input_ids(input_ids_list)

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "images": flat_images,
        }

    # ------------------------------------------------------------------
    def _tokenize_batch(self, text: Sequence[str]) -> List[torch.Tensor]:
        # tokenizer_image_token()은 배치를 지원하지 않으므로(단일 문자열 전용) 개별 순회.
        input_ids_list = []
        for t in text:
            ids = tokenizer_image_token(
                t, self.tokenizer, self.image_token_index, return_tensors="pt"
            )
            n_img_tokens = (ids == self.image_token_index).sum().item()
            if n_img_tokens != 1:
                raise ValueError(
                    f"이 아키텍처는 프롬프트당 정확히 1개의 <image> 플레이스홀더를 요구합니다. "
                    f"실제 개수: {n_img_tokens} (text={t!r})"
                )
            input_ids_list.append(ids)
        return input_ids_list

    def _pad_input_ids(self, input_ids_list: List[torch.Tensor]):
        batch_size = len(input_ids_list)
        max_len = max(ids.shape[0] for ids in input_ids_list)

        padded_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            L = ids.shape[0]
            if self.padding_side == "right":
                padded_ids[i, :L] = ids
                attention_mask[i, :L] = 1
            elif self.padding_side == "left":
                padded_ids[i, -L:] = ids
                attention_mask[i, -L:] = 1
            else:
                raise ValueError(f"padding_side='{self.padding_side}' 은 'left'/'right'만 지원")

        return padded_ids, attention_mask

    def _normalize_images(
        self, images: Sequence[Union[torch.Tensor, List[torch.Tensor]]]
    ) -> List[torch.Tensor]:
        """
        지원하는 images 입력 형태 2가지:
        1) List[Tensor]            - 샘플당 이미지 1개, 바로 텐서 (권장 형태)
        2) List[List[Tensor]]      - 메시지 블록 스타일 호환용(content=[{"image":...}] 등에서
                                      추출되어 넘어올 경우). 각 내부 리스트는 길이 1이어야 함
                                      (이 아키텍처는 샘플당 이미지 1장만 지원).
        N(패치 수)은 원소마다 달라도 됨 — 여기서는 맞추지 않고 그대로 ResamplerWrapper에 위임.
        """
        flat_images: List[torch.Tensor] = []
        for entry in images:
            if isinstance(entry, torch.Tensor):
                img = entry
            elif isinstance(entry, (list, tuple)):
                if len(entry) != 1:
                    raise ValueError(
                        f"샘플당 이미지 1개만 지원하는 아키텍처인데 {len(entry)}개가 들어왔습니다."
                    )
                img = entry[0]
            else:
                raise TypeError(f"지원하지 않는 images 원소 타입: {type(entry)}")

            if not isinstance(img, torch.Tensor):
                raise TypeError(f"이미지 원소는 torch.Tensor여야 합니다. got {type(img)}")
            if img.ndim != 2:
                raise ValueError(f"이미지 텐서는 2D (N, feat_dim)이어야 합니다. got shape={img.shape}")
            if self.expected_feat_dim is not None and img.shape[1] != self.expected_feat_dim:
                raise ValueError(
                    f"feat_dim 불일치: 기대값={self.expected_feat_dim}, 실제={img.shape[1]}"
                )
            flat_images.append(img)

        return flat_images

    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # GRPOTrainer가 processing_class.<tokenizer 속성>에 접근하려는 경우를 위한 passthrough.
        # (예: pad_token_id는 이미 self에 있지만, decode/batch_decode/eos_token_id 등 미확인 요구사항 대비)
        return getattr(self.tokenizer, name)
