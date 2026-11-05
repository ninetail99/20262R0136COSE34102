import json
from pathlib import Path

OUT_PATH = Path("/workspace/home/Daejeon/mmkoo/wsi-llava-resampler/WSI-LLaVA-dev/data/REG-2-smoke/reg-2-smoke.json")

# 여기에 smoke로 쓸 .pt 파일명 10개를 채우면 됨
PT_FILES = [
    "PIT_01_00019_01.pt",
    "PIT_01_00031_01.pt",
    "PIT_01_00029_01.pt",
    "PIT_01_00030_01.pt",
    "PIT_01_00028_01.pt",
    "PIT_01_00027_01.pt",
    "PIT_01_00026_01.pt",
    "PIT_01_00025_01.pt",
    "PIT_01_00021_01.pt",
    "PIT_01_00020_01.pt",
]

HUMAN_PROMPT = "<image>\nDescribe this whole slide image."
GPT_RESPONSE = "This is a placeholder response for smoke testing."

def build_record(idx: int, pt_name: str):
    return {
        "id": f"reg2_smoke_{idx:04d}",
        "image": pt_name,
        "conversations": [
            {
                "from": "human",
                "value": HUMAN_PROMPT
            },
            {
                "from": "gpt",
                "value": GPT_RESPONSE
            }
        ]
    }

def main():
    if not PT_FILES:
        raise ValueError("PT_FILES is empty. Add at least one .pt filename.")

    if len(PT_FILES) != len(set(PT_FILES)):
        raise ValueError("Duplicate filenames found in PT_FILES.")

    data = [build_record(i, pt_name) for i, pt_name in enumerate(PT_FILES, start=1)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"saved: {OUT_PATH}")
    print(f"count: {len(data)}")
    print("first record:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()