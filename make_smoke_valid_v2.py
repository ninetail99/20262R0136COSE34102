import json
from pathlib import Path

train_path = Path("./data/WSI-Bench/WSI-Bench-train.json")
image_root = Path("./data/WSIBench-pt")

out_dir = Path("./data/WSI-Bench-smoke")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "WSI-Bench-train-smoke-valid-200.json"

text = train_path.read_text(encoding="utf-8")
text_strip = text.lstrip()

if text_strip.startswith("["):
    items = json.loads(text)
else:
    items = [json.loads(line) for line in text.splitlines() if line.strip()]

print(f"total samples in train: {len(items)}")

valid = []
for sample in items:
    image_rel = sample.get("image")
    if not image_rel:
        continue
    image_path = image_root / image_rel
    if image_path.exists():
        valid.append(sample)
    if len(valid) >= 200:
        break

print(f"valid samples: {len(valid)}")

with out_path.open("w", encoding="utf-8") as f:
    json.dump(valid, f, ensure_ascii=False, indent=2)

print(f"saved to: {out_path}")