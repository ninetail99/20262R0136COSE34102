import json
import re

INPUT_FILE = "outputs/wsi-vqa-open-question.jsonl"

SUBTYPE_KEYWORDS = [
    "Luminal A",
    "Luminal B",
    "HER2-enriched",
    "HER2 enriched",
    "Basal-like",
    "Triple-negative",
    "Normal-like",
]

def extract_gt_subtype(t_answer: str):
    for kw in SUBTYPE_KEYWORDS:
        if kw.lower() in t_answer.lower():
            return kw
    return None

def get_pred_text(sample: dict) -> str:
    for key in ("answer", "pred_answer", "text", "response"):
        if key in sample and sample[key]:
            return sample[key]
    return ""

total = 0
correct = 0
missing_gt = 0
mismatches = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)

        if sample.get("metadata") != "Subtype":
            continue

        total += 1
        gt_answer = sample.get("T-answer", "")
        pred_text = get_pred_text(sample)

        gt_subtype = extract_gt_subtype(gt_answer)
        if gt_subtype is None:
            missing_gt += 1
            continue

        is_correct = gt_subtype.lower() in pred_text.lower()
        if is_correct:
            correct += 1
        else:
            mismatches.append({
                "question_id": sample.get("question_id"),
                "gt_subtype": gt_subtype,
                "pred_text": pred_text,
            })

accuracy = correct / total if total > 0 else 0.0

print(f"Total Subtype samples: {total}")
print(f"GT subtype not recognized: {missing_gt}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2%}")

for m in mismatches[:5]:
    print(m)