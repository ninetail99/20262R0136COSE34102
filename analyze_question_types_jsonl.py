#!/usr/bin/env python3
"""JSONL 추론 출력에서 질문 표현의 다양성과 GT 질문 목록과의 차이를 분석한다.

JSONL 예시 (모델 추론 출력):
    {
      "question_id": "PIT_01_00050_01",
      "image": "PIT_01_00050_01.pt",
      "question": "Write a report about this image",
      "answer": "question: What is the procedure?\\nanswer: Core needle biopsy\\n...",
      "T-answer": "question: What is the procedure?\\nanswer: Core needle biopsy\\n...",
      "metadata": "general"
    }

기본 실행:
    python analyze_question_types_jsonl.py --input predictions.jsonl

필요하면 --answer-key, --id-key 로 필드명을 바꿀 수 있다.
분석 결과는 터미널에 출력하고, --output-dir 를 지정하면 JSON/CSV 도 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

QAPair = Tuple[str, str]

FINAL_REPORT_RE = re.compile(
    r"(?:^|\n)question:\s*"
    r"what is the final(?: pathology)? report\?\s*"
    r"answer:\s*",
    flags=re.IGNORECASE,
)

QA_RE = re.compile(
    r"question:\s*(.*?)\n"
    r"answer:\s*(.*?)(?=\nquestion:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_text(text: Any) -> str:
    """비교용 정규화: 대소문자, 공백, 줄바꿈, 가장자리 구두점을 안정화한다."""
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_question(text: Any) -> str:
    """질문 표현 비교용 정규화.

    의미를 임의로 합치지 않도록 단어 삭제나 동의어 치환은 하지 않는다.
    따라서 결과의 `exact_normalized` 는 안전한 표면형 비교용이다.
    """
    q = normalize_text(text)
    q = re.sub(r"\s+", " ", q)
    return q


def parse_qa_pairs(text: str, include_final_report: bool = False) -> List[QAPair]:
    """question:/answer: 형식의 QA pair 를 추출한다."""
    text = str(text or "")
    if not include_final_report:
        match = FINAL_REPORT_RE.search(text)
        if match:
            text = text[: match.start()].strip()

    return [
        (question.strip(), answer.strip())
        for question, answer in QA_RE.findall(text)
        if question.strip() or answer.strip()
    ]


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARNING] line {line_number}: JSON parsing 실패: {exc}", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                print(f"[WARNING] line {line_number}: JSON object 가 아닙니다.", file=sys.stderr)
                continue
            yield line_number, obj


def load_gt_questions(path: Optional[Path]) -> Dict[str, str]:
    """GT 질문 목록을 JSON/JSONL/텍스트에서 읽어 canonical 목록으로 만든다."""
    if path is None:
        return {}

    questions: List[str] = []
    if path.suffix.lower() == ".jsonl":
        for _, obj in iter_jsonl(path):
            for key in ("question", "questions", "unique_questions"):
                if key in obj:
                    value = obj[key]
                    if isinstance(value, list):
                        questions.extend(str(v).strip() for v in value)
                    else:
                        questions.append(str(value).strip())
                    break
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            value = obj.get("unique_questions", obj.get("questions", obj.get("question", [])))
        else:
            value = obj
        if isinstance(value, list):
            questions.extend(str(v).strip() for v in value)
        else:
            questions.append(str(value).strip())
    else:
        questions = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]

    result = {}
    for question in questions:
        normalized = normalize_question(question)
        if normalized:
            result[normalized] = question.strip()
    return result


def classify_answer(answer: str) -> str:
    value = normalize_text(answer)
    if re.match(r"^(yes|no)\b", value):
        return "yes_no"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return "numeric"
    if len(value) > 150 or "\n" in answer:
        return "long_multiline"
    return "short_phrase"


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    gt_questions = load_gt_questions(Path(args.gt_questions) if args.gt_questions else None)
    question_counts: Counter[str] = Counter()
    question_examples: Dict[str, List[str]] = defaultdict(list)
    answer_types: Dict[str, Counter[str]] = defaultdict(Counter)
    raw_forms: Dict[str, Counter[str]] = defaultdict(Counter)
    unknown_question_counts: Counter[str] = Counter()
    malformed_records = 0
    total_records = 0
    total_pairs = 0

    for line_number, obj in iter_jsonl(Path(args.input)):
        total_records += 1
        value = obj.get(args.answer_key)
        if value is None:
            # 흔한 대체 필드도 자동 탐색
            for key in ("response", "completion", "output", "generated_text", "text"):
                if key in obj:
                    value = obj[key]
                    break
        if value is None:
            malformed_records += 1
            print(f"[WARNING] line {line_number}: answer 필드를 찾지 못했습니다.", file=sys.stderr)
            continue

        pairs = parse_qa_pairs(str(value), include_final_report=args.include_final_report)
        if not pairs:
            malformed_records += 1
            continue

        for question, answer in pairs:
            q_norm = normalize_question(question)
            if not q_norm:
                continue
            total_pairs += 1
            question_counts[q_norm] += 1
            raw_forms[q_norm][question.strip()] += 1
            answer_types[q_norm][classify_answer(answer)] += 1
            if len(question_examples[q_norm]) < args.examples:
                question_examples[q_norm].append(question.strip())
            if gt_questions and q_norm not in gt_questions:
                unknown_question_counts[q_norm] += 1

    unique_count = len(question_counts)
    known_count = sum(q in gt_questions for q in question_counts) if gt_questions else None
    unknown_count = len(unknown_question_counts) if gt_questions else None

    result = {
        "input": str(args.input),
        "gt_questions": str(args.gt_questions) if args.gt_questions else None,
        "total_records": total_records,
        "malformed_records": malformed_records,
        "total_qa_pairs": total_pairs,
        "unique_questions": unique_count,
        "known_unique_questions": known_count,
        "unknown_unique_questions": unknown_count,
        "unknown_pair_count": sum(unknown_question_counts.values()) if gt_questions else None,
        "question_counts": dict(question_counts),
        "unknown_question_counts": dict(unknown_question_counts),
        "question_examples": dict(question_examples),
        "question_raw_forms": {k: dict(v) for k, v in raw_forms.items()},
        "question_to_answer_types": {k: dict(v) for k, v in answer_types.items()},
    }
    return result


def write_outputs(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "question_type_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "question_type_counts.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["question_normalized", "count", "known_in_gt", "raw_forms", "answer_types"])
        gt_questions = load_gt_questions(Path(result["gt_questions"])) if result["gt_questions"] else {}
        for q, count in sorted(result["question_counts"].items(), key=lambda x: (-x[1], x[0])):
            writer.writerow([
                q,
                count,
                q in gt_questions if gt_questions else "",
                json.dumps(result["question_raw_forms"].get(q, {}), ensure_ascii=False),
                json.dumps(result["question_to_answer_types"].get(q, {}), ensure_ascii=False),
            ])

    unknown = result.get("unknown_question_counts") or {}
    with (output_dir / "unknown_questions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["question_normalized", "count", "examples"])
        for q, count in sorted(unknown.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow([q, count, " || ".join(result["question_examples"].get(q, []))])


def print_report(result: Dict[str, Any], top_k: int) -> None:
    print("=" * 72)
    print(f"입력 파일                 : {result['input']}")
    print(f"전체 JSONL record 수       : {result['total_records']}")
    print(f"파싱 실패/QA 없음 record 수: {result['malformed_records']}")
    print(f"전체 QA pair 수            : {result['total_qa_pairs']}")
    print(f"고유 질문 수               : {result['unique_questions']}")
    if result["gt_questions"]:
        print(f"GT 목록 파일               : {result['gt_questions']}")
        print(f"GT 목록에 있는 고유 질문  : {result['known_unique_questions']}")
        print(f"GT 목록에 없는 고유 질문  : {result['unknown_unique_questions']}")
        print(f"GT 목록에 없는 QA pair 수 : {result['unknown_pair_count']}")
    print("-" * 72)
    print(f"등장 빈도 상위 {top_k}개 질문")
    for question, count in sorted(result["question_counts"].items(), key=lambda x: (-x[1], x[0]))[:top_k]:
        types = result["question_to_answer_types"].get(question, {})
        flag = " [GT 밖]" if question in (result.get("unknown_question_counts") or {}) else ""
        print(f"[{count:6d}] {question}{flag}")
        print(f"         answer types: {types}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="분석할 JSONL 파일")
    parser.add_argument("--answer-key", default="answer", help="생성 텍스트가 담긴 JSON key")
    parser.add_argument("--id-key", default="question_id", help="샘플 ID key (현재 출력/검증용)")
    parser.add_argument("--gt-questions", default=None, help="GT 질문 목록 JSON/JSONL/TXT 파일")
    parser.add_argument("--output-dir", default=None, help="JSON/CSV 결과를 저장할 디렉터리")
    parser.add_argument("--top-k", type=int, default=40, help="터미널에 표시할 상위 질문 수")
    parser.add_argument("--examples", type=int, default=5, help="질문별 원문 예시 수")
    parser.add_argument("--include-final-report", action="store_true", help="final pathology report 도 포함")
    args = parser.parse_args()

    result = analyze(args)
    print_report(result, args.top_k)
    if args.output_dir:
        write_outputs(result, Path(args.output_dir))
        print(f"결과 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()