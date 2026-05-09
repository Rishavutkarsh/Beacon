from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SAFETY_CRITICAL_TAGS = {
    "critical",
    "electricity",
    "diabetes",
    "medicine",
    "dehydration",
    "wound",
    "carbon_monoxide",
    "landslide",
    "structural",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.lower()
    prompt = re.sub(r"\d+", "0", prompt)
    prompt = re.sub(r"[^a-z0-9]+", " ", prompt)
    return " ".join(prompt.split())


def token_set(prompt: str) -> set[str]:
    return set(normalize_prompt(prompt).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def review_rows(dataset_dir: Path) -> dict[str, dict]:
    path = dataset_dir / "review_queue.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def source_assertion_ok(row: dict) -> bool:
    return bool(row.get("source_ids")) and bool(row.get("guidance_fact_ids"))


def find_near_duplicates(rows: list[dict], threshold: float) -> list[tuple[str, str, float]]:
    prompts = [(row.get("example_id") or row.get("prompt_id") or "<missing>", token_set(row.get("user_prompt", ""))) for row in rows]
    hits: list[tuple[str, str, float]] = []
    for index, (left_id, left_tokens) in enumerate(prompts):
        for right_id, right_tokens in prompts[index + 1 :]:
            score = jaccard(left_tokens, right_tokens)
            if score >= threshold:
                hits.append((left_id, right_id, score))
                if len(hits) >= 50:
                    return hits
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset lint gate before scaled Sankat Saathi text SFT.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--train-file", default="sft_text_approved.jsonl")
    parser.add_argument("--eval-file", default="eval_text_approved.jsonl")
    parser.add_argument("--min-train", type=int, default=300)
    parser.add_argument("--min-eval", type=int, default=60)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--max-train-near-duplicates", type=int, default=30)
    parser.add_argument("--max-eval-near-duplicates", type=int, default=5)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_rows = read_jsonl(dataset_dir / args.train_file)
    eval_rows = read_jsonl(dataset_dir / args.eval_file)
    reviews = review_rows(dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if len(train_rows) < args.min_train:
        errors.append(f"approved train rows below target: {len(train_rows)} < {args.min_train}")
    if len(eval_rows) < args.min_eval:
        errors.append(f"approved eval rows below target: {len(eval_rows)} < {args.min_eval}")
    train_prompts = [normalize_prompt(row.get("user_prompt", "")) for row in train_rows]
    eval_prompts = [normalize_prompt(row.get("user_prompt", "")) for row in eval_rows]
    overlap = sorted(set(train_prompts) & set(eval_prompts))
    if overlap:
        errors.append(f"exact train/eval prompt overlap: {len(overlap)}")
    duplicate_train = sum(count - 1 for count in Counter(train_prompts).values() if count > 1)
    duplicate_eval = sum(count - 1 for count in Counter(eval_prompts).values() if count > 1)
    if duplicate_train:
        errors.append(f"exact duplicate approved train prompts: {duplicate_train}")
    if duplicate_eval:
        errors.append(f"exact duplicate approved eval prompts: {duplicate_eval}")
    train_near = find_near_duplicates(train_rows, args.near_duplicate_threshold)
    eval_near = find_near_duplicates(eval_rows, args.near_duplicate_threshold)
    if len(train_near) > args.max_train_near_duplicates:
        errors.append(f"too many near-duplicate approved train prompt pairs: {len(train_near)}")
    if len(eval_near) > args.max_eval_near_duplicates:
        errors.append(f"too many near-duplicate approved eval prompt pairs: {len(eval_near)}")
    for row in [*train_rows, *eval_rows]:
        rid = row.get("example_id")
        review = reviews.get(rid or "")
        if not review or review.get("review_status") != "approved":
            errors.append(f"{rid}: approved export row is not approved in review_queue.csv")
        if review and review.get("source_check_status") != "approved":
            errors.append(f"{rid}: source_check_status is not approved")
        if not source_assertion_ok(row):
            errors.append(f"{rid}: missing source assertion ids")
        if SAFETY_CRITICAL_TAGS & set(row.get("risk_tags", [])):
            notes = (review or {}).get("review_notes", "")
            if "review" not in notes.lower() and "approved" not in notes.lower():
                warnings.append(f"{rid}: safety-critical row has sparse reviewer notes")
    report = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "exact_train_eval_overlap": len(overlap),
        "exact_duplicate_train_prompts": duplicate_train,
        "exact_duplicate_eval_prompts": duplicate_eval,
        "near_duplicate_train_pairs_sample": train_near[:10],
        "near_duplicate_eval_pairs_sample": eval_near[:10],
        "errors": errors,
        "warnings": warnings[:50],
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
