from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark a small text SFT/eval subset approved after reviewer clearance.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--review-note", default="Approved for tiny text SFT smoke after two reviewer content checks.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_ids = [row["example_id"] for row in read_jsonl(dataset_dir / "sft_text.jsonl")[: args.train_count]]
    eval_ids = [row["example_id"] for row in read_jsonl(dataset_dir / "eval.jsonl") if row.get("modality") == "text"][: args.eval_count]
    approve_ids = set(train_ids + eval_ids)
    review_path = dataset_dir / "review_queue.csv"
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    changed = 0
    for row in rows:
        if row.get("record_id") in approve_ids:
            row["review_status"] = "approved"
            row["source_check_status"] = "approved"
            row["image_license_check_status"] = "not_applicable"
            row["review_notes"] = args.review_note
            changed += 1
    if changed != len(approve_ids):
        missing = approve_ids - {row.get("record_id") for row in rows}
        raise SystemExit(f"expected to approve {len(approve_ids)} rows but changed {changed}; missing={sorted(missing)[:10]}")
    if not args.dry_run:
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{'would approve' if args.dry_run else 'approved'} {changed} rows ({len(train_ids)} train, {len(eval_ids)} eval)")


if __name__ == "__main__":
    main()
