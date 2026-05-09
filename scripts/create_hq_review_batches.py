from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create high-quality dataset review batches.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--train-batch-size", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=75)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    review_path = dataset_dir / "review_queue.csv"
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    train_rows = [row for row in rows if row.get("record_type") == "text"]
    eval_rows = [row for row in rows if row.get("record_type") == "eval_text"]
    outputs: dict[str, int] = {}
    for batch_index, start in enumerate(range(0, len(train_rows), args.train_batch_size), start=1):
        batch = train_rows[start : start + args.train_batch_size]
        path = dataset_dir / f"review_batch_hq_train_{batch_index:02d}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch)
        outputs[path.name] = len(batch)
    for batch_index, start in enumerate(range(0, len(eval_rows), args.eval_batch_size), start=1):
        batch = eval_rows[start : start + args.eval_batch_size]
        path = dataset_dir / f"review_batch_hq_eval_{batch_index:02d}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch)
        outputs[path.name] = len(batch)
    for name, count in outputs.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
