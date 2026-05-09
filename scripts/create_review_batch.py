from __future__ import annotations

import argparse
import csv
from pathlib import Path


RECORD_TYPES_BY_MODE = {
    "text": {"text", "dpo", "eval_text"},
    "vision": {"vision", "eval_vision"},
    "full": {"text", "vision", "dpo", "eval_text", "eval_vision"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small mode-specific review batch CSV.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--mode", choices=["text", "vision", "full"], default="text")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    review_path = dataset_dir / "review_queue.csv"
    if not review_path.exists():
        raise SystemExit(f"missing {review_path}")
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    wanted_types = RECORD_TYPES_BY_MODE[args.mode]
    batch = [row for row in rows if row.get("record_type") in wanted_types and row.get("review_status") == "pending"][: args.limit]
    out_path = Path(args.out) if args.out else dataset_dir / f"review_batch_{args.mode}_{len(batch)}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch)
    print(f"wrote {len(batch)} rows to {out_path}")


if __name__ == "__main__":
    main()
