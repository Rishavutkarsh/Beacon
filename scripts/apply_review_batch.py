from __future__ import annotations

import argparse
import csv
from pathlib import Path


VALID_STATUSES = {"pending", "approved", "rejected", "edit_needed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a reviewed batch CSV back into review_queue.csv.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    queue_path = dataset_dir / "review_queue.csv"
    batch_path = Path(args.batch)
    if not queue_path.exists():
        raise SystemExit(f"missing {queue_path}")
    if not batch_path.exists():
        raise SystemExit(f"missing {batch_path}")

    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        queue_rows = list(csv.DictReader(handle))
        fieldnames = list(queue_rows[0].keys()) if queue_rows else []
    with batch_path.open("r", encoding="utf-8", newline="") as handle:
        batch_rows = list(csv.DictReader(handle))

    queue_by_id = {row["record_id"]: row for row in queue_rows}
    changed = 0
    skipped = 0
    for batch_row in batch_rows:
        record_id = batch_row.get("record_id")
        status = batch_row.get("review_status", "")
        if record_id not in queue_by_id:
            skipped += 1
            continue
        if status not in VALID_STATUSES:
            raise SystemExit(f"{record_id}: invalid review_status {status!r}")
        if status == "approved" and batch_row.get("source_check_status") != "approved":
            raise SystemExit(f"{record_id}: approved rows must also have source_check_status=approved")
        if status == "approved" and batch_row.get("image_license_check_status") == "pending":
            raise SystemExit(f"{record_id}: approved rows must resolve image_license_check_status")
        target = queue_by_id[record_id]
        for field in fieldnames:
            if field in batch_row:
                target[field] = batch_row[field]
        changed += 1

    if not args.dry_run:
        with queue_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(queue_rows)
    print(f"{'would update' if args.dry_run else 'updated'} {changed} row(s), skipped {skipped}")


if __name__ == "__main__":
    main()
