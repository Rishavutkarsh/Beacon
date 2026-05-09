import csv
import subprocess
import sys
from pathlib import Path


def test_apply_review_batch_dry_run():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/apply_review_batch.py",
            "data/processed/hardened_text",
            "--batch",
            "data/processed/hardened_text/review_batch_text_200.csv",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "would update 200 row" in result.stdout


def test_apply_review_batch_rejects_approved_without_source_check():
    src = Path("data/processed/hardened_text/review_batch_text_200.csv")
    batch = Path("data/processed/hardened_text/test_bad_batch.csv")
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows[0]["review_status"] = "approved"
    rows[0]["source_check_status"] = "pending"
    with batch.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerow(rows[0])
    result = subprocess.run(
        [
            sys.executable,
            "scripts/apply_review_batch.py",
            "data/processed/hardened_text",
            "--batch",
            str(batch),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    batch.unlink()
    assert result.returncode != 0
    assert "source_check_status=approved" in (result.stdout + result.stderr)
