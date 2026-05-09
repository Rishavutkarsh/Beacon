import csv
import json
import subprocess
import sys
from pathlib import Path


def test_text_strict_failure_does_not_mention_images():
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py", "data/processed/hardened_text", "--strict", "--mode", "text"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "approved export is empty" in output
    assert "image file does not exist" not in output
    assert "manifest-ready images" not in output


def test_export_text_mode_writes_text_eval_file():
    subprocess.run(
        [sys.executable, "scripts/export_approved.py", "data/processed/hardened_text", "--mode", "text"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert Path("data/processed/hardened_text/eval_text_approved.jsonl").exists()


def test_review_batch_text_contains_only_text_lane_types():
    out = Path("data/processed/hardened_text/test_review_batch_text_25.csv")
    if out.exists():
        out.unlink()
    subprocess.run(
        [
            sys.executable,
            "scripts/create_review_batch.py",
            "data/processed/hardened_text",
            "--mode",
            "text",
            "--limit",
            "25",
            "--out",
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 25
    assert {row["record_type"] for row in rows}.issubset({"text", "dpo", "eval_text"})
    out.unlink()
