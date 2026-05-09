from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def reviewed_count(dataset_dir: Path) -> int:
    review_path = dataset_dir / "review_queue.csv"
    if not review_path.exists():
        return 0
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if row.get("review_status") == "approved")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision SFT LoRA training entrypoint.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--output-dir", default="outputs/vision_sft_lora")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    subprocess.run([sys.executable, "scripts/validate_dataset.py", str(dataset_dir), "--strict", "--mode", "vision", "--task", "sft", "--review-scope", "approved"], check=True)
    subprocess.run([sys.executable, "scripts/pre_training_gate.py", str(dataset_dir), "--mode", "vision"], check=True)
    approved_path = dataset_dir / "sft_vision_approved.jsonl"
    if not approved_path.exists():
        raise SystemExit("Missing sft_vision_approved.jsonl. Run scripts/export_approved.py --mode vision after review.")
    approved = reviewed_count(dataset_dir)
    if approved == 0:
        raise SystemExit("No approved rows found in review_queue.csv. Review dataset before vision training.")
    if args.dry_run:
        print(f"dry run ok: {approved} approved rows, model={args.model_id}, output={args.output_dir}")
        return
    raise SystemExit(
        "Vision training backend is intentionally not executed yet. After review, wire this entrypoint to the selected Gemma vision fine-tuning stack."
    )


if __name__ == "__main__":
    main()
