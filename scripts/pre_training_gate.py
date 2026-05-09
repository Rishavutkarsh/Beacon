from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_REVIEWERS_BY_MODE = {
    "text": {"safety_source_grounding", "training_eval_readiness"},
    "vision": {"safety_source_grounding", "multimodal_image_dataset", "training_eval_readiness"},
    "full": {"safety_source_grounding", "multimodal_image_dataset", "training_eval_readiness"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate second 3-reviewer pre-training gate receipt.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--mode", choices=["text", "vision", "full"], default="full")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    receipt_path = dataset_dir / "pre_training_review.json"
    if not receipt_path.exists():
        raise SystemExit("missing pre_training_review.json; run the second 3-reviewer review before training")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_mode = receipt.get("mode", "full")
    if receipt_mode not in {args.mode, "full"}:
        raise SystemExit(f"pre-training receipt mode {receipt_mode!r} does not satisfy requested mode {args.mode!r}")
    reviewers = {item.get("reviewer") for item in receipt.get("reviews", [])}
    missing = REQUIRED_REVIEWERS_BY_MODE[args.mode] - reviewers
    if missing:
        raise SystemExit(f"missing reviewer approvals: {sorted(missing)}")
    no_go = [item for item in receipt.get("reviews", []) if item.get("decision") != "go"]
    if no_go:
        raise SystemExit(f"pre-training review has no-go entries: {[item.get('reviewer') for item in no_go]}")
    if not receipt.get("user_approved_training"):
        raise SystemExit("pre-training receipt must include user_approved_training=true")
    print(f"pre-training gate passed ({args.mode} lane): {receipt_path}")


if __name__ == "__main__":
    main()
