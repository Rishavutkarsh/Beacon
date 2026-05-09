from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt(row: dict) -> str:
    return row.get("user_prompt") or row.get("prompt") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Sankat Saathi dataset quality report.")
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    text = read_jsonl(dataset_dir / "sft_text.jsonl")
    vision = read_jsonl(dataset_dir / "sft_vision.jsonl")
    dpo = read_jsonl(dataset_dir / "dpo_pairs.jsonl")
    eval_rows = read_jsonl(dataset_dir / "eval.jsonl")
    train = [*text, *vision]
    train_prompts = Counter(prompt(row) for row in train)
    eval_prompts = Counter(prompt(row) for row in eval_rows)
    review_counts = Counter()
    review_path = dataset_dir / "review_queue.csv"
    if review_path.exists():
        with review_path.open("r", encoding="utf-8", newline="") as handle:
            review_counts = Counter(row.get("review_status", "") for row in csv.DictReader(handle))
    report = {
        "counts": {"sft_text": len(text), "sft_vision": len(vision), "dpo_pairs": len(dpo), "eval": len(eval_rows)},
        "unique_prompts": {"train": len(train_prompts), "eval": len(eval_prompts)},
        "duplicate_ratio": {"train": 1 - (len(train_prompts) / len(train) if train else 0), "eval": 1 - (len(eval_prompts) / len(eval_rows) if eval_rows else 0)},
        "train_eval_overlap": len(set(train_prompts).intersection(eval_prompts)),
        "risk_tags": Counter(tag for row in [*train, *dpo, *eval_rows] for tag in row.get("risk_tags", [])).most_common(),
        "language_mix": Counter(row.get("language_mix", "dpo") for row in [*text, *vision, *eval_rows]).most_common(),
        "source_coverage": Counter(source for row in [*train, *dpo, *eval_rows] for source in row.get("source_ids", [])).most_common(),
        "image_license_status": Counter((row.get("image_metadata") or {}).get("license", "not_vision") for row in [*vision, *eval_rows]).most_common(),
        "image_unique_ids": len({(row.get("image_metadata") or {}).get("image_id") for row in [*vision, *eval_rows] if row.get("modality") == "vision"}),
        "image_manifest_ready": Counter(str((row.get("image_metadata") or {}).get("manifest_ready", "not_vision")) for row in [*vision, *eval_rows]).most_common(),
        "eval_rubric_rows": sum(1 for row in eval_rows if row.get("eval_rubric")),
        "review_status": dict(review_counts),
        "dpo_failure_modes": Counter(row.get("target_failure_mode", "") for row in dpo).most_common(),
    }
    out = dataset_dir / "dataset_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
