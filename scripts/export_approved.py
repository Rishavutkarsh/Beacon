from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FILES_BY_MODE = {
    "text": {
        "sft_text": ("sft_text.jsonl", "sft_text_approved.jsonl", "example_id", None),
        "dpo": ("dpo_pairs.jsonl", "dpo_pairs_approved.jsonl", "pair_id", None),
        "eval_text": ("eval.jsonl", "eval_text_approved.jsonl", "example_id", "text"),
    },
    "vision": {
        "sft_vision": ("sft_vision.jsonl", "sft_vision_approved.jsonl", "example_id", None),
        "eval_vision": ("eval.jsonl", "eval_vision_approved.jsonl", "example_id", "vision"),
    },
    "full": {
        "sft_text": ("sft_text.jsonl", "sft_text_approved.jsonl", "example_id", None),
        "sft_vision": ("sft_vision.jsonl", "sft_vision_approved.jsonl", "example_id", None),
        "dpo": ("dpo_pairs.jsonl", "dpo_pairs_approved.jsonl", "pair_id", None),
        "eval": ("eval.jsonl", "eval_approved.jsonl", "example_id", None),
    },
}
FILES_BY_TASK = {
    ("text", "sft"): {
        "sft_text": ("sft_text.jsonl", "sft_text_approved.jsonl", "example_id", None),
        "eval_text": ("eval.jsonl", "eval_text_approved.jsonl", "example_id", "text"),
    },
    ("text", "dpo"): {
        "dpo": ("dpo_pairs.jsonl", "dpo_pairs_approved.jsonl", "pair_id", None),
        "eval_text": ("eval.jsonl", "eval_text_approved.jsonl", "example_id", "text"),
    },
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def approved_ids(dataset_dir: Path) -> set[str]:
    review_path = dataset_dir / "review_queue.csv"
    if not review_path.exists():
        raise SystemExit("review_queue.csv is missing")
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["record_id"] for row in rows if row.get("review_status") == "approved" and row.get("source_check_status") == "approved" and row.get("image_license_check_status") in {"approved", "not_applicable"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export approved-only Sankat Saathi datasets.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--mode", choices=["text", "vision", "full"], default="full")
    parser.add_argument("--task", choices=["sft", "dpo", "all"], default="all")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    ids = approved_ids(dataset_dir)
    summary = {}
    files = FILES_BY_TASK.get((args.mode, args.task), FILES_BY_MODE[args.mode])
    for key, (input_name, output_name, id_key, modality) in files.items():
        rows = read_jsonl(dataset_dir / input_name)
        if modality:
            rows = [row for row in rows if row.get("modality") == modality]
        approved = [row for row in rows if row.get(id_key) in ids]
        write_jsonl(dataset_dir / output_name, approved)
        summary[output_name] = len(approved)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
