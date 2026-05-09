from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_to_text(answer: dict) -> str:
    sections = [
        ("risk_level", answer.get("risk_level", "")),
        ("immediate_action", answer.get("immediate_action", [])),
        ("resource_plan", answer.get("resource_plan", [])),
        ("unsafe_items", answer.get("unsafe_items", [])),
        ("missing_information", answer.get("missing_information", [])),
        ("escalation_signs", answer.get("escalation_signs", [])),
        ("what_not_to_do", answer.get("what_not_to_do", [])),
        ("hindi_hinglish", answer.get("hindi_hinglish", [])),
        ("uncertainty_note", answer.get("uncertainty_note", "")),
    ]
    lines: list[str] = []
    for key, value in sections:
        if isinstance(value, list):
            lines.append(f"{key}: " + " | ".join(str(item) for item in value))
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def make_prompt(example: dict) -> str:
    return (
        "### Instruction\n"
        "You are Sankat Saathi, an offline crisis companion. Give conservative, practical, structured guidance.\n\n"
        f"### User\n{example['user_prompt']}\n\n"
        "### Assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tiny CPU SFT smoke dataset for Kaggle.")
    parser.add_argument("--dataset-dir", default="data/processed/hardened_text")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--out", default="kaggle/input/sankat-saathi-cpu-smoke/train.jsonl")
    parser.add_argument("--allow-unapproved-smoke", action="store_true", help="Use raw sft_text.jsonl when approved exports are not ready. Only for infrastructure smoke tests.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    approved_path = dataset_dir / "sft_text_approved.jsonl"
    raw_path = dataset_dir / "sft_text.jsonl"
    if approved_path.exists() and approved_path.stat().st_size > 0:
        rows = read_jsonl(approved_path)[: args.limit]
        source_path = approved_path
    elif args.allow_unapproved_smoke:
        rows = read_jsonl(raw_path)[: args.limit]
        source_path = raw_path
    else:
        raise SystemExit("No approved SFT rows found. Use --allow-unapproved-smoke only for infrastructure smoke tests.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "id": row["example_id"],
                "prompt": make_prompt(row),
                "response": answer_to_text(row["assistant_response"]),
                "text": make_prompt(row) + answer_to_text(row["assistant_response"]),
                "source_ids": row["source_ids"],
                "guidance_fact_ids": row["guidance_fact_ids"],
                "risk_tags": row["risk_tags"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out_path} from {source_path}")


if __name__ == "__main__":
    main()
