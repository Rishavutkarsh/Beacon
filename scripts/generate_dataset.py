from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.hardened import HARDENED_FACTS, make_hardened_dpo_pairs, make_hardened_text_examples, make_hardened_vision_examples
from sankat_saathi_dataset.high_quality import REFERENCE_DATASETS, INCIDENT_PATTERNS, all_high_quality_facts, all_high_quality_sources, make_high_quality_records
from sankat_saathi_dataset.manifests import load_source_manifest
from sankat_saathi_dataset.sources import GUIDANCE_FACTS, SOURCES
from sankat_saathi_dataset.templates import IMAGE_CASES, make_dpo_pairs, make_text_examples, make_vision_examples


PROFILES = {
    "tiny": {"sft_text": 24, "sft_vision": 15, "dpo": 18, "eval": 20},
    "starter": {"sft_text": 120, "sft_vision": 60, "dpo": 60, "eval": 60},
    "full": {"sft_text": 720, "sft_vision": 420, "dpo": 360, "eval": 160},
    "hardened": {"sft_text": 720, "sft_vision": 420, "dpo": 360, "eval": 160},
    "hardened_text": {"sft_text": 720, "sft_vision": 0, "dpo": 360, "eval": 160},
    "hardened_vision": {"sft_text": 0, "sft_vision": 420, "dpo": 0, "eval": 160},
    "hardened_full": {"sft_text": 720, "sft_vision": 420, "dpo": 360, "eval": 160},
    "high_quality_text": {"sft_text": 1000, "sft_vision": 0, "dpo": 0, "eval": 150},
}


def write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            item = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, text_examples: list[object], vision_examples: list[object], dpo_pairs: list[object], eval_examples: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_status",
        "record_type",
        "record_id",
        "risk_tags",
        "source_ids",
        "guidance_fact_ids",
        "image_path",
        "prompt",
        "expected_risk",
        "expected_summary",
        "hazard_category",
        "counterintuitive_mistake",
        "visual_context",
        "visual_attention_required",
        "scenario_seed_id",
        "incident_pattern_ids",
        "review_notes",
        "safety_flags",
        "source_check_status",
        "image_license_check_status",
    ]
    rows = []
    for example in [*text_examples, *vision_examples, *eval_examples]:
        item = asdict(example) if hasattr(example, "__dataclass_fields__") else example
        answer = item["assistant_response"]
        immediate = answer["immediate_action"] if isinstance(answer, dict) else answer.immediate_action
        risk_level = answer["risk_level"] if isinstance(answer, dict) else answer.risk_level
        rows.append(
            {
                "review_status": "pending",
                "record_type": "eval_" + item["modality"] if item["split"] == "eval" else item["modality"],
                "record_id": item["example_id"],
                "risk_tags": "|".join(item["risk_tags"]),
                "source_ids": "|".join(item["source_ids"]),
                "guidance_fact_ids": "|".join(item["guidance_fact_ids"]),
                "image_path": item.get("image_path") or "",
                "prompt": item["user_prompt"],
                "expected_risk": risk_level,
                "expected_summary": " / ".join(immediate[:2]),
                "hazard_category": item.get("hazard_category", ""),
                "counterintuitive_mistake": item.get("counterintuitive_mistake", ""),
                "visual_context": item.get("visual_context", ""),
                "visual_attention_required": item.get("visual_attention_required", ""),
                "scenario_seed_id": item.get("scenario_seed_id", ""),
                "incident_pattern_ids": "|".join(item.get("incident_pattern_ids", [])),
                "review_notes": "",
                "safety_flags": "",
                "source_check_status": "pending",
                "image_license_check_status": "pending" if item.get("image_path") else "not_applicable",
            }
        )
    for pair in dpo_pairs:
        item = asdict(pair) if hasattr(pair, "__dataclass_fields__") else pair
        chosen = item["chosen"]
        chosen_risk = chosen["risk_level"] if isinstance(chosen, dict) else chosen.risk_level
        rows.append(
            {
                "review_status": "pending",
                "record_type": "dpo",
                "record_id": item["pair_id"],
                "risk_tags": "|".join(item["risk_tags"]),
                "source_ids": "|".join(item["source_ids"]),
                "guidance_fact_ids": "|".join(item["guidance_fact_ids"]),
                "image_path": item.get("image_path") or "",
                "prompt": item["prompt"],
                "expected_risk": chosen_risk,
                "expected_summary": "REJECTED: " + item["rejected"][:140],
                "hazard_category": item.get("hazard_category", ""),
                "counterintuitive_mistake": item.get("counterintuitive_mistake", ""),
                "visual_context": "",
                "visual_attention_required": "",
                "scenario_seed_id": "",
                "incident_pattern_ids": "|".join(item.get("incident_pattern_ids", [])),
                "review_notes": "",
                "safety_flags": "",
                "source_check_status": "pending",
                "image_license_check_status": "pending" if item.get("image_path") else "not_applicable",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_image_cards() -> None:
    out_dir = ROOT / "data" / "images" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "cloudy_water_bottle": ("#7aa7b7", "#f3f8fa"),
        "dented_can": ("#a9b0b8", "#fff7e8"),
        "medicine_strip": ("#78a77a", "#f5fff4"),
        "wet_rice_bag": ("#d4b56a", "#fffbed"),
        "power_cable_water": ("#30343b", "#eef7ff"),
    }
    for case in IMAGE_CASES:
        image_id = case["image_id"]
        fg, bg = colors[image_id]
        observations = "".join(f"<text x='36' y='{190 + i * 28}'>{obs}</text>" for i, obs in enumerate(case["observations"]))
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
  <rect width="640" height="420" fill="{bg}"/>
  <rect x="28" y="28" width="584" height="120" rx="8" fill="{fg}" opacity="0.9"/>
  <text x="36" y="78" font-family="Arial, sans-serif" font-size="32" fill="white">{image_id.replace('_', ' ').title()}</text>
  <text x="36" y="118" font-family="Arial, sans-serif" font-size="18" fill="white">Synthetic review card; replace with open-license photo before final training.</text>
  <g font-family="Arial, sans-serif" font-size="20" fill="#1f2933">{observations}</g>
  <text x="36" y="380" font-family="Arial, sans-serif" font-size="16" fill="#52606d">Visible observations are labels, not safety proof.</text>
</svg>
"""
        (ROOT / case["image_path"]).write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sankat Saathi reviewable datasets.")
    parser.add_argument("--profile", choices=PROFILES, default="starter")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    counts = PROFILES[args.profile]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "processed" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.profile == "high_quality_text":
        train_text, eval_text = make_high_quality_records(counts["sft_text"], counts["eval"])
        train_vision = []
        eval_vision = []
        dpo_pairs = []
        guidance_facts = all_high_quality_facts()
        source_records = all_high_quality_sources()
        notes = "High-quality text plus visual-attention dataset. Public crisis datasets and incident reports are reference material only; rows are anonymized/source-grounded synthetic scenarios."
    elif args.profile in {"hardened", "hardened_text", "hardened_vision", "hardened_full"}:
        train_text = make_hardened_text_examples(counts["sft_text"], split="train")
        train_vision = make_hardened_vision_examples(counts["sft_vision"], split="train")
        dpo_pairs = make_hardened_dpo_pairs(counts["dpo"])
        if args.profile == "hardened_text":
            eval_text = make_hardened_text_examples(counts["eval"], split="eval")
            eval_vision = []
        elif args.profile == "hardened_vision":
            eval_text = []
            eval_vision = make_hardened_vision_examples(counts["eval"], split="eval")
        else:
            eval_text = make_hardened_text_examples(int(counts["eval"] * 0.6), split="eval")
            eval_vision = make_hardened_vision_examples(counts["eval"] - len(eval_text), split="eval")
        guidance_facts = HARDENED_FACTS
        source_records = [*SOURCES, *load_source_manifest()]
        notes = {
            "hardened_text": "Text-only hardened dataset. Vision rows are excluded so image placeholders do not block text review/training gates.",
            "hardened_vision": "Vision-only hardened dataset. Requires real verified images before strict validation and training.",
            "hardened_full": "Full hardened dataset. Requires both text and verified vision gates before training.",
            "hardened": "Full hardened dataset. Requires both text and verified vision gates before training.",
        }[args.profile]
    else:
        write_image_cards()
        train_text = make_text_examples(counts["sft_text"], split="train")
        train_vision = make_vision_examples(counts["sft_vision"], split="train")
        dpo_pairs = make_dpo_pairs(counts["dpo"])
        eval_text = make_text_examples(int(counts["eval"] * 0.6), split="eval")
        eval_vision = make_vision_examples(counts["eval"] - len(eval_text), split="eval")
        guidance_facts = GUIDANCE_FACTS
        source_records = SOURCES
        notes = "Generated image cards are placeholders for review and should be replaced or supplemented with open-license photos before final multimodal training."

    write_jsonl(out_dir / "sources.jsonl", source_records)
    write_jsonl(out_dir / "guidance_facts.jsonl", guidance_facts)
    write_jsonl(out_dir / "sft_text.jsonl", train_text)
    write_jsonl(out_dir / "sft_vision.jsonl", train_vision)
    write_jsonl(out_dir / "dpo_pairs.jsonl", dpo_pairs)
    write_jsonl(out_dir / "eval.jsonl", [*eval_text, *eval_vision])
    if args.profile == "high_quality_text":
        write_jsonl(out_dir / "reference_datasets.jsonl", REFERENCE_DATASETS)
        write_jsonl(out_dir / "incident_patterns.jsonl", INCIDENT_PATTERNS)
    eval_rows = [*eval_text, *eval_vision]
    write_review_csv(out_dir / "review_queue.csv", train_text, train_vision, dpo_pairs, eval_rows)

    manifest = {
        "profile": args.profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "sources": len(source_records),
            "guidance_facts": len(guidance_facts),
            "sft_text": len(train_text),
            "sft_vision": len(train_vision),
            "dpo_pairs": len(dpo_pairs),
            "eval": len(eval_rows),
            "review_queue": len(train_text) + len(train_vision) + len(dpo_pairs) + len(eval_rows),
        },
        "review_required_before_training": True,
        "pre_training_review_required": True,
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
