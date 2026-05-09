from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.quality import unsafe_hits, validate_example

DATASET_FILES = ["sft_text.jsonl", "sft_vision.jsonl", "dpo_pairs.jsonl", "eval.jsonl"]
APPROVED_FILES_BY_MODE = {
    "text": ["sft_text_approved.jsonl", "dpo_pairs_approved.jsonl", "eval_text_approved.jsonl"],
    "vision": ["sft_vision_approved.jsonl", "eval_vision_approved.jsonl"],
    "full": ["sft_text_approved.jsonl", "sft_vision_approved.jsonl", "dpo_pairs_approved.jsonl", "eval_approved.jsonl"],
}
APPROVED_FILES_BY_TASK = {
    ("text", "sft"): ["sft_text_approved.jsonl", "eval_text_approved.jsonl"],
    ("text", "dpo"): ["dpo_pairs_approved.jsonl", "eval_text_approved.jsonl"],
    ("text", "all"): APPROVED_FILES_BY_MODE["text"],
    ("vision", "sft"): ["sft_vision_approved.jsonl", "eval_vision_approved.jsonl"],
    ("vision", "all"): APPROVED_FILES_BY_MODE["vision"],
    ("full", "all"): APPROVED_FILES_BY_MODE["full"],
}
PLACEHOLDER_VALUES = {"", "TBD", "unknown", "REPLACE_WITH_VERIFIED_URL", "REPLACE_WITH_VERIFIED_LICENSE", "REPLACE_WITH_LICENSE_URL"}
OPEN_LICENSE_HINTS = ("public domain", "cc0", "cc-by", "cc by", "creative commons", "us government", "government works", "open government")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(record: dict) -> str:
    return record.get("example_id") or record.get("pair_id") or "<missing-id>"


def prompt_of(record: dict) -> str:
    return record.get("user_prompt") or record.get("prompt") or ""


def validate_guidance_facts(dataset_dir: Path, strict: bool) -> list[str]:
    errors: list[str] = []
    facts = read_jsonl(dataset_dir / "guidance_facts.jsonl")
    sources = {source.get("source_id"): source for source in read_jsonl(dataset_dir / "sources.jsonl")}
    required = ["accessed_at", "published_at", "source_section", "jurisdiction", "evidence_notes"]
    for fact in facts:
        missing_sources = [source_id for source_id in fact.get("source_ids", []) if source_id not in sources]
        if missing_sources:
            errors.append(f"{fact.get('fact_id')}: source_ids missing from sources.jsonl: {missing_sources}")
        for key in required:
            if strict and not fact.get(key):
                errors.append(f"{fact.get('fact_id')}: missing source provenance {key}")
        if strict and not fact.get("source_ready"):
            errors.append(f"{fact.get('fact_id')}: source_ready must be true for strict validation")
    return errors


def validate_image_manifest(dataset_dir: Path, strict: bool, mode: str) -> list[str]:
    if not strict or mode == "text":
        return []
    errors: list[str] = []
    manifest_path = ROOT / "data" / "images" / "verified" / "images_manifest.csv"
    if not manifest_path.exists():
        return [f"missing image manifest: {manifest_path}"]
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ready = [row for row in rows if row.get("manifest_ready", "").lower() == "true"]
    if len(ready) < 60:
        errors.append(f"strict mode requires at least 60 manifest-ready images; found {len(ready)}")
    train_ids = {row["image_id"] for row in rows if row.get("split_group") in {"train", "shared"} and row.get("manifest_ready", "").lower() == "true"}
    eval_ids = {row["image_id"] for row in rows if row.get("split_group") in {"eval", "shared"} and row.get("manifest_ready", "").lower() == "true"}
    overlap = train_ids.intersection(eval_ids)
    if overlap:
        errors.append(f"train/eval image identity overlap in manifest: {sorted(overlap)[:10]}")
    for row in ready:
        license_text = row.get("license", "").lower()
        if not any(hint in license_text for hint in OPEN_LICENSE_HINTS):
            errors.append(f"{row.get('image_id')}: unclear image license: {row.get('license')}")
    return errors


def validate_images(records: list[dict], strict: bool, mode: str) -> list[str]:
    errors: list[str] = []
    if mode == "text":
        return errors
    for record in records:
        if record.get("modality") != "vision":
            continue
        rid = record_id(record)
        image_path = record.get("image_path")
        if not image_path:
            errors.append(f"{rid}: missing image_path")
            continue
        if strict:
            if image_path.lower().endswith(".svg") or "generated" in image_path.replace("\\", "/"):
                errors.append(f"{rid}: SVG/generated placeholder image is not allowed in strict mode")
            abs_path = ROOT / image_path
            if not abs_path.exists():
                errors.append(f"{rid}: image file does not exist: {image_path}")
        meta = record.get("image_metadata")
        if strict and not meta:
            errors.append(f"{rid}: missing image_metadata")
            continue
        if strict and meta:
            for key in ["source_url", "license", "license_url", "author", "retrieved_at", "local_path", "event_id", "hazard_type"]:
                if meta.get(key) in PLACEHOLDER_VALUES:
                    errors.append(f"{rid}: image metadata placeholder for {key}")
            if not meta.get("manifest_ready"):
                errors.append(f"{rid}: image metadata manifest_ready must be true")
            license_text = str(meta.get("license", "")).lower()
            if not any(hint in license_text for hint in OPEN_LICENSE_HINTS):
                errors.append(f"{rid}: image license is not clearly open/public-domain: {meta.get('license')}")
            if not meta.get("visible_labels") or not meta.get("not_determinable_labels"):
                errors.append(f"{rid}: image metadata needs visible and not_determinable labels")
    return errors


def validate_splits(dataset_dir: Path, strict: bool, mode: str) -> list[str]:
    errors: list[str] = []
    text_train = read_jsonl(dataset_dir / "sft_text.jsonl")
    vision_train = read_jsonl(dataset_dir / "sft_vision.jsonl")
    eval_all = read_jsonl(dataset_dir / "eval.jsonl")
    if mode == "text":
        train = text_train
        eval_rows = [row for row in eval_all if row.get("modality") == "text"]
    elif mode == "vision":
        train = vision_train
        eval_rows = [row for row in eval_all if row.get("modality") == "vision"]
    else:
        train = [*text_train, *vision_train]
        eval_rows = eval_all
    train_prompts = Counter(prompt_of(row) for row in train)
    eval_prompts = Counter(prompt_of(row) for row in eval_rows)
    leaked = sorted(set(train_prompts).intersection(eval_prompts))
    if leaked:
        errors.append(f"train/eval prompt overlap: {len(leaked)} exact prompt(s)")
    if strict and train:
        unique_ratio = len(train_prompts) / len(train)
        if unique_ratio < 0.30:
            errors.append(f"unique train prompt ratio too low: {unique_ratio:.3f}")
        if mode in {"vision", "full"}:
            train_image_ids = {((row.get("image_metadata") or {}).get("image_id")) for row in train if row.get("modality") == "vision"}
            eval_image_ids = {((row.get("image_metadata") or {}).get("image_id")) for row in eval_rows if row.get("modality") == "vision"}
            train_image_ids.discard(None)
            eval_image_ids.discard(None)
            image_overlap = train_image_ids.intersection(eval_image_ids)
            if image_overlap:
                errors.append(f"train/eval image identity overlap: {sorted(image_overlap)[:10]}")
    return errors


def review_relevant_record_ids(dataset_dir: Path, mode: str, task: str) -> set[str]:
    eval_rows = read_jsonl(dataset_dir / "eval.jsonl")
    ids: set[str] = set()
    if mode in {"text", "full"}:
        if task in {"sft", "all"}:
            ids.update(record_id(row) for row in read_jsonl(dataset_dir / "sft_text.jsonl"))
        if task in {"dpo", "all"}:
            ids.update(record_id(row) for row in read_jsonl(dataset_dir / "dpo_pairs.jsonl"))
        ids.update(record_id(row) for row in eval_rows if row.get("modality") == "text")
    if mode in {"vision", "full"}:
        ids.update(record_id(row) for row in read_jsonl(dataset_dir / "sft_vision.jsonl"))
        ids.update(record_id(row) for row in eval_rows if row.get("modality") == "vision")
    return ids


def approved_file_names(mode: str, task: str) -> list[str]:
    return APPROVED_FILES_BY_TASK.get((mode, task), APPROVED_FILES_BY_MODE[mode])


def approved_export_ids(dataset_dir: Path, mode: str, task: str) -> set[str]:
    ids: set[str] = set()
    file_names = approved_file_names(mode, task)
    id_key_by_file = {
        "sft_text_approved.jsonl": "example_id",
        "sft_vision_approved.jsonl": "example_id",
        "dpo_pairs_approved.jsonl": "pair_id",
        "eval_text_approved.jsonl": "example_id",
        "eval_vision_approved.jsonl": "example_id",
        "eval_approved.jsonl": "example_id",
    }
    for name in file_names:
        for row in read_jsonl(dataset_dir / name):
            key = id_key_by_file[name]
            if row.get(key):
                ids.add(row[key])
    return ids


def validate_review(dataset_dir: Path, strict: bool, mode: str, review_scope: str, task: str) -> list[str]:
    errors: list[str] = []
    review_path = dataset_dir / "review_queue.csv"
    if not review_path.exists():
        return ["missing review_queue.csv"]
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids_in_review = {row.get("record_id") for row in rows}
    relevant_ids = review_relevant_record_ids(dataset_dir, mode, task)
    if strict and review_scope == "approved":
        exported_ids = approved_export_ids(dataset_dir, mode, task)
        if not exported_ids:
            errors.append("review_scope=approved requires non-empty approved exports")
        relevant_ids = exported_ids
    missing = [rid for rid in relevant_ids if rid not in ids_in_review]
    if missing:
        errors.append(f"review queue missing {len(missing)} trainable record(s)")
    if strict:
        relevant_rows = [row for row in rows if row.get("record_id") in relevant_ids]
        bad_status = [row.get("record_id") for row in relevant_rows if row.get("review_status") not in {"approved", "rejected", "edit_needed"}]
        if bad_status:
            errors.append(f"strict mode requires completed review_status for all rows; pending/invalid rows: {len(bad_status)}")
        for row in relevant_rows:
            if row.get("review_status") == "approved" and row.get("source_check_status") != "approved":
                errors.append(f"{row.get('record_id')}: approved row needs approved source_check_status")
            if row.get("review_status") == "approved" and row.get("image_license_check_status") == "pending":
                errors.append(f"{row.get('record_id')}: approved row needs image license check")
    return errors


def validate_dpo(records: list[dict], strict: bool, mode: str, task: str) -> list[str]:
    if mode == "vision" or task == "sft":
        return []
    errors: list[str] = []
    modes = Counter()
    for record in records:
        rid = record_id(record)
        mode = record.get("target_failure_mode")
        if strict and not mode:
            errors.append(f"{rid}: missing target_failure_mode")
        if mode:
            modes[mode] += 1
        if not record.get("rejection_reasons"):
            errors.append(f"{rid}: missing rejection_reasons")
        if strict and not unsafe_hits(record.get("rejected", "")):
            reasons = " ".join(record.get("rejection_reasons", []))
            if not any(key in reasons for key in ["overconfidence", "unsafe", "medical", "image", "vague", "missed"]):
                errors.append(f"{rid}: rejected answer lacks detectable failure signal")
    if strict and records and len(modes) < 4:
        errors.append("DPO target_failure_mode coverage too narrow")
    return errors


def validate_approved_exports(dataset_dir: Path, strict: bool, mode: str, task: str) -> list[str]:
    if not strict:
        return []
    errors: list[str] = []
    for name in approved_file_names(mode, task):
        path = dataset_dir / name
        if not path.exists():
            errors.append(f"missing approved export {name}")
        elif not read_jsonl(path):
            errors.append(f"approved export is empty: {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated Sankat Saathi dataset files.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--strict", action="store_true", help="Require training-ready provenance, review, image, split, and approved-export gates.")
    parser.add_argument("--mode", choices=["text", "vision", "full"], default="full")
    parser.add_argument("--task", choices=["sft", "dpo", "all"], default="all")
    parser.add_argument("--review-scope", choices=["all", "approved"], default="all", help="all requires every relevant review row to be completed; approved validates only exported approved rows.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    total_errors: list[str] = []
    all_examples: list[dict] = []
    for name in DATASET_FILES:
        path = dataset_dir / name
        if not path.exists():
            total_errors.append(f"missing {path}")
            continue
        rows = read_jsonl(path)
        all_examples.extend(rows)
        for index, record in enumerate(rows):
            errors = validate_example(record)
            for error in errors:
                total_errors.append(f"{record_id(record)}:{index}: {error}")

    dpo_rows = read_jsonl(dataset_dir / "dpo_pairs.jsonl")
    total_errors.extend(validate_guidance_facts(dataset_dir, args.strict))
    total_errors.extend(validate_image_manifest(dataset_dir, args.strict, args.mode))
    total_errors.extend(validate_images(all_examples, args.strict, args.mode))
    total_errors.extend(validate_splits(dataset_dir, args.strict, args.mode))
    total_errors.extend(validate_review(dataset_dir, args.strict, args.mode, args.review_scope, args.task))
    total_errors.extend(validate_dpo(dpo_rows, args.strict, args.mode, args.task))
    total_errors.extend(validate_approved_exports(dataset_dir, args.strict, args.mode, args.task))

    if total_errors:
        for error in total_errors[:80]:
            print(f"ERROR {error}")
        if len(total_errors) > 80:
            print(f"... {len(total_errors) - 80} more error(s)")
        raise SystemExit(f"validation failed with {len(total_errors)} error(s)")
    mode = "strict" if args.strict else "basic"
    print(f"{mode} validation passed ({args.mode} lane): {dataset_dir}")


if __name__ == "__main__":
    main()
