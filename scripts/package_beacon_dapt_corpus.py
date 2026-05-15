from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [
    Path("data/source_corpus/dapt_clean/dapt_clean.jsonl"),
    Path("data/source_corpus/retrieval_chunks/retrieval_chunks.jsonl"),
    Path("data/dapt_corpus/beacon_crisis_v1_download/dapt_all.jsonl"),
    Path("data/dapt_corpus/beacon_crisis_v1_download/dapt_train.jsonl"),
    Path("data/dapt_corpus/beacon_crisis_v1_download/dapt_dev.jsonl"),
    Path("data/dapt_corpus/beacon_crisis_v1/extracted"),
]
DEFAULT_OUT_DIR = Path("data/dapt_corpus/beacon_crisis_v1_train_ready")
MIN_TEXT_CHARS = 500
DEV_DOC_RATIO = 0.08
MIN_READY_TOKENS = 2_000_000
PREFERRED_TOKENS = 5_000_000
SCHEMA_VERSION = "beacon-dapt-train-ready-v1"

LIVE_OR_NEWS_TYPES = {"news", "news_event", "event_card", "live_alert", "live_status"}
BOILERPLATE_PATTERNS = [
    "accept all cookies",
    "enable javascript",
    "subscribe to newsletter",
    "skip to main content",
    "share this page",
]
LOCAL_RELEVANCE_TERMS = {
    "disaster", "emergency", "flood", "hurricane", "cyclone", "wildfire", "fire", "earthquake",
    "heat", "cold", "winter", "storm", "landslide", "mudslide", "power", "outage", "water",
    "food", "insulin", "diabetes", "carbon_monoxide", "co_", "generator", "preparedness",
    "response", "nims", "incident", "shelter", "evacuation", "hazard", "safety", "mitigation",
    "recovery", "risk", "medical", "search", "rescue",
}
LOCAL_REJECT_TERMS = {
    "accessibility", "privacy", "about_fda", "advisory_committees", "jobs_and_training",
    "visitor_information", "cosmetics", "tobacco", "newsroom_press_announcements",
    "recalls_market_withdrawals", "regulatory_information", "freedom_information",
    "campaignproxyservice", "courseoverview", "searchisbycurriculum", "nationalpreparednesssymposium",
    "website_policies", "no_fear_act",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_key(text: str) -> str:
    normalized = re.sub(r"\W+", " ", text.lower()).strip()
    return sha256_text(normalized)


def estimate_tokens(text: str) -> int:
    # A cheap, stable planning estimate. The trainer will do exact tokenization later.
    return max(1, round(len(text) / 4))


def stable_dev_bucket(document_id: str) -> bool:
    prefix = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8]
    bucket = int(prefix, 16) / 0xFFFFFFFF
    return bucket < DEV_DOC_RATIO


def list_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.jsonl")))
            expanded.extend(sorted(path.rglob("*.txt")))
        elif path.exists():
            expanded.append(path)
    seen = set()
    unique = []
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def row_text(row: dict[str, Any]) -> str:
    for key in ["text", "content", "chunk_text", "extracted_text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def row_id(row: dict[str, Any], fallback: str) -> str:
    for key in ["text_id", "chunk_id", "row_id", "id", "example_id"]:
        value = row.get(key)
        if value:
            return str(value)
    return fallback


def list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def canonical_record(row: dict[str, Any], source_path: Path, index: int) -> tuple[dict[str, Any] | None, str | None]:
    text = normalize_text(row_text(row))
    if len(text) < MIN_TEXT_CHARS:
        return None, "too_short_or_empty"
    lowered = text[:2000].lower()
    if any(pattern in lowered for pattern in BOILERPLATE_PATTERNS):
        return None, "boilerplate_or_navigation"
    document_type = str(row.get("document_type") or row.get("card_type") or "").lower()
    staleness = str(row.get("staleness_class") or "").lower()
    if document_type in LIVE_OR_NEWS_TYPES or staleness == "live":
        return None, "news_or_live_operational_text"

    document_id = str(row.get("document_id") or row.get("source_document_id") or row_id(row, f"{source_path.stem}_{index}"))
    source_id = str(row.get("source_id") or row.get("organization") or source_path.stem)
    text_hash = sha256_text(text)
    return {
        "schema_version": SCHEMA_VERSION,
        "text_id": row_id(row, f"{document_id}_{index:05d}"),
        "document_id": document_id,
        "source_id": source_id,
        "url": str(row.get("url") or row.get("source_url") or ""),
        "title": str(row.get("title") or row.get("document_title") or document_id),
        "organization": str(row.get("organization") or source_id),
        "jurisdiction": str(row.get("jurisdiction") or "unknown"),
        "language": str(row.get("language") or "unknown"),
        "hazards": list_value(row.get("hazards") or row.get("hazard_tags")),
        "source_tier": str(row.get("source_tier") or row.get("tier") or "unknown"),
        "license": str(row.get("license") or "recorded_for_educational_hackathon_use"),
        "retrieved_at": str(row.get("retrieved_at") or row.get("created_at_utc") or ""),
        "text": text,
        "estimated_tokens": estimate_tokens(text),
        "text_sha256": text_hash,
        "normalized_text_sha256": dedupe_key(text),
        "input_path": str(source_path),
    }, None


def canonical_text_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    path_key = path.stem.lower()
    if any(term in path_key for term in LOCAL_REJECT_TERMS):
        return None, "local_extracted_irrelevant_page"
    if not any(term in path_key for term in LOCAL_RELEVANCE_TERMS):
        return None, "local_extracted_missing_relevance_signal"
    text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if len(text) < MIN_TEXT_CHARS:
        return None, "too_short_or_empty"
    lowered = text[:2000].lower()
    if any(pattern in lowered for pattern in BOILERPLATE_PATTERNS):
        return None, "boilerplate_or_navigation"
    source_id = path.stem.split("_", 1)[0] if "_" in path.stem else "local_extracted"
    document_id = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower()
    return {
        "schema_version": SCHEMA_VERSION,
        "text_id": f"{document_id}_txt_0000",
        "document_id": document_id,
        "source_id": source_id,
        "url": f"local_extracted:{path.as_posix()}",
        "title": path.stem.replace("_", " "),
        "organization": source_id,
        "jurisdiction": "unknown",
        "language": "unknown",
        "hazards": [],
        "source_tier": "local_extracted_prior_crawl",
        "license": "recorded_for_educational_hackathon_use",
        "retrieved_at": "",
        "text": text,
        "estimated_tokens": estimate_tokens(text),
        "text_sha256": sha256_text(text),
        "normalized_text_sha256": dedupe_key(text),
        "input_path": str(path),
    }, None


def load_records(input_paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_normalized: set[str] = set()
    for path in list_inputs(input_paths):
        if path.suffix.lower() == ".txt":
            record, reason = canonical_text_file(path)
            if record is None:
                rejected.append({
                    "input_path": str(path),
                    "index": 0,
                    "reason": reason or "not_accepted",
                    "id": path.stem,
                    "document_id": path.stem,
                    "url": f"local_extracted:{path.as_posix()}",
                })
                continue
            key = record["normalized_text_sha256"]
            if key in seen_normalized:
                rejected.append({
                    "input_path": str(path),
                    "index": 0,
                    "reason": "duplicate_text",
                    "id": record["text_id"],
                    "document_id": record["document_id"],
                    "url": record["url"],
                })
                continue
            seen_normalized.add(key)
            records.append(record)
            continue
        for index, row in enumerate(read_jsonl(path)):
            record, reason = canonical_record(row, path, index)
            if record is None:
                rejected.append({
                    "input_path": str(path),
                    "index": index,
                    "reason": reason or "not_accepted",
                    "id": row_id(row, f"{path.stem}_{index}"),
                    "document_id": row.get("document_id"),
                    "url": row.get("url"),
                })
                continue
            key = record["normalized_text_sha256"]
            if key in seen_normalized:
                rejected.append({
                    "input_path": str(path),
                    "index": index,
                    "reason": "duplicate_text",
                    "id": record["text_id"],
                    "document_id": record["document_id"],
                    "url": record["url"],
                })
                continue
            seen_normalized.add(key)
            records.append(record)
    records.sort(key=lambda item: (item["document_id"], item["text_id"], item["text_sha256"]))
    return records, rejected


def split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for row in records:
        target = dev if stable_dev_bucket(row["document_id"]) else train
        split = "dev" if target is dev else "train"
        copied = dict(row)
        copied["split"] = split
        target.append(copied)
    if records and not dev:
        # Keep validation meaningful for very small smoke corpora.
        last_doc = records[-1]["document_id"]
        train = []
        dev = []
        for row in records:
            copied = dict(row)
            copied["split"] = "dev" if row["document_id"] == last_doc else "train"
            (dev if copied["split"] == "dev" else train).append(copied)
    return train, dev


def copy_sidecar_jsonl(out_dir: Path, input_paths: list[Path]) -> None:
    source_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for root in {path.parent for path in list_inputs(input_paths)} | {Path("data/source_corpus")}:
        if not root.exists():
            continue
        for name in ["source_cards.jsonl"]:
            source_rows.extend(read_jsonl(root / name))
        for name in ["document_cards.jsonl"]:
            document_rows.extend(read_jsonl(root / name))
        for candidate in [root / "rejected_document_cards.jsonl", root / "rejected" / "document_cards_rejected.jsonl"]:
            rejected_rows.extend(read_jsonl(candidate))
    if source_rows:
        write_jsonl(out_dir / "source_cards.jsonl", source_rows)
    if document_rows:
        write_jsonl(out_dir / "document_cards.jsonl", document_rows)
    if rejected_rows:
        write_jsonl(out_dir / "rejected_document_cards_source.jsonl", rejected_rows)


def build_manifest(out_dir: Path, rows: list[dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]], rejected: list[dict[str, Any]], input_paths: list[Path]) -> dict[str, Any]:
    total_tokens = sum(int(row["estimated_tokens"]) for row in rows)
    by_hazard = Counter(hazard for row in rows for hazard in row.get("hazards", []))
    by_language = Counter(row.get("language", "unknown") for row in rows)
    by_org = Counter(row.get("organization", "unknown") for row in rows)
    by_source = Counter(row.get("source_id", "unknown") for row in rows)
    doc_ids = {row["document_id"] for row in rows}
    ready = total_tokens >= MIN_READY_TOKENS and bool(train) and bool(dev)
    files = {
        "dapt_all": out_dir / "dapt_all.jsonl",
        "dapt_train": out_dir / "dapt_train.jsonl",
        "dapt_dev": out_dir / "dapt_dev.jsonl",
        "rejected": out_dir / "rejected_document_cards.jsonl",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "dapt_ready": ready,
        "readiness_reason": "ready" if ready else "below_minimum_tokens_or_empty_split",
        "minimum_ready_tokens": MIN_READY_TOKENS,
        "preferred_tokens": PREFERRED_TOKENS,
        "estimated_tokens": total_tokens,
        "row_count": len(rows),
        "document_count": len(doc_ids),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_documents": len({row["document_id"] for row in train}),
        "dev_documents": len({row["document_id"] for row in dev}),
        "rejected_rows": len(rejected),
        "input_paths": [str(path) for path in list_inputs(input_paths)],
        "coverage": {
            "hazards": dict(by_hazard.most_common()),
            "languages": dict(by_language.most_common()),
            "organizations_top_30": dict(by_org.most_common(30)),
            "sources": dict(by_source.most_common()),
        },
        "hashes": {key: sha256_file(path) for key, path in files.items() if path.exists()},
    }


def build(args: argparse.Namespace) -> None:
    input_paths = [Path(item) for item in args.inputs] if args.inputs else DEFAULT_INPUTS
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, rejected = load_records(input_paths)
    train, dev = split_records(records)
    write_jsonl(out_dir / "dapt_all.jsonl", records)
    write_jsonl(out_dir / "dapt_train.jsonl", train)
    write_jsonl(out_dir / "dapt_dev.jsonl", dev)
    write_jsonl(out_dir / "rejected_document_cards.jsonl", rejected)
    copy_sidecar_jsonl(out_dir, input_paths)
    manifest = build_manifest(out_dir, records, train, dev, rejected, input_paths)
    write_json(out_dir / "dapt_manifest.json", manifest)
    write_json(out_dir / "training_config.json", default_training_config(manifest))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def validate_required_row(row: dict[str, Any], split: str) -> list[str]:
    errors: list[str] = []
    for key in ["text_id", "document_id", "source_id", "url", "title", "text", "estimated_tokens", "text_sha256"]:
        if row.get(key) in (None, ""):
            errors.append(f"{split}:{row.get('text_id', '<missing>')}:missing:{key}")
    if row.get("split") != split:
        errors.append(f"{split}:{row.get('text_id')}:bad_split")
    if len(str(row.get("text", ""))) < MIN_TEXT_CHARS:
        errors.append(f"{split}:{row.get('text_id')}:short_text")
    if row.get("staleness_class") == "live":
        errors.append(f"{split}:{row.get('text_id')}:live_text")
    return errors


def validate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    train = read_jsonl(data_dir / "dapt_train.jsonl")
    dev = read_jsonl(data_dir / "dapt_dev.jsonl")
    manifest = json.loads((data_dir / "dapt_manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if not train:
        errors.append("empty_train")
    if not dev:
        errors.append("empty_dev")
    train_docs = {row.get("document_id") for row in train}
    dev_docs = {row.get("document_id") for row in dev}
    overlap = sorted(doc for doc in train_docs & dev_docs if doc)
    if overlap:
        errors.append(f"train_dev_document_overlap:{overlap[:20]}")
    seen_ids: set[str] = set()
    for split, rows in [("train", train), ("dev", dev)]:
        for row in rows:
            errors.extend(validate_required_row(row, split))
            text_id = str(row.get("text_id", ""))
            if text_id in seen_ids:
                errors.append(f"duplicate_text_id:{text_id}")
            seen_ids.add(text_id)
            if sha256_text(str(row.get("text", ""))) != row.get("text_sha256"):
                errors.append(f"{split}:{text_id}:text_hash_mismatch")
    total_tokens = sum(int(row.get("estimated_tokens", 0)) for row in train + dev)
    if total_tokens != manifest.get("estimated_tokens"):
        errors.append("manifest_token_count_mismatch")
    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors[:200],
        "error_count": len(errors),
        "estimated_tokens": total_tokens,
        "dapt_ready": bool(manifest.get("dapt_ready")) and not errors,
        "manifest_ready": bool(manifest.get("dapt_ready")),
    }
    write_json(data_dir / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def default_training_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "purpose": "Beacon first DAPT/CPT run configuration draft; tune before launch.",
        "dataset": {
            "train_file": "dapt_train.jsonl",
            "dev_file": "dapt_dev.jsonl",
            "estimated_tokens": manifest.get("estimated_tokens"),
            "train_rows": manifest.get("train_rows"),
            "dev_rows": manifest.get("dev_rows"),
        },
        "model": {
            "base_model": "google/gemma-4-e2b-it",
            "kaggle_model_path": "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
        },
        "qlora_draft": {
            "max_seq_length": 2048,
            "learning_rate": 2e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_scope": "attention_plus_mlp",
            "optim": "adamw_8bit",
            "max_grad_norm": 0.3,
            "packing": False,
        },
        "launch_policy": "do_not_train_until_user_approves_variables",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package existing Beacon crisis text into a DAPT-ready train/dev corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    build_parser.add_argument("--inputs", nargs="*", default=[])
    build_parser.add_argument("--clean", action="store_true", default=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "validate":
        validate(args)


if __name__ == "__main__":
    main()
