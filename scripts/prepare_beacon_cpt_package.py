from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_DIR = Path("data/dapt_corpus/beacon_crisis_v1_train_ready")
OUT_DIR = Path("data/dapt_corpus/beacon_crisis_v1_cpt_kaggle")
SCHEMA_VERSION = "beacon-cpt-kaggle-package-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def token_estimate(text: str) -> int:
    return max(1, round(len(text) / 4))


def normalize_row(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    text = str(row["text"]).strip()
    content_hash = sha256_text(text)
    source_row_id = str(row.get("text_id") or row.get("chunk_id") or f"{split}_{index:06d}")
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": f"beacon_cpt_{split}_{index:06d}",
        "chunk_id": source_row_id,
        "source_id": str(row.get("source_id") or "unknown"),
        "document_id": str(row.get("document_id") or source_row_id),
        "split": split,
        "text": text,
        "content_hash": content_hash,
        "token_estimate": int(row.get("estimated_tokens") or token_estimate(text)),
        "url": str(row.get("url") or ""),
        "title": str(row.get("title") or ""),
        "organization": str(row.get("organization") or ""),
        "language": str(row.get("language") or "unknown"),
        "hazards": row.get("hazards") or [],
    }


def dev_or_test(document_id: str) -> str:
    value = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16)
    return "test" if value % 2 else "dev"


def build(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_source = read_jsonl(source_dir / "dapt_train.jsonl")
    heldout_source = read_jsonl(source_dir / "dapt_dev.jsonl")
    train = [normalize_row(row, "train", index) for index, row in enumerate(train_source)]
    dev_rows = []
    test_rows = []
    for row in heldout_source:
        target = dev_rows if dev_or_test(str(row.get("document_id"))) == "dev" else test_rows
        target.append(row)
    if not dev_rows or not test_rows:
        midpoint = max(1, len(heldout_source) // 2)
        dev_rows, test_rows = heldout_source[:midpoint], heldout_source[midpoint:]
    dev = [normalize_row(row, "dev", index) for index, row in enumerate(dev_rows)]
    test = [normalize_row(row, "test", index) for index, row in enumerate(test_rows)]
    write_jsonl(out_dir / "cpt_train.jsonl", train)
    write_jsonl(out_dir / "cpt_dev.jsonl", dev)
    write_jsonl(out_dir / "cpt_test.jsonl", test)
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "split_method": "train inherited from dapt_train; dapt_dev split by document hash into cpt_dev/cpt_test",
        "train_documents": len({row["document_id"] for row in train}),
        "dev_documents": len({row["document_id"] for row in dev}),
        "test_documents": len({row["document_id"] for row in test}),
        "train_dev_overlap": sorted({row["document_id"] for row in train} & {row["document_id"] for row in dev}),
        "train_test_overlap": sorted({row["document_id"] for row in train} & {row["document_id"] for row in test}),
        "dev_test_overlap": sorted({row["document_id"] for row in dev} & {row["document_id"] for row in test}),
    }
    write_json(out_dir / "cpt_split_manifest.json", split_manifest)
    training_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "draft_user_tunes_before_launch",
        "max_seq_length": 1024,
        "packing": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "num_train_epochs": 2,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_scope": "language_attention_plus_mlp",
        "optim": "paged_adamw_8bit",
        "max_grad_norm": 0.3,
        "weight_decay": 0.01,
        "eval_steps": 25,
        "save_steps": 50,
        "save_total_limit": 3,
    }
    write_json(out_dir / "cpt_training_config.json", training_config)
    files = ["cpt_train.jsonl", "cpt_dev.jsonl", "cpt_test.jsonl", "cpt_split_manifest.json", "cpt_training_config.json"]
    package_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_dir": str(source_dir),
        "dapt_manifest_sha256": sha256_file(source_dir / "dapt_manifest.json"),
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "token_estimates": {
            "train": sum(row["token_estimate"] for row in train),
            "dev": sum(row["token_estimate"] for row in dev),
            "test": sum(row["token_estimate"] for row in test),
            "total": sum(row["token_estimate"] for row in train + dev + test),
        },
        "hashes": {name: sha256_file(out_dir / name) for name in files},
        "launch_policy": "training_not_started_user_tunes_params_first",
    }
    write_json(out_dir / "cpt_package_manifest.json", package_manifest)
    write_json(out_dir / "dataset-metadata.json", {"title": "Beacon Crisis CPT/DAPT v1", "id": "rishavutkarsh/beacon-crisis-v1-cpt"})
    print(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True))


def validate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    train = read_jsonl(out_dir / "cpt_train.jsonl")
    dev = read_jsonl(out_dir / "cpt_dev.jsonl")
    test = read_jsonl(out_dir / "cpt_test.jsonl")
    manifest = json.loads((out_dir / "cpt_package_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for split, rows in [("train", train), ("dev", dev), ("test", test)]:
        if not rows:
            errors.append(f"empty_{split}")
        for row in rows:
            for key in ["row_id", "chunk_id", "source_id", "document_id", "split", "text", "content_hash", "token_estimate"]:
                if row.get(key) in (None, ""):
                    errors.append(f"{split}:{row.get('row_id', '<missing>')}:missing:{key}")
            if row.get("split") != split:
                errors.append(f"{split}:{row.get('row_id')}:bad_split")
            if sha256_text(str(row.get("text", "")).strip()) != row.get("content_hash"):
                errors.append(f"{split}:{row.get('row_id')}:hash_mismatch")
            forbidden = ["<|turn>user", "<|turn>model", "assistant_response", "expected_contract"]
            if any(item in row["text"] for item in forbidden):
                errors.append(f"{split}:{row.get('row_id')}:forbidden_sft_or_eval_marker")
    split_docs = {
        "train": {row["document_id"] for row in train},
        "dev": {row["document_id"] for row in dev},
        "test": {row["document_id"] for row in test},
    }
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        overlap = split_docs[left] & split_docs[right]
        if overlap:
            errors.append(f"{left}_{right}_document_overlap:{sorted(overlap)[:10]}")
    for name, expected in manifest.get("hashes", {}).items():
        observed = sha256_file(out_dir / name)
        if observed != expected:
            errors.append(f"hash_mismatch:{name}")
    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors[:200],
        "error_count": len(errors),
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "token_estimates": manifest.get("token_estimates", {}),
    }
    write_json(out_dir / "cpt_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Beacon DAPT corpus as a Kaggle CPT package.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    build_parser.add_argument("--out-dir", default=str(OUT_DIR))
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "validate":
        validate(args)


if __name__ == "__main__":
    main()
