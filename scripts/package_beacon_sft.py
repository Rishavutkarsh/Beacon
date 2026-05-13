from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("data/expanded/beacon_v2_1015/run_018_pruned")
DEFAULT_OUT = Path("kaggle/input/beacon-sft-run018-pruned")
EXPECTED_COUNTS = {"train": 893, "dev": 95, "final_eval": 93}
SYSTEM_PROMPT = (
    "You are Beacon, an offline crisis companion. Give conservative, practical, "
    "source-grounded guidance. State uncertainty clearly, avoid live-status claims, "
    "and give concrete safer next steps before escalation."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gemma_text(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "assistant":
            role = "model"
        chunks.append(f"<|turn>{role}\n{content}<turn|>")
    return "\n".join(chunks)


def package_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt", "")).strip()
    target = str(row.get("target_response", "")).strip()
    if not prompt or not target:
        raise ValueError(f"{row.get('row_id', '<missing>')}: missing prompt or target_response")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    text = gemma_text(messages)
    return {
        "id": row["row_id"],
        "messages": messages,
        "text": text,
        "prompt": prompt,
        "target_response": target,
        "split": row["split"],
        "hazard_domain": row.get("hazard_domain", ""),
        "risk_level": row.get("risk_level", ""),
        "renderer_style": row.get("renderer_style", ""),
        "source_rule_ids": row.get("source_rule_ids", []),
        "target_behavior_tags": row.get("target_behavior_tags", []),
        "forbidden_behavior_tags": row.get("forbidden_behavior_tags", []),
        "seed_id": row.get("seed_id", ""),
        "seed_family_id": row.get("seed_family_id", ""),
        "scenario_cluster_id": row.get("scenario_cluster_id", ""),
        "content_hash": row.get("content_hash", sha256_text(prompt + "\n" + target)),
    }


def validate_packaged(records_by_split: dict[str, list[dict[str, Any]]], freeze: dict[str, Any]) -> None:
    counts = {split: len(rows) for split, rows in records_by_split.items()}
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"Packaged counts mismatch: expected {EXPECTED_COUNTS}, got {counts}")
    if freeze.get("status") != "pass":
        raise SystemExit("Freeze manifest is not pass.")
    if freeze.get("accepted_counts") != EXPECTED_COUNTS:
        raise SystemExit(f"Freeze accepted counts mismatch: {freeze.get('accepted_counts')}")
    ids: list[str] = []
    for split, rows in records_by_split.items():
        for record in rows:
            if record["split"] != split:
                raise SystemExit(f"{record['id']}: split field mismatch")
            if split in {"train", "dev"} and record["id"].startswith("ss_exp_final_eval_"):
                raise SystemExit(f"{record['id']}: final_eval row leaked into {split}")
            if "<|turn>user\n" not in record["text"] or "<|turn>model\n" not in record["text"]:
                raise SystemExit(f"{record['id']}: missing Gemma turn markers")
            if not record["target_response"].strip():
                raise SystemExit(f"{record['id']}: empty assistant response")
            ids.append(record["id"])
    duplicates = [row_id for row_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"Duplicate row IDs: {duplicates[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package frozen Beacon rows for Kaggle Unsloth SFT.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    rows_path = run_dir / "final_accepted_rows.jsonl"
    freeze_path = run_dir / "dataset_freeze_manifest.json"
    if not rows_path.exists() or not freeze_path.exists():
        raise SystemExit(f"Missing freeze artifacts under {run_dir}")

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to overwrite")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_rows = read_jsonl(rows_path)
    if any(row.get("quality_status") != "accepted" or row.get("review_state") != "frozen" for row in source_rows):
        raise SystemExit("All source rows must be accepted/frozen.")

    records_by_split = {"train": [], "dev": [], "final_eval": []}
    for row in source_rows:
        split = row.get("split")
        if split not in records_by_split:
            raise SystemExit(f"{row.get('row_id')}: unsupported split {split}")
        records_by_split[split].append(package_row(row))
    validate_packaged(records_by_split, freeze)

    for split, records in records_by_split.items():
        records.sort(key=lambda item: item["id"])
        write_jsonl(args.out / f"{split}.jsonl", records)
    shutil.copy2(freeze_path, args.out / "dataset_freeze_manifest.json")

    training_config = {
        "dataset": "beacon_run018_pruned",
        "counts": EXPECTED_COUNTS,
        "model_path": "/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1",
        "instruction_part": "<|turn>user\n",
        "response_part": "<|turn>model\n",
        "smoke": {
            "max_length": 512,
            "max_steps": 20,
            "train_rows": 16,
            "dev_rows": 8,
            "learning_rate": 1.5e-4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
        "full": {
            "max_length": 768,
            "learning_rate": 1e-4,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 2,
            "warmup_ratio": 0.03,
            "eval_steps": 25,
            "logging_steps": 5,
            "early_stopping_patience_evals": 4,
        },
    }
    write_json(args.out / "training_config.json", training_config)

    hash_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "source_final_accepted_rows_sha256": sha256_file(rows_path),
        "source_freeze_manifest_sha256": sha256_file(freeze_path),
        "files": {},
    }
    for path in sorted(args.out.iterdir()):
        if path.is_file():
            hash_manifest["files"][path.name] = sha256_file(path)
    write_json(args.out / "hash_manifest.json", hash_manifest)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": "beacon-sft-run018-pruned",
        "counts": EXPECTED_COUNTS,
        "system_prompt": SYSTEM_PROMPT,
        "freeze_status": freeze.get("status"),
        "freeze_artifact_hashes": freeze.get("artifact_hashes", {}),
        "hash_manifest_sha256": sha256_file(args.out / "hash_manifest.json"),
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(
        args.out / "dataset-metadata.json",
        {
            "id": "rishavutkarsh/beacon-sft-run018-pruned",
            "title": "Beacon SFT Run 018 Pruned",
            "licenses": [{"name": "CC0-1.0"}],
        },
    )
    print(json.dumps({"status": "pass", "out": str(args.out), "counts": EXPECTED_COUNTS}, indent=2))


if __name__ == "__main__":
    main()
