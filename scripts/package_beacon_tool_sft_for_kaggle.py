from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "assistant_sft" / "beacon_tool_plus_no_tool_sft_v1_final_reviewed"
DEFAULT_OUT = ROOT / "data" / "assistant_sft" / "beacon_tool_plus_no_tool_sft_v1_kaggle_text_export"
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"
MODEL_MARKER = "<|turn>model\n"
USER_MARKER = "<|turn>user\n"
SYSTEM_MARKER = "<|turn>system\n"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def turn(role: str, content: str) -> str:
    return f"{TURN_OPEN}{role}\n{content.strip()}{TURN_CLOSE}"


def tool_result_content(message: dict[str, Any]) -> str:
    name = str(message.get("name", "")).strip()
    if not name:
        raise ValueError("tool result missing name")
    content = str(message.get("content", "")).strip()
    if not content:
        raise ValueError(f"tool result {name} missing content")
    return f'<tool_result name="{name}">{content}</tool_result>'


def render_text(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    rendered: list[str] = []
    converted: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role == "assistant":
            out_role = "model"
            content = str(message.get("content", ""))
        elif role == "tool":
            out_role = "user"
            content = tool_result_content(message)
        elif role in {"system", "user"}:
            out_role = role
            content = str(message.get("content", ""))
        else:
            raise ValueError(f"unsupported message role: {role!r}")
        rendered.append(turn(out_role, content))
        converted.append({"role": out_role, "content": content.strip()})
    return "\n".join(rendered), converted


def model_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    while True:
        marker_start = text.find(MODEL_MARKER, start)
        if marker_start < 0:
            break
        content_start = marker_start + len(MODEL_MARKER)
        content_end = text.find(TURN_CLOSE, content_start)
        if content_end < 0:
            raise ValueError("model turn missing closing marker")
        spans.append((content_start, content_end, text[content_start:content_end]))
        start = content_end + len(TURN_CLOSE)
    return spans


def non_model_text(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end, _content in model_spans(text):
        marker_start = text.rfind(MODEL_MARKER, 0, start)
        pieces.append(text[cursor:marker_start])
        cursor = end + len(TURN_CLOSE)
    pieces.append(text[cursor:])
    return "".join(pieces)


def validate_record(record: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    text = str(record.get("text", ""))
    messages = source_row.get("messages", [])
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    spans = model_spans(text)
    if len(spans) != len(assistant_messages):
        errors.append(f"model turn count {len(spans)} != assistant message count {len(assistant_messages)}")
    for (_start, _end, span), message in zip(spans, assistant_messages, strict=False):
        if span.strip() != str(message.get("content", "")).strip():
            errors.append("model span does not exactly match source assistant content")
            break
    if record["tool_required"]:
        if len(tool_messages) != 2:
            errors.append(f"tool row expected 2 tool result messages, got {len(tool_messages)}")
        if text.count("<tool_call>") != 2 or text.count("</tool_call>") != 2:
            errors.append("tool row expected exactly 2 assistant tool calls")
        if text.count("<tool_result name=") != 2 or text.count("</tool_result>") != 2:
            errors.append("tool row expected exactly 2 synthetic user tool results")
        if "<tool_result" in " ".join(span for _start, _end, span in spans):
            errors.append("tool result leaked into supervised model span")
    else:
        if tool_messages:
            errors.append("no-tool row contains source tool message")
        if "<tool_call>" in text or "<tool_result" in text:
            errors.append("no-tool row contains tool markers")
    outside_model = non_model_text(text)
    if "<tool_call>" in outside_model:
        errors.append("tool call appears outside model span")
    if not text.startswith(SYSTEM_MARKER):
        errors.append("text must start with system turn")
    if USER_MARKER not in text or MODEL_MARKER not in text:
        errors.append("text missing user or model marker")
    if text.count(TURN_OPEN) != text.count(TURN_CLOSE):
        errors.append("turn marker counts differ")
    if not text.endswith(TURN_CLOSE):
        errors.append("text must end with closing turn marker")
    return errors


def package_row(row: dict[str, Any]) -> dict[str, Any]:
    text, converted_messages = render_text(row.get("messages", []))
    return {
        "id": row["row_id"],
        "row_id": row["row_id"],
        "split": row["split"],
        "row_family": row.get("row_family", ""),
        "case_family_id": row.get("case_family_id", ""),
        "hazard": row.get("hazard", ""),
        "tool_required": bool(row.get("tool_required")),
        "query_rewrite_required": bool(row.get("query_rewrite_required")),
        "tool_names": row.get("tool_names", []),
        "doc_ids": row.get("doc_ids", []),
        "section_ids": row.get("section_ids", []),
        "messages": converted_messages,
        "text": text,
        "target_response": row.get("target_response", ""),
        "source_content_hash": sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True)),
        "text_hash": sha256_text(text),
    }


def validate_package(records_by_split: dict[str, list[dict[str, Any]]], source_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    all_records = [record for rows in records_by_split.values() for record in rows]
    ids = [record["id"] for record in all_records]
    duplicates = [row_id for row_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append({"package": "all", "errors": [f"duplicate ids: {duplicates[:20]}"]})
    for split, records in records_by_split.items():
        for record in records:
            source = source_by_id.get(record["id"])
            if source is None:
                errors.append({"id": record["id"], "errors": ["missing source row"]})
                continue
            if record["split"] != split:
                errors.append({"id": record["id"], "errors": ["split mismatch"]})
            row_errors = validate_record(record, source)
            if row_errors:
                errors.append({"id": record["id"], "errors": row_errors})
            char_len = len(record["text"])
            if char_len > 20000:
                warnings.append({"id": record["id"], "warning": f"long text chars={char_len}"})
    all_text = "\n".join(record["text"] for record in all_records)
    manifest = {
        "created_at_utc": utc_now(),
        "status": "valid" if not errors else "invalid",
        "row_count": len(all_records),
        "by_split": {split: len(rows) for split, rows in records_by_split.items()},
        "by_row_family": dict(Counter(record["row_family"] for record in all_records).most_common()),
        "tool_required_count": sum(record["tool_required"] for record in all_records),
        "no_tool_count": sum(not record["tool_required"] for record in all_records),
        "query_rewrite_count": sum(record["query_rewrite_required"] for record in all_records),
        "model_turn_count": all_text.count(MODEL_MARKER),
        "synthetic_tool_result_turn_count": all_text.count("<tool_result name="),
        "assistant_tool_call_count": all_text.count("<tool_call>"),
        "mask_policy": {
            "dataset_text_field": "text",
            "instruction_part": USER_MARKER,
            "response_part": MODEL_MARKER,
            "tool_results_are_synthetic_user_turns": True,
            "supervised_spans": "model turns only: assistant tool calls and final assistant answers",
        },
        "length_chars": {
            "min": min(len(record["text"]) for record in all_records),
            "max": max(len(record["text"]) for record in all_records),
            "avg": round(sum(len(record["text"]) for record in all_records) / max(len(all_records), 1), 2),
        },
        "validation": {"errors": errors[:200], "error_count": len(errors), "warnings": warnings[:200], "warning_count": len(warnings)},
        "training_ready": True,
        "training_export_allowed": True,
        "training_launched": False,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package approved Beacon tool/no-tool SFT rows as plain text for Kaggle.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to overwrite")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    source_manifest = read_json(args.source / "manifest.json")
    if source_manifest.get("training_export_allowed") is not True:
        raise SystemExit("Source package is not marked training_export_allowed=true.")
    source_rows = read_jsonl(args.source / "all_rows.jsonl")
    source_by_id = {row["row_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise SystemExit("Source rows have duplicate row_id values.")

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "final_eval": []}
    for row in source_rows:
        split = row.get("split")
        if split not in records_by_split:
            raise SystemExit(f"{row.get('row_id')}: unsupported split {split!r}")
        records_by_split[split].append(package_row(row))
    for records in records_by_split.values():
        records.sort(key=lambda item: item["id"])

    validation_manifest = validate_package(records_by_split, source_by_id)
    if validation_manifest["validation"]["error_count"]:
        write_json(args.out / "validation_report.json", validation_manifest)
        raise SystemExit(f"Validation failed: {validation_manifest['validation']['error_count']} row(s)")

    for split, records in records_by_split.items():
        write_jsonl(args.out / f"{split}.jsonl", records)
    write_jsonl(args.out / "all_rows.jsonl", [record for split in ["train", "dev", "final_eval"] for record in records_by_split[split]])

    training_config = {
        "dataset": "beacon-tool-plus-no-tool-sft-v1-kaggle-text",
        "base_adapter": "beacon-assistant-sft-v1-ckpt300-best-adapter",
        "dataset_text_field": "text",
        "train_on_responses_only": True,
        "instruction_part": USER_MARKER,
        "response_part": MODEL_MARKER,
        "max_length_recommendation": 4096,
        "tool_result_policy": "tool results are rendered as synthetic user turns and must not be supervised",
        "counts": validation_manifest["by_split"],
    }
    write_json(args.out / "training_config.json", training_config)
    write_json(args.out / "validation_report.json", validation_manifest)

    hash_manifest = {
        "created_at_utc": utc_now(),
        "source_dir": str(args.source),
        "source_manifest_sha256": sha256_file(args.source / "manifest.json"),
        "source_all_rows_sha256": sha256_file(args.source / "all_rows.jsonl"),
        "files": {},
    }
    for path in sorted(args.out.iterdir()):
        if path.is_file():
            hash_manifest["files"][path.name] = sha256_file(path)
    write_json(args.out / "hash_manifest.json", hash_manifest)

    manifest = {
        **validation_manifest,
        "name": "beacon-tool-plus-no-tool-sft-v1-kaggle-text",
        "source_dir": str(args.source),
        "source_status": source_manifest.get("status"),
        "hash_manifest_sha256": sha256_file(args.out / "hash_manifest.json"),
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(
        args.out / "dataset-metadata.json",
        {
            "id": "rishavutkarsh/beacon-tool-plus-no-tool-sft-v1-kaggle-text",
            "title": "Beacon Tool-Aware SFT v1 Kaggle Text Export",
            "licenses": [{"name": "other"}],
        },
    )
    print(json.dumps({"status": "pass", "out": str(args.out), "counts": validation_manifest["by_split"]}, indent=2))


if __name__ == "__main__":
    main()
