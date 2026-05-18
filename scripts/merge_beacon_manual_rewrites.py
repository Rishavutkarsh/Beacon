from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1_manual_rewrite"
SOURCE_PACKAGE = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1"
OUT_DIR = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1_manual_candidate"

SCAFFOLD_RE = re.compile(
    r"(tool query|rewrite this|normalize this|returned section|returned snippet|use the returned|"
    r"document-backed|as an ai|i am an ai|training|dataset|evidence:)",
    re.I,
)
EXACT_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:hours?|minutes?|days?|feet|degrees?|grams?|g|percent)|"
    r"40\s*degrees|20\s*feet|15\s*(?:grams?|g)|1\s*minute|3\s*minutes|30\s*minutes)\b",
    re.I,
)
NEGATED_MYTHS = {"60 degrees", "72 hours", "3 days", "3 din"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def tool_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    sections = []
    for message in row.get("messages", []):
        if message.get("role") == "tool" and message.get("name") == "read_official_doc":
            try:
                payload = json.loads(str(message.get("content", "{}")))
            except json.JSONDecodeError:
                payload = {}
            sections.extend(payload.get("sections", []))
    return sections


def evidence_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for section in tool_sections(row):
        parts.append(str(section.get("snippet", "")))
        parts.extend(str(fact) for fact in section.get("key_facts", []))
    return " ".join(parts).lower()


def unsupported_exact_claims(text: str, evidence: str) -> list[str]:
    out: list[str] = []
    lower = text.lower()
    for match in EXACT_RE.finditer(text):
        claim = match.group(0).lower()
        window = lower[max(0, match.start() - 48): match.end() + 48]
        if claim in NEGATED_MYTHS and re.search(r"\b(no|not|nahi|mat|do not|don't|unsafe|assume)\b", window):
            continue
        if claim not in evidence:
            out.append(claim)
    return out


def validate_candidate(row: dict[str, Any], output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    response = str(output.get("final_response", "")).strip()
    if not response:
        errors.append("missing final_response")
    if output.get("decision") not in {"approved", "reject"}:
        errors.append("decision must be approved or reject")
    if SCAFFOLD_RE.search(response):
        errors.append("visible scaffold/tool-process language")
    if row.get("tool_required") and output.get("decision") == "approved":
        claims = unsupported_exact_claims(response, evidence_text(row))
        if claims:
            errors.append("unsupported exact claims: " + ", ".join(claims))
    if str(row.get("row_family", "")).endswith("no_support") and output.get("decision") == "approved":
        low = response.lower()
        if not re.search(r"\b(cannot|can't|not enough|do not|don't|nahi|mat|unsafe|verify|confirm|certify)\b", low):
            errors.append("no-support response lacks clear abstention")
        if EXACT_RE.search(response):
            errors.append("no-support response contains exact numeric claim")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--source", type=Path, default=SOURCE_PACKAGE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    source_rows = {row["row_id"]: row for row in read_jsonl(args.source / "all_rows.jsonl")}
    approved_seed = read_jsonl(args.work_dir / "approved_seed_rows.jsonl")
    worker_outputs: dict[str, dict[str, Any]] = {}
    missing_shards: list[str] = []
    for idx in range(1, 7):
        path = args.work_dir / "shards" / f"shard_{idx:02d}_output.jsonl"
        if not path.exists():
            missing_shards.append(str(path))
            continue
        for row in read_jsonl(path):
            worker_outputs[str(row.get("row_id", ""))] = row

    gate_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in approved_seed:
        candidate = json.loads(json.dumps(row))
        candidate["manual_gate"] = {"status": "approved_seed_unchanged"}
        accepted_rows.append(candidate)

    for row_id, output in sorted(worker_outputs.items()):
        source = source_rows.get(row_id)
        if not source:
            gate_rows.append({"row_id": row_id, "gate_status": "reject", "errors": ["unknown row_id"]})
            continue
        errors = validate_candidate(source, output)
        gate_status = "approved" if output.get("decision") == "approved" and not errors else "reject"
        candidate = json.loads(json.dumps(source))
        candidate["target_response"] = str(output.get("final_response", "")).strip()
        if candidate.get("messages") and candidate["messages"][-1].get("role") == "assistant":
            candidate["messages"][-1]["content"] = candidate["target_response"]
        candidate["manual_gate"] = {
            "status": gate_status,
            "worker_decision": output.get("decision"),
            "worker_notes": output.get("notes", ""),
            "errors": errors,
        }
        gate_rows.append({"row_id": row_id, "gate_status": gate_status, "errors": errors, "worker_notes": output.get("notes", "")})
        if gate_status == "approved":
            accepted_rows.append(candidate)
        else:
            rejected_rows.append(candidate)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "all_approved_rows.jsonl", accepted_rows)
    write_jsonl(args.out / "rejected_or_failed_rows.jsonl", rejected_rows)
    write_jsonl(args.out / "gate_report_rows.jsonl", gate_rows)
    for split in ["train", "dev", "final_eval"]:
        write_jsonl(args.out / f"{split}.jsonl", [row for row in accepted_rows if row.get("split") == split])
    manifest = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_package": str(args.source),
        "work_dir": str(args.work_dir),
        "missing_shards": missing_shards,
        "approved_seed_count": len(approved_seed),
        "worker_output_count": len(worker_outputs),
        "approved_count": len(accepted_rows),
        "rejected_or_failed_count": len(rejected_rows),
        "by_split": dict(Counter(str(row.get("split", "")) for row in accepted_rows).most_common()),
        "by_family": dict(Counter(str(row.get("row_family", "")) for row in accepted_rows).most_common()),
        "training_export_allowed": False,
        "status": "incomplete" if missing_shards else "candidate_needs_main_gate_review",
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
