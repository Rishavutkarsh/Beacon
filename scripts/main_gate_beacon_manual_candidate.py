from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1_manual_candidate"
OUT_DIR = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1_manual_1000"
TARGET_COUNTS = {"train": 800, "dev": 100, "final_eval": 100}

BAD_PHRASES = re.compile(
    r"(document-backed|dataset|training|tool query|rewrite this|normalize this|returned section|returned snippet|"
    r"use the returned|as an ai|i am an ai|evidence:)",
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tool_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    sections = []
    for message in row.get("messages", []):
        if message.get("role") != "tool" or message.get("name") != "read_official_doc":
            continue
        try:
            payload = json.loads(str(message.get("content", "{}")))
        except json.JSONDecodeError:
            payload = {}
        sections.extend(payload.get("sections", []))
    return sections


def evidence_text(row: dict[str, Any]) -> str:
    parts = []
    for section in tool_sections(row):
        parts.append(str(section.get("snippet", "")))
        parts.extend(str(fact) for fact in section.get("key_facts", []))
    return " ".join(parts).lower()


def unsupported_claims(row: dict[str, Any]) -> list[str]:
    target = str(row.get("target_response", ""))
    lower = target.lower()
    evidence = evidence_text(row)
    out = []
    for match in EXACT_RE.finditer(target):
        claim = match.group(0).lower()
        window = lower[max(0, match.start() - 48): match.end() + 48]
        if claim in NEGATED_MYTHS and re.search(r"\b(no|not|nahi|mat|do not|don't|unsafe|assume)\b", window):
            continue
        if row.get("tool_required") and claim not in evidence:
            out.append(claim)
    return out


def main_gate(row: dict[str, Any], approved_stems: Counter[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    target = str(row.get("target_response", "")).strip()
    low = target.lower()
    if len(target.split()) < 12:
        reasons.append("too short to teach behavior")
    if BAD_PHRASES.search(target):
        reasons.append("visible scaffold/meta/tool language")
    claims = unsupported_claims(row)
    if claims:
        reasons.append("unsupported exact claims: " + ", ".join(claims))
    if str(row.get("row_family", "")).endswith("no_support"):
        if not re.search(r"\b(cannot|can't|do not|don't|nahi|mat|unsafe|verify|confirm|certify)\b", low):
            reasons.append("no-support answer lacks explicit abstention")
        if EXACT_RE.search(target):
            reasons.append("no-support answer includes exact number")
    if row.get("tool_required") and not row.get("section_ids"):
        reasons.append("tool row has no section evidence")
    stem = " ".join(target.split()[:16]).lower()
    if approved_stems[stem] >= 3:
        reasons.append("too many already-approved rows share the same opening")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    rows = read_jsonl(args.source / "all_approved_rows.jsonl")
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    approved_stems: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    family_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)

    rows_sorted = sorted(rows, key=lambda row: (str(row.get("split")) != "dev", str(row.get("split")) != "final_eval", row.get("row_id", "")))
    for row in rows_sorted:
        split = str(row.get("split", ""))
        if split_counts[split] >= TARGET_COUNTS.get(split, 0):
            candidate = dict(row)
            candidate["main_gate"] = {"status": "not_selected", "reasons": ["split target already filled"]}
            rejected.append(candidate)
            continue
        ok, reasons = main_gate(row, approved_stems)
        candidate = json.loads(json.dumps(row))
        if ok:
            candidate["main_gate"] = {"status": "approved_by_main", "reasons": []}
            approved.append(candidate)
            split_counts[split] += 1
            family_counts_by_split[split][str(row.get("row_family", ""))] += 1
            approved_stems[" ".join(str(row.get("target_response", "")).split()[:16]).lower()] += 1
        else:
            candidate["main_gate"] = {"status": "rejected_by_main", "reasons": reasons}
            rejected.append(candidate)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "all_rows.jsonl", approved)
    write_jsonl(args.out / "rejected_rows.jsonl", rejected)
    for split in ["train", "dev", "final_eval"]:
        write_jsonl(args.out / f"{split}.jsonl", [row for row in approved if row.get("split") == split])
    manifest = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source": str(args.source),
        "row_count": len(approved),
        "target_counts": TARGET_COUNTS,
        "by_split": dict(split_counts),
        "by_family": dict(Counter(str(row.get("row_family", "")) for row in approved).most_common()),
        "by_split_family": {split: dict(counter.most_common()) for split, counter in family_counts_by_split.items()},
        "rejected_count": len(rejected),
        "training_export_allowed": False,
        "status": "ready_for_human_spot_check" if split_counts == Counter(TARGET_COUNTS) else "incomplete",
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
