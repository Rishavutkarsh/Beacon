import argparse
import json
import re
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = {
    "row_id",
    "packet_id",
    "scenario_family",
    "split",
    "hazard",
    "language",
    "user_prompt",
    "target_response",
    "messages",
    "must_include",
    "must_avoid",
    "risk_level",
    "review_status",
    "training_ready",
}

SYSTEM_PROMPT = (
    "You are Beacon, an offline crisis companion for India-relevant disaster "
    "situations. Give conservative, practical guidance. State uncertainty "
    "clearly, do not invent live facts, and give safer next steps before escalation."
)

BAD_TOOL_PATTERNS = [
    r"<tool_call>",
    r"search_official_docs",
    r"read_official_doc",
    r"\bEvidence:\s*[a-z_]+:",
    r"\bdoc_id\b",
    r"\bsection_id\b",
]

LIVE_CERTAINTY_PATTERNS = [
    r"\b(the|this) (road|bridge|route|shelter|camp|building|room|house) is safe\b",
    r"\b(rescue|relief|ambulance|beds?|stock|food packets?) (is|are) (coming|available|open|confirmed)\b",
    r"\bofficials? (said|confirmed) (now|today|tonight)\b",
]

SCARY_OVERCLAIMS = [
    r"\bcarbon monoxide.*smell.*safe\b",
    r"\bgenerator.*inside.*safe\b",
    r"\bwhatsapp.*confirm",
    r"\bchildren can play in floodwater\b",
]


def load_jsonl(path):
    rows = []
    errors = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({"line": line_no, "error": str(exc)})
    return rows, errors


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def first_words(text, n=8):
    return " ".join(norm(text).split()[:n])


def row_errors(row):
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
    if row.get("schema_version") != "beacon-no-tool-natural-sft-v1":
        errors.append("wrong_schema_version")
    if row.get("review_status") != "needs_main_strict_review":
        errors.append("wrong_review_status")
    if row.get("training_ready") is not False:
        errors.append("training_ready_not_false")

    messages = row.get("messages", [])
    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        errors.append("wrong_message_roles")
    elif messages[0].get("content") != SYSTEM_PROMPT:
        errors.append("wrong_system_prompt")

    prompt = row.get("user_prompt", "")
    target = row.get("target_response", "")
    all_text = json.dumps(row, ensure_ascii=False)
    if len(prompt.split()) < 7:
        errors.append("user_prompt_too_short")
    if len(target.split()) < 30:
        errors.append("target_too_short")
    if len(target.split()) > 115:
        errors.append("target_too_long")
    if any(ord(ch) > 127 for ch in prompt + target):
        errors.append("non_ascii_prompt_or_target")
    for pattern in BAD_TOOL_PATTERNS:
        if re.search(pattern, all_text, re.I):
            errors.append("contains_tool_or_citation_artifact")
            break
    target_norm = norm(target)
    target_without_negated_safe = re.sub(
        r"\b(do not|don't|cannot|can't|never|not)\b[^.?!]{0,90}\b(safe|confirm|certify)\b",
        "",
        target_norm,
    )
    for pattern in LIVE_CERTAINTY_PATTERNS:
        if re.search(pattern, target_without_negated_safe, re.I):
            errors.append("unsupported_live_certainty")
            break
    for pattern in SCARY_OVERCLAIMS:
        if re.search(pattern, target_without_negated_safe, re.I):
            errors.append("unsafe_overclaim")
            break
    for phrase in row.get("must_include", []):
        if phrase and phrase.startswith("literal:") and norm(phrase.removeprefix("literal:")) not in target_norm:
            errors.append(f"missing_must_include:{phrase}")
            break
    for phrase in row.get("must_avoid", []):
        if phrase and norm(phrase) in target_norm:
            errors.append(f"contains_must_avoid:{phrase}")
            break
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--write-decisions", action="store_true")
    args = parser.parse_args()

    summary = {"packet_dir": str(args.packet_dir), "files": {}}
    for path in sorted(args.packet_dir.glob("packet_*_candidates.jsonl")):
        rows, parse_errors = load_jsonl(path)
        row_ids = Counter(r.get("row_id") for r in rows)
        prompts = Counter(norm(r.get("user_prompt", "")) for r in rows)
        stems = Counter(first_words(r.get("target_response", "")) for r in rows)
        error_counts = Counter()
        decisions = []
        for i, row in enumerate(rows, 1):
            errors = row_errors(row)
            if row_ids[row.get("row_id")] > 1:
                errors.append("duplicate_row_id")
            if prompts[norm(row.get("user_prompt", ""))] > 1:
                errors.append("duplicate_user_prompt")
            if stems[first_words(row.get("target_response", ""))] > 5:
                errors.append("answer_stem_overused")
            for e in errors:
                error_counts[e] += 1
            decisions.append({
                "row_number": i,
                "row_id": row.get("row_id"),
                "packet_id": row.get("packet_id"),
                "status": "needs_human_review" if not errors else "reject",
                "errors": errors,
            })

        summary["files"][path.name] = {
            "row_count": len(rows),
            "parse_errors": parse_errors,
            "error_counts": dict(error_counts),
            "reject_count": sum(d["status"] == "reject" for d in decisions),
            "needs_human_review_count": sum(d["status"] == "needs_human_review" for d in decisions),
            "split_counts": dict(Counter(r.get("split") for r in rows)),
        }
        if args.write_decisions:
            out = path.with_name(path.stem.replace("_candidates", "_auto_decisions") + ".jsonl")
            with out.open("w", encoding="utf-8") as handle:
                for d in decisions:
                    handle.write(json.dumps(d, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
