import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "row_id",
    "packet_id",
    "scenario_family",
    "split",
    "hazard",
    "user_prompt",
    "target_response",
    "tool_names",
    "tool_required",
    "doc_ids",
    "section_ids",
    "expected_facts",
    "risk_level",
    "review_status",
    "training_ready",
    "messages",
}

SCAFFOLD_PATTERNS = [
    r"\bthat keeps the decision usable\b",
    r"\banswer using only\b",
    r"\bdo not add details beyond\b",
    r"\bkeep the reply focused\b",
    r"\bthe practical answer is\b",
    r"\bstart with the card-backed point\b",
    r"\buse this boundary\b",
    r"\blocal grounding cards retrieved\b",
]

UNSUPPORTED_LIVE_PATTERNS = [
    r"\b(the|this) (road|bridge|shelter|route|house|room|building) is safe\b",
    r"\b(rescue|relief|ambulance|stock|beds?) (is|are) (available|coming|open)\b",
    r"\bofficials? (said|confirmed) (today|now|tonight)\b",
]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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
                errors.append({"line": line_no, "error": f"json_decode: {exc}"})
    return rows, errors


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def first_words(text, n=7):
    return " ".join(normalize(text).split()[:n])


def message_roles(row):
    return [m.get("role") for m in row.get("messages", [])]


def extract_tool_names(row):
    names = []
    for msg in row.get("messages", []):
        content = msg.get("content", "")
        for match in TOOL_CALL_RE.finditer(content):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                names.append("__bad_tool_json__")
                continue
            names.append(payload.get("name", ""))
    return names


def row_errors(row, known_doc_ids, known_section_ids):
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")

    if row.get("schema_version") != "beacon-official-doc-tool-v2-natural-packet":
        errors.append("wrong_schema_version")
    if row.get("training_ready") is not False:
        errors.append("training_ready_not_false")
    if row.get("review_status") != "needs_main_strict_review":
        errors.append("wrong_review_status")
    if row.get("tool_required") is not True:
        errors.append("tool_required_not_true")

    roles = message_roles(row)
    if roles.count("tool") < 2:
        errors.append("needs_two_tool_messages")
    if roles[:2] != ["system", "user"]:
        errors.append("messages_must_start_system_user")
    if not roles or roles[-1] != "assistant":
        errors.append("final_message_not_assistant")

    tool_names = extract_tool_names(row)
    if tool_names[:2] != ["search_official_docs", "read_official_doc"]:
        errors.append("missing_search_then_read_tool_calls")
    if "__bad_tool_json__" in tool_names:
        errors.append("bad_tool_call_json")

    user_prompt = row.get("user_prompt", "")
    final = row.get("target_response", "")
    if len(user_prompt.split()) < 10:
        errors.append("user_prompt_too_short")
    if len(final.split()) < 25:
        errors.append("target_response_too_short")
    if len(final.split()) > 125:
        errors.append("target_response_too_long")
    if "Evidence:" not in final and "evidence:" not in final:
        errors.append("missing_evidence_citation")

    final_norm = normalize(final)
    for pattern in SCAFFOLD_PATTERNS:
        if re.search(pattern, final_norm, re.I):
            errors.append("visible_scaffold_language")
            break
    for pattern in UNSUPPORTED_LIVE_PATTERNS:
        if re.search(pattern, final_norm, re.I):
            errors.append("unsupported_live_or_safety_certainty")
            break

    if len(set(row.get("doc_ids", []))) != len(row.get("doc_ids", [])):
        errors.append("duplicate_doc_ids")
    unknown_docs = sorted(set(row.get("doc_ids", [])) - known_doc_ids)
    if unknown_docs:
        errors.append(f"unknown_doc_ids:{','.join(unknown_docs[:4])}")
    unknown_sections = sorted(set(row.get("section_ids", [])) - known_section_ids)
    if unknown_sections:
        errors.append(f"unknown_section_ids:{','.join(unknown_sections[:4])}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--doc-index", type=Path, default=Path("data/local_grounding/official_doc_tool_v1/official_doc_index.jsonl"))
    parser.add_argument("--chunk-index", type=Path, default=Path("data/local_grounding/official_doc_tool_v1/official_doc_chunk_index.jsonl"))
    parser.add_argument("--write-decisions", action="store_true")
    args = parser.parse_args()

    doc_rows, doc_parse_errors = load_jsonl(args.doc_index)
    chunk_rows, chunk_parse_errors = load_jsonl(args.chunk_index)
    known_doc_ids = {r.get("doc_id") for r in doc_rows if r.get("doc_id")}
    known_section_ids = {r.get("section_id") for r in chunk_rows if r.get("section_id")}

    summary = {
        "packet_dir": str(args.packet_dir),
        "doc_index_parse_errors": doc_parse_errors,
        "chunk_index_parse_errors": chunk_parse_errors,
        "files": {},
    }

    for path in sorted(args.packet_dir.glob("packet_*_candidates.jsonl")):
        rows, parse_errors = load_jsonl(path)
        row_id_counts = Counter(r.get("row_id") for r in rows)
        prompt_counts = Counter(normalize(r.get("user_prompt", "")) for r in rows)
        answer_stem_counts = Counter(first_words(r.get("target_response", "")) for r in rows)
        family_counts = Counter(r.get("scenario_family") for r in rows)
        split_counts = Counter(r.get("split") for r in rows)
        error_counts = Counter()
        decisions = []

        for idx, row in enumerate(rows, 1):
            errors = row_errors(row, known_doc_ids, known_section_ids)
            if row_id_counts[row.get("row_id")] > 1:
                errors.append("duplicate_row_id")
            if prompt_counts[normalize(row.get("user_prompt", ""))] > 1:
                errors.append("duplicate_user_prompt")
            if answer_stem_counts[first_words(row.get("target_response", ""))] > 6:
                errors.append("answer_stem_overused")
            for error in errors:
                error_counts[error] += 1
            decisions.append({
                "row_number": idx,
                "row_id": row.get("row_id"),
                "packet_id": row.get("packet_id"),
                "status": "needs_human_review" if not errors else "reject",
                "errors": errors,
            })

        summary["files"][path.name] = {
            "row_count": len(rows),
            "parse_errors": parse_errors,
            "family_counts": dict(family_counts),
            "split_counts": dict(split_counts),
            "error_counts": dict(error_counts),
            "needs_human_review_count": sum(1 for d in decisions if d["status"] == "needs_human_review"),
            "reject_count": sum(1 for d in decisions if d["status"] == "reject"),
        }

        if args.write_decisions:
            out_path = path.with_name(path.stem.replace("_candidates", "_auto_decisions") + ".jsonl")
            with out_path.open("w", encoding="utf-8") as handle:
                for decision in decisions:
                    handle.write(json.dumps(decision, ensure_ascii=True, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
