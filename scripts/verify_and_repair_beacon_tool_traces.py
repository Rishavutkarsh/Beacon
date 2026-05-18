from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_doc_tool import (  # noqa: E402
    exact_claims,
    load_doc_index,
    load_section_index,
    read_official_doc,
    search_official_docs,
    write_json,
    write_jsonl,
)


CALL_RE = re.compile(r"^<tool_call>(.*)</tool_call>$", re.S)
TOOL_RESULT_FILES = ["all_rows.jsonl", "train.jsonl", "dev.jsonl", "final_eval.jsonl"]


def parse_call(content: str) -> dict[str, Any]:
    match = CALL_RE.match(content)
    if not match:
        raise ValueError("assistant message is not a tool call")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict) or not isinstance(payload.get("arguments"), dict):
        raise ValueError("tool call payload must contain arguments object")
    return payload


def serialize_call(name: str, args: dict[str, Any]) -> str:
    return "<tool_call>" + json.dumps({"arguments": args, "name": name}, ensure_ascii=False, sort_keys=True) + "</tool_call>"


def search_payload(args: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
    hits = search_official_docs(
        query=str(args.get("query", "")),
        doc_index=docs,
        hazard=args.get("hazard"),
        organization=args.get("organization"),
        top_k=int(args.get("top_k", 5)),
    )
    return {
        "documents": [
            {
                "doc_id": hit.doc_id,
                "hazards": hit.hazards,
                "organization": hit.organization,
                "score": hit.score,
                "title": hit.title,
            }
            for hit in hits
        ]
    }


def read_payload(args: dict[str, Any], sections: list[dict[str, Any]]) -> dict[str, Any]:
    query = str(args.get("section_or_page_query", args.get("section_or_query", args.get("query", ""))))
    hits = read_official_doc(
        doc_id=str(args.get("doc_id", "")),
        section_query=query,
        section_index=sections,
        top_k=int(args.get("top_k", 3)),
    )
    return {
        "sections": [
            {
                "doc_id": hit.doc_id,
                "key_facts": hit.key_facts,
                "score": hit.score,
                "section_id": hit.section_id,
                "snippet": hit.snippet,
                "title": hit.title,
            }
            for hit in hits
        ]
    }


def all_search_hits(args: dict[str, Any], docs: list[dict[str, Any]]) -> list[str]:
    widened = dict(args)
    widened["top_k"] = 1000
    return [row["doc_id"] for row in search_payload(widened, docs)["documents"]]


def tool_messages(messages: list[dict[str, Any]]) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    calls: list[tuple[int, dict[str, Any]]] = []
    results: list[tuple[int, dict[str, Any]]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and str(message.get("content", "")).startswith("<tool_call>"):
            calls.append((index, parse_call(str(message.get("content", "")))))
        if message.get("role") == "tool":
            results.append((index, message))
    return calls, results


def evidence_text(payload: dict[str, Any]) -> str:
    return " ".join(
        str(section.get("snippet", "")) + " " + " ".join(str(fact) for fact in section.get("key_facts", []))
        for section in payload.get("sections", [])
    ).lower()


def claim_is_negative_myth(claim: str, answer: str) -> bool:
    lowered = answer.lower()
    claim_lower = claim.lower()
    index = lowered.find(claim_lower)
    if index < 0:
        return False
    window = lowered[max(0, index - 90) : index + len(claim_lower) + 80]
    return bool(re.search(r"\b(no|not|nahi|mat|do not|don't|unsafe|wrong|myth|assume mat|safe assume mat)\b", window))


def repair_row(row: dict[str, Any], docs: list[dict[str, Any]], sections: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    repairs: list[str] = []
    messages = [dict(message) for message in row.get("messages", [])]
    calls, results = tool_messages(messages)
    if row.get("tool_required"):
        if len(calls) != 2 or len(results) != 2:
            return row, [f"expected 2 calls and 2 results, got {len(calls)} calls and {len(results)} results"], repairs
        search_call_index, search_call = calls[0]
        read_call_index, read_call = calls[1]
        _, search_result = results[0]
        _, read_result = results[1]
        if search_call.get("name") != "search_official_docs" or read_call.get("name") != "read_official_doc":
            return row, ["unexpected tool call order"], repairs
        if search_result.get("name") != "search_official_docs" or read_result.get("name") != "read_official_doc":
            return row, ["unexpected tool result order"], repairs

        search_args = dict(search_call["arguments"])
        read_args = dict(read_call["arguments"])
        read_doc_id = str(read_args.get("doc_id", ""))
        current_doc_ids = all_search_hits(search_args, docs)
        if read_doc_id not in current_doc_ids:
            errors.append(f"read doc {read_doc_id} is not retrievable by search call")
        else:
            needed_top_k = current_doc_ids.index(read_doc_id) + 1
            if needed_top_k > int(search_args.get("top_k", 5)):
                search_args["top_k"] = needed_top_k
                messages[search_call_index]["content"] = serialize_call("search_official_docs", search_args)
                repairs.append(f"widened search top_k to {needed_top_k}")

        fresh_search = search_payload(search_args, docs)
        fresh_read = read_payload(read_args, sections)
        messages[results[0][0]]["content"] = json.dumps(fresh_search, ensure_ascii=False)
        messages[results[1][0]]["content"] = json.dumps(fresh_read, ensure_ascii=False)

        if not fresh_search.get("documents"):
            errors.append("current search returns no documents")
        if not fresh_read.get("sections"):
            errors.append("current read returns no sections")
        returned_doc_ids = {doc.get("doc_id") for doc in fresh_search.get("documents", [])}
        if read_doc_id not in returned_doc_ids:
            errors.append(f"read doc {read_doc_id} is not in current search results")

        section_ids = {section.get("section_id") for section in fresh_read.get("sections", [])}
        for section_id in row.get("section_ids", []):
            if section_id not in section_ids:
                errors.append(f"section_id {section_id} is not in current read results")

        evidence = evidence_text(fresh_read)
        for fact in row.get("expected_facts", []):
            if str(fact).lower() not in evidence:
                errors.append(f"expected fact {fact!r} not in current read evidence")
        if row.get("row_family") in {"tool_grounded", "query_rewrite_tool_grounded"}:
            for claim in exact_claims(str(row.get("target_response", ""))):
                if claim.lower() not in evidence and not claim_is_negative_myth(claim, str(row.get("target_response", ""))):
                    errors.append(f"exact claim {claim!r} not in current read evidence")

        repaired = dict(row)
        repaired["messages"] = messages
        return repaired, errors, repairs

    if calls or results:
        errors.append("no-tool row contains tool trace")
    return row, errors, repairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and repair Beacon tool traces against the current local doc tool.")
    parser.add_argument(
        "package_dir",
        type=Path,
        default=ROOT / "data" / "assistant_sft" / "beacon_tool_plus_no_tool_sft_v1_final_reviewed",
        nargs="?",
    )
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data" / "local_grounding" / "official_doc_tool_v1")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    docs = load_doc_index(args.index_dir)
    sections = load_section_index(args.index_dir)
    rows = [json.loads(line) for line in (args.package_dir / "all_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    repaired_rows: list[dict[str, Any]] = []
    row_errors: list[dict[str, Any]] = []
    row_repairs: list[dict[str, Any]] = []

    for row in rows:
        repaired, errors, repairs = repair_row(row, docs, sections)
        repaired_rows.append(repaired)
        if errors:
            row_errors.append({"row_id": row.get("row_id"), "errors": errors})
        if repairs:
            row_repairs.append({"row_id": row.get("row_id"), "repairs": repairs})

    manifest = {
        "status": "valid" if not row_errors else "invalid",
        "package_dir": str(args.package_dir),
        "index_dir": str(args.index_dir),
        "row_count": len(rows),
        "tool_required_count": sum(bool(row.get("tool_required")) for row in rows),
        "no_tool_count": sum(not bool(row.get("tool_required")) for row in rows),
        "assistant_tool_call_count": sum(
            1
            for row in repaired_rows
            for message in row.get("messages", [])
            if message.get("role") == "assistant" and str(message.get("content", "")).startswith("<tool_call>")
        ),
        "tool_result_count": sum(1 for row in repaired_rows for message in row.get("messages", []) if message.get("role") == "tool"),
        "repair_count": len(row_repairs),
        "repair_types": dict(Counter(repair for item in row_repairs for repair in item["repairs"]).most_common()),
        "error_count": len(row_errors),
        "errors": row_errors[:200],
    }

    if args.write:
        write_jsonl(args.package_dir / "all_rows.jsonl", repaired_rows)
        for split in ["train", "dev", "final_eval"]:
            write_jsonl(args.package_dir / f"{split}.jsonl", [row for row in repaired_rows if row.get("split") == split])
        write_json(args.package_dir / "tool_trace_validation_report.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    if row_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
