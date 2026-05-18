from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TURN_START = "<|turn>"
TURN_END = "<turn|>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def transcript(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            name = message.get("name", "tool")
            role = "user"
            content = f'<tool_result name="{name}">{message["content"]}</tool_result>'
        else:
            content = message["content"]
        chunks.append(f"{TURN_START}{role}\n{content}\n{TURN_END}")
    return "\n".join(chunks)


def split_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = row["messages"]
    if len(messages) < 3 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"Unexpected message shape for {row.get('row_id')}")
    return messages[:2], messages[2:]


def generic_rejected_answer(row: dict[str, Any]) -> str:
    hazard = str(row.get("hazard", "this situation")).replace("_", " ")
    expected = [str(item) for item in row.get("expected_facts", []) if str(item).strip()]
    if expected:
        return (
            "I do not need to check the offline documents for this. "
            f"For {hazard}, use the usual rule of thumb and decide locally; the exact number is not important right now."
        )
    return (
        "I can answer this from memory without checking the offline documents. "
        "Follow the forwarded advice if it sounds familiar, and ask people nearby to confirm later."
    )


def fabricated_live_answer(row: dict[str, Any]) -> str:
    return (
        "Yes, this looks verified: the route or shelter mentioned in the message should be usable now. "
        "You can proceed based on that update."
    )


def unsupported_constant_answer(row: dict[str, Any]) -> str:
    expected = [str(item) for item in row.get("expected_facts", []) if str(item).strip()]
    if any("40" in fact for fact in expected) and any("4" in fact for fact in expected):
        return "Food in the fridge should be fine for about 24 hours, and 60 degrees F is the main threshold to watch."
    if any("30" in fact for fact in expected):
        return "After adding bleach, the water can be used immediately if it looks clear."
    if any("15" in fact for fact in expected):
        return "For low blood sugar, do not give quick sugar; wait and see if the person improves."
    if any("1 minute" in fact.lower() or "3 minute" in fact.lower() for fact in expected):
        return "Boiling time does not matter much; a quick boil is enough at any altitude."
    return generic_rejected_answer(row)


def wrong_tool_trajectory(row: dict[str, Any]) -> list[dict[str, Any]]:
    query = row.get("tool_query") or row.get("user_prompt") or "general emergency advice"
    return [
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"search_official_docs","arguments":{"query":"general emergency advice unrelated source","hazard":"general_crisis","top_k":1}}</tool_call>',
        },
        {
            "role": "tool",
            "name": "search_official_docs",
            "content": '{"documents":[]}',
        },
        {
            "role": "assistant",
            "content": (
                "The documents did not return anything, but I will answer anyway from memory. "
                f"For this case, use a rough local guess for: {query[:120]}"
            ),
        },
    ]


def unnecessary_tool_trajectory(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"search_official_docs","arguments":{"query":"broad generic crisis advice exact official threshold","hazard":"general_crisis","top_k":5}}</tool_call>',
        },
        {
            "role": "tool",
            "name": "search_official_docs",
            "content": '{"documents":[{"doc_id":"ready_floods","title":"Floods","organization":"Ready.gov / FEMA","score":0.1}]}',
        },
        {
            "role": "assistant",
            "content": (
                "According to the document, do not act until you can look up more official pages. "
                "I cannot give practical steps without another source."
            ),
        },
    ]


def rejected_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    family = row.get("row_family", "")
    tool_required = bool(row.get("tool_required"))
    if not tool_required:
        return unnecessary_tool_trajectory(row), "unnecessary_tool_use"
    if "no_support" in family or row.get("hazard") == "live_fact_uncertainty":
        return [{"role": "assistant", "content": fabricated_live_answer(row)}], "fabricated_live_or_missing_support"
    if row.get("expected_facts"):
        return [{"role": "assistant", "content": unsupported_constant_answer(row)}], "skipped_tool_wrong_or_unsupported_constant"
    if int(row_hash(row), 16) % 3 == 0:
        return wrong_tool_trajectory(row), "wrong_tool_query_or_doc"
    return [{"role": "assistant", "content": generic_rejected_answer(row)}], "skipped_required_tool"


def build_pair(row: dict[str, Any]) -> dict[str, Any]:
    prompt_messages, chosen_messages = split_messages(row)
    bad_messages, rejected_type = rejected_messages(row)
    prompt_text = transcript(prompt_messages)
    chosen_text = transcript(chosen_messages)
    rejected_text = transcript(bad_messages)
    return {
        "chosen": chosen_text,
        "chosen_messages": chosen_messages,
        "doc_ids": row.get("doc_ids", []),
        "dpo_pair_id": f"dpo_{row['row_id']}",
        "expected_facts": row.get("expected_facts", []),
        "hazard": row.get("hazard"),
        "prompt": prompt_text,
        "prompt_messages": prompt_messages,
        "rejected": rejected_text,
        "rejected_messages": bad_messages,
        "rejected_type": rejected_type,
        "row_family": row.get("row_family"),
        "source_sft_row_id": row["row_id"],
        "split": row["split"],
        "tool_required": bool(row.get("tool_required")),
    }


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [row["dpo_pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate dpo_pair_id")
    for row in rows:
        for field in ["prompt", "chosen", "rejected"]:
            if not row[field].strip():
                errors.append(f"{row['dpo_pair_id']} empty {field}")
        if row["chosen"] == row["rejected"]:
            errors.append(f"{row['dpo_pair_id']} chosen equals rejected")
        if row["tool_required"] and "<tool_call>" not in row["chosen"]:
            errors.append(f"{row['dpo_pair_id']} required tool but chosen has no tool call")
        if not row["tool_required"] and "<tool_call>" in row["chosen"]:
            errors.append(f"{row['dpo_pair_id']} no-tool chosen contains tool call")
    return {
        "status": "valid" if not errors else "invalid",
        "error_count": len(errors),
        "errors": errors[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Beacon tool-use DPO preference pairs from approved SFT rows.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/assistant_sft/beacon_tool_plus_no_tool_sft_v1_final_reviewed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/preference_dpo/beacon_tool_use_dpo_v1"))
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_dir / "all_rows.jsonl")
    ready_rows = [
        row for row in source_rows
        if row.get("training_ready") is True
        and row.get("training_export_allowed") is True
        and row.get("review_status") == "approved_by_main_review"
    ]
    pairs = [build_pair(row) for row in ready_rows]
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "final_eval": []}
    for pair in pairs:
        by_split[pair["split"]].append(pair)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "all_pairs.jsonl", pairs)
    for split, rows in by_split.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", rows)

    validation = validate(pairs)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "schema_version": "beacon-tool-use-dpo-v1",
        "status": "ready_for_review" if validation["status"] == "valid" else "needs_fix",
        "row_count": len(pairs),
        "by_split": {split: len(rows) for split, rows in by_split.items()},
        "by_row_family": dict(Counter(row["row_family"] for row in pairs)),
        "by_rejected_type": dict(Counter(row["rejected_type"] for row in pairs)),
        "tool_required_count": sum(1 for row in pairs if row["tool_required"]),
        "no_tool_count": sum(1 for row in pairs if not row["tool_required"]),
        "validation": validation,
        "notes": [
            "Chosen continuations come from approved Beacon SFT rows.",
            "Rejected continuations are synthetic preference contrasts for tool-use policy, not new gold answers.",
            "Use for DPO-style preference tuning only after review; do not treat rejected text as factual supervision.",
        ],
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_json(args.out_dir / "validation_report.json", validation)
    print(json.dumps({"status": manifest["status"], "row_count": len(pairs), "by_split": manifest["by_split"]}, indent=2))


if __name__ == "__main__":
    main()
