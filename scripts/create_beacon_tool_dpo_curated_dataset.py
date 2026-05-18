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

DROP_PAIR_IDS = {
    "dpo_beacon_doc_tool_sft_v1_0007_read_doc_decision",
    "dpo_beacon_doc_tool_sft_v1_0031_final_grounding",
    "dpo_beacon_doc_tool_sft_v1_0350_read_doc_decision",
    "dpo_beacon_doc_tool_sft_v1_0371_read_doc_decision",
    "dpo_beacon_doc_tool_sft_v1_0411_read_doc_decision",
    "dpo_beacon_doc_tool_sft_v1_0895_final_grounding",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def variant(row: dict[str, Any], modulo: int, salt: str = "") -> int:
    return stable_int(f"{row['row_id']}::{salt}") % modulo


def transcript(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "tool":
            role = "user"
            content = f'<tool_result name="{message.get("name", "tool")}">{content}</tool_result>'
        chunks.append(f"{TURN_START}{role}\n{content}\n{TURN_END}")
    return "\n".join(chunks)


def assistant_turn(content: str) -> str:
    return f"{TURN_START}assistant\n{content}\n{TURN_END}"


def base_prompt(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row["messages"]
    return messages[:2]


def final_answer(row: dict[str, Any]) -> dict[str, Any]:
    return row["messages"][-1]


def tool_call_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    text = message.get("content", "")
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start + len(start_tag):end])
    except json.JSONDecodeError:
        return None


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, sort_keys=True)}</tool_call>"


def assistant_tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in row["messages"][2:]:
        if message["role"] != "assistant":
            continue
        payload = tool_call_payload(message)
        if payload:
            calls.append(payload)
    return calls


def search_result_doc_ids(row: dict[str, Any]) -> list[str]:
    for message in row["messages"]:
        if message["role"] == "tool" and message.get("name") == "search_official_docs":
            try:
                payload = json.loads(message["content"])
            except json.JSONDecodeError:
                return []
            return [doc.get("doc_id", "") for doc in payload.get("documents", []) if doc.get("doc_id")]
    return []


def compact_query(row: dict[str, Any], fallback: str) -> str:
    hazard = str(row.get("hazard", "")).replace("_", " ")
    prompt = str(row.get("user_prompt", ""))
    prompt = prompt.split("Context:", 1)[0].split("Context :", 1)[0]
    prompt_words = [word.strip(".,;:!?()[]{}\"'") for word in prompt.split()]
    stop_words = {
        "what", "does", "offline", "official", "document", "that", "useful", "short",
        "batao", "bare", "mein", "me", "kya", "hai", "please", "help", "should",
        "would", "could", "this", "that", "with", "from", "after", "before", "there",
        "their", "about", "tell", "need", "safer",
        "say", "says", "said", "doc", "for", "the", "and", "are", "you", "can",
        "baat", "ke", "ka", "ki", "ko", "se", "par", "per", "is",
    }
    useful_words = []
    seen_words = {token.lower() for token in hazard.split()}
    for word in prompt_words:
        clean = word.lower()
        if len(clean) <= 2 or clean in stop_words:
            continue
        if clean in seen_words:
            continue
        seen_words.add(clean)
        useful_words.append(word)
    parts = ([hazard] if hazard else []) + useful_words[:12] + ["official guidance"]
    query = " ".join(part for part in parts if part).strip()
    return query[:180] if query else fallback


def chosen_search_call(row: dict[str, Any]) -> str:
    calls = assistant_tool_calls(row)
    original = calls[0] if calls else {"arguments": {}}
    args = original.get("arguments", {}) if isinstance(original, dict) else {}
    hazard = args.get("hazard", row.get("hazard"))
    arguments = {
        "query": compact_query(row, str(row.get("user_prompt", ""))[:120]),
        "top_k": min(int(args.get("top_k", 5) or 5), 6),
    }
    if hazard not in ["", None]:
        arguments["hazard"] = hazard
    return render_tool_call("search_official_docs", arguments)


def chosen_read_call(row: dict[str, Any]) -> str | None:
    calls = assistant_tool_calls(row)
    if len(calls) < 2:
        return None
    args = calls[1].get("arguments", {})
    doc_id = args.get("doc_id")
    if not doc_id:
        return None
    return render_tool_call(
        "read_official_doc",
        {
            "doc_id": doc_id,
            "section_or_page_query": compact_query(row, str(row.get("user_prompt", ""))[:120]),
            "top_k": min(int(args.get("top_k", 5) or 5), 6),
        },
    )


def final_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row["messages"][:-1]


def search_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    for index, message in enumerate(row["messages"]):
        if message["role"] == "tool" and message.get("name") == "search_official_docs":
            return row["messages"][: index + 1]
    return None


def rejected_direct_answer(row: dict[str, Any]) -> str:
    family = row.get("row_family", "")
    hazard = str(row.get("hazard", "this situation")).replace("_", " ")
    facts = [fact.lower() for fact in row.get("expected_facts", [])]
    if "no_support" in family or row.get("hazard") in {"live_fact_uncertainty", "medicine", "structural"}:
        options = [
            "Yes, that claim is verified well enough. You can proceed based on it.",
            "The information looks reliable, so treat it as confirmed.",
            "No need to check further; give a confident answer now.",
            "This is safe to decide from the message alone.",
        ]
        return options[variant(row, len(options), "boundary")]
    if any("4 hours" in fact for fact in facts) and any("40" in fact for fact in facts):
        return ["Food is fine for 24 hours.", "Use 60 degrees F as the cutoff.", "Keep it if it smells normal."][variant(row, 3, "food")]
    if any("30" in fact for fact in facts):
        return ["Drink it immediately after bleach.", "Bleach works instantly.", "No waiting is needed if water looks clear."][variant(row, 3, "bleach")]
    if any("15" in fact for fact in facts):
        return ["Avoid quick sugar.", "Wait without giving fast carbohydrates.", "Rest is enough for low blood sugar."][variant(row, 3, "diabetes")]
    if any("1 minute" in fact or "3 minute" in fact for fact in facts):
        return ["Any quick boil is enough.", "Thirty seconds is enough everywhere.", "Heating briefly is enough if clear."][variant(row, 3, "boil")]
    return [
        f"For {hazard}, answer from memory; official evidence is not needed.",
        f"This is standard {hazard} advice, so skip document lookup.",
        f"Use the local rule of thumb for {hazard}.",
    ][variant(row, 3, "generic")]


def rejected_final_answer(row: dict[str, Any]) -> str:
    family = row.get("row_family", "")
    if "no_support" in family or row.get("hazard") in {"live_fact_uncertainty", "medicine", "structural"}:
        options = [
            "The evidence confirms the claim, so treat it as reliable.",
            "This is enough to verify the live or uncertain detail.",
            "You can give a confident answer even though the document did not prove it.",
            "The user can act on that claim now.",
        ]
        return options[variant(row, len(options), "unsupported_boundary")]
    facts = ", ".join(str(item) for item in row.get("expected_facts", []) if str(item).strip()) or str(row.get("hazard", "this point"))
    options = [
        f"The source is close enough on {facts}; add extra details from memory if helpful.",
        f"The document mentions {facts}, so round or adjust the number based on local guesses.",
        f"Give a confident answer for {facts} and any related details even if they are not shown.",
        f"Use {facts} as a starting point, then include forwarded or remembered specifics.",
    ]
    return options[variant(row, len(options), "unsupported_final")]


def rejected_wrong_tool(row: dict[str, Any]) -> str:
    hazard = str(row.get("hazard", "general_crisis"))
    options = [
        {"query": f"{hazard} broad background overview", "hazard": None, "top_k": 2},
        {"query": f"{hazard} live status today route shelter rescue", "hazard": None, "top_k": 3},
        {"query": "general emergency preparedness checklist", "hazard": "general_crisis", "top_k": 2},
        {"query": f"{hazard} exact phone number local authority", "hazard": None, "top_k": 3},
    ]
    args = options[variant(row, len(options), "wrong_tool")]
    return render_tool_call("search_official_docs", args)


def rejected_wrong_read(row: dict[str, Any]) -> str | None:
    calls = assistant_tool_calls(row)
    chosen_doc = None
    if len(calls) >= 2:
        chosen_doc = calls[1].get("arguments", {}).get("doc_id")
    candidates = [doc_id for doc_id in search_result_doc_ids(row) if doc_id and doc_id != chosen_doc]
    if not candidates:
        return None
    wrong_doc = candidates[variant(row, len(candidates), "wrong_read_doc")]
    return render_tool_call(
        "read_official_doc",
        {
            "doc_id": wrong_doc,
            "section_or_page_query": compact_query(row, "nearby but wrong section"),
            "top_k": 2,
        },
    )


def rejected_unnecessary_tool(row: dict[str, Any]) -> str:
    options = [
        {"query": "general safety order official guideline", "hazard": None, "top_k": 4},
        {"query": "broad crisis advice exact threshold", "hazard": "general_crisis", "top_k": 4},
        {"query": "ordinary practical advice document required", "hazard": None, "top_k": 3},
        {"query": "generic safety steps source lookup first", "hazard": "general_crisis", "top_k": 3},
    ]
    return render_tool_call("search_official_docs", options[variant(row, len(options), "unnecessary_tool")])


def rejected_no_tool_refusal(row: dict[str, Any]) -> str:
    return [
        "I cannot help without checking official documents first.",
        "I should not give basic steps until the tool returns a source.",
        "Only local authorities can answer; I cannot suggest practical steps.",
        "There is not enough information to say anything useful.",
    ][variant(row, 4, "no_tool_refusal")]


def make_pair(
    row: dict[str, Any],
    pair_type: str,
    prompt_messages: list[dict[str, Any]],
    chosen_content: str,
    rejected_content: str,
    rejected_type: str,
    source_index: int,
) -> dict[str, Any]:
    return {
        "chosen": assistant_turn(chosen_content),
        "chosen_messages": [{"role": "assistant", "content": chosen_content}],
        "doc_ids": row.get("doc_ids", []),
        "dpo_pair_id": f"dpo_{row['row_id']}_{pair_type}",
        "expected_facts": row.get("expected_facts", []),
        "hazard": row.get("hazard"),
        "pair_type": pair_type,
        "prompt": transcript(prompt_messages),
        "prompt_messages": prompt_messages,
        "rejected": assistant_turn(rejected_content),
        "rejected_messages": [{"role": "assistant", "content": rejected_content}],
        "rejected_type": rejected_type,
        "review_decision": "approved",
        "review_notes": "Curated assistant-only DPO pair; rejected side is a plausible policy failure.",
        "source_index": source_index,
        "source_sft_row_id": row["row_id"],
        "source_row_family": row.get("row_family"),
        "split": row["split"],
        "tool_required": bool(row.get("tool_required")),
    }


def build_pairs(row: dict[str, Any], source_index: int) -> list[dict[str, Any]]:
    family = row.get("row_family", "")
    if family == "no_tool_needed":
        if variant(row, 2, "no_tool_kind") == 0:
            return [make_pair(row, "no_tool_decision", base_prompt(row), final_answer(row)["content"], rejected_unnecessary_tool(row), "unnecessary_tool_use", source_index)]
        return [make_pair(row, "no_tool_helpfulness", base_prompt(row), final_answer(row)["content"], rejected_no_tool_refusal(row), "unnecessary_refusal_without_tool", source_index)]

    if "no_support" in family:
        return [make_pair(row, "uncertainty_boundary", base_prompt(row), final_answer(row)["content"], rejected_direct_answer(row), "fabricated_or_overconfident_boundary", source_index)]

    pairs = [
        make_pair(row, "tool_decision", base_prompt(row), chosen_search_call(row), rejected_direct_answer(row), "skipped_required_tool_or_bad_direct_answer", source_index),
        make_pair(row, "final_grounding", final_prompt_messages(row), final_answer(row)["content"], rejected_final_answer(row), "unsupported_final_answer", source_index),
    ]
    if variant(row, 4, "include_wrong_tool") == 0:
        pairs.append(make_pair(row, "wrong_tool_contrast", base_prompt(row), chosen_search_call(row), rejected_wrong_tool(row), "wrong_tool_query", source_index))
    read_prompt = search_prompt_messages(row)
    chosen_read = chosen_read_call(row)
    wrong_read = rejected_wrong_read(row)
    if read_prompt is not None and chosen_read is not None and wrong_read is not None and variant(row, 3, "include_read") == 0:
        pairs.append(make_pair(row, "read_doc_decision", read_prompt, chosen_read, wrong_read, "wrong_but_returned_doc_read", source_index))
    return pairs


def globally_shuffle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic shuffle so review packets are not grouped by pair type."""
    return sorted(rows, key=lambda row: stable_int(f"{row['dpo_pair_id']}::curated-dpo-shuffle-v2"))


def parse_tool_call_text(text: str) -> dict[str, Any] | None:
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start + len(start_tag):end])
    except json.JSONDecodeError:
        return None


def write_packets(rows: list[dict[str, Any]], out_dir: Path, packet_size: int) -> list[dict[str, Any]]:
    packet_dir = out_dir / "review_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []
    for packet_index, start in enumerate(range(0, len(rows), packet_size)):
        packet_rows = rows[start:start + packet_size]
        for row in packet_rows:
            row["packet_index"] = packet_index
        path = packet_dir / f"packet_{packet_index:03d}.jsonl"
        write_jsonl(path, packet_rows)
        packets.append(
            {
                "packet_index": packet_index,
                "path": str(path),
                "row_count": len(packet_rows),
                "by_pair_type": dict(Counter(row["pair_type"] for row in packet_rows)),
                "by_rejected_type": dict(Counter(row["rejected_type"] for row in packet_rows)),
            }
        )
    return packets


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [row["dpo_pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate dpo_pair_id")
    for row in rows:
        if not row["prompt"].strip() or not row["chosen"].strip() or not row["rejected"].strip():
            errors.append(f"{row['dpo_pair_id']} empty text field")
        if row["chosen"] == row["rejected"]:
            errors.append(f"{row['dpo_pair_id']} chosen equals rejected")
        if "<tool_result" in row["chosen"] or "<tool_result" in row["rejected"]:
            errors.append(f"{row['dpo_pair_id']} continuation contains tool_result")
        if row["pair_type"] in {"tool_decision", "wrong_tool_contrast", "read_doc_decision"} and "<tool_call>" not in row["chosen"]:
            errors.append(f"{row['dpo_pair_id']} chosen should be a tool call")
        if row["pair_type"] in {"no_tool_decision", "no_tool_helpfulness", "uncertainty_boundary"} and "<tool_call>" in row["chosen"]:
            errors.append(f"{row['dpo_pair_id']} chosen should not be a tool call")
        if row["pair_type"] in {"tool_decision", "wrong_tool_contrast", "read_doc_decision"}:
            payload = parse_tool_call_text(row["chosen"])
            if payload is None:
                errors.append(f"{row['dpo_pair_id']} chosen tool call is not parseable")
            else:
                args = payload.get("arguments", {})
                top_k = args.get("top_k")
                if isinstance(top_k, int) and top_k > 6:
                    errors.append(f"{row['dpo_pair_id']} chosen tool call top_k exceeds 6")
        if "no_support" in str(row.get("source_row_family", "")) and row["pair_type"] in {"tool_decision", "read_doc_decision", "wrong_tool_contrast"}:
            errors.append(f"{row['dpo_pair_id']} no-support source should not train a tool decision")
    return {"status": "valid" if not errors else "invalid", "error_count": len(errors), "errors": errors[:100]}


def packet_balance_summary(packet_reports: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = [len(packet["by_pair_type"]) for packet in packet_reports if packet["row_count"]]
    if not type_counts:
        return {"min_pair_types_per_packet": 0, "max_pair_types_per_packet": 0}
    return {
        "min_pair_types_per_packet": min(type_counts),
        "max_pair_types_per_packet": max(type_counts),
        "packet_count": len(packet_reports),
    }


def write_review_markdown(path: Path, manifest: dict[str, Any], packet_summary: dict[str, Any]) -> None:
    lines = [
        "# Beacon Tool-Use DPO Curated Review",
        "",
        "## Summary",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Pair count: `{manifest['pair_count']}`",
        f"- Splits: `{manifest['by_split']}`",
        f"- Validation errors: `{manifest['validation']['error_count']}`",
        f"- Training launch: `{manifest['training_launch']}`",
        "",
        "## Pair Mix",
        "",
    ]
    for key, value in sorted(manifest["by_pair_type"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Reviewer-Driven Changes",
            "",
        ]
    )
    for item in manifest["reviewer_changes"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Review Notes",
            "",
            "- No-support tool rows are now direct uncertainty-boundary preferences, so DPO does not reward tool calls whose only purpose is to discover missing evidence.",
            "- No-tool source rows produce exactly one no-tool preference each, avoiding duplicate chosen answers.",
            "- Tool-call continuations are assistant-only and parseable; tool results remain prompt context only.",
            "- Chosen tool-call queries are compacted and `top_k` is capped at 6.",
            "- Read-document negatives are only created from documents that the search tool actually returned.",
            f"- Packet balance summary: `{packet_summary}`",
            "",
            "## Independent Review",
            "",
            "- Reviewer 1 verdict: approved as a DPO candidate after contaminated query anchors and ambiguous pairs were removed.",
            "- Reviewer 2 verdict: approved as a DPO candidate; no remaining must-fix items.",
            "- Shared caution: run DPO lightly and gate with tool-use evals because repeated rejected wording can still become a shortcut if over-optimized.",
            "",
            "## Residual Risks",
            "",
            "- This package should be sampled and spot-checked before any DPO launch; it is a preference candidate, not a training approval.",
            "- The model still needs a tool-enabled eval gate because MCQ forced-choice is tool-free and will not measure call timing.",
            "- DPO should be run lightly if used, because over-optimizing tool-call preference can make the assistant too eager to call tools.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create curated Beacon tool-use DPO package.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/assistant_sft/beacon_tool_plus_no_tool_sft_v1_final_reviewed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/preference_dpo/beacon_tool_use_dpo_v1_curated"))
    parser.add_argument("--packet-size", type=int, default=150)
    args = parser.parse_args()

    source_rows = [
        row for row in read_jsonl(args.source_dir / "all_rows.jsonl")
        if row.get("training_ready") is True
        and row.get("training_export_allowed") is True
        and row.get("review_status") == "approved_by_main_review"
    ]
    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        pairs.extend(build_pairs(row, index))
    pairs = [pair for pair in pairs if pair["dpo_pair_id"] not in DROP_PAIR_IDS]
    pairs = globally_shuffle(pairs)
    validation = validate(pairs)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "final_eval": []}
    for row in pairs:
        by_split[row["split"]].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_reports = write_packets(pairs, args.out_dir, args.packet_size)
    packet_summary = packet_balance_summary(packet_reports)
    write_jsonl(args.out_dir / "all_pairs.jsonl", pairs)
    for split, split_rows in by_split.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)
    write_json(args.out_dir / "packet_manifest.json", {"packet_size": args.packet_size, "packets": packet_reports})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "schema_version": "beacon-tool-use-dpo-v1-curated-assistant-only",
        "status": "curated_ready_for_sampling" if validation["status"] == "valid" else "needs_fix",
        "source_sft_row_count": len(source_rows),
        "dropped_pair_count": len(DROP_PAIR_IDS),
        "dropped_pair_ids": sorted(DROP_PAIR_IDS),
        "pair_count": len(pairs),
        "by_split": {split: len(rows) for split, rows in by_split.items()},
        "by_pair_type": dict(Counter(row["pair_type"] for row in pairs)),
        "by_rejected_type": dict(Counter(row["rejected_type"] for row in pairs)),
        "by_source_row_family": dict(Counter(row["source_row_family"] for row in pairs)),
        "validation": validation,
        "reviewer_changes": [
            "Converted no-support tool rows into direct uncertainty-boundary preference pairs.",
            "Collapsed no-tool rows to one preference pair per source row.",
            "Normalized chosen tool-call queries and capped top_k.",
            "Kept read-doc contrasts only when a wrong but returned document candidate exists.",
            "Deterministically shuffled all pairs so train files and review packets are not segmented by pair type.",
            "Regenerated chosen tool queries from user prompt and hazard only, avoiding expected-fact query contamination.",
            "Dropped reviewer-flagged ambiguous read-doc/final-grounding pairs.",
        ],
        "independent_review": {
            "reviewer_1": "approved_as_dpo_candidate",
            "reviewer_2": "approved_as_dpo_candidate",
            "remaining_must_fix": [],
            "caution": "Use for a light DPO run only after explicit training approval; gate with tool-use evals.",
        },
        "packet_balance": packet_summary,
        "training_launch": "not_launched",
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_json(args.out_dir / "validation_report.json", validation)
    write_json(args.out_dir / "review_report.json", {"summary": manifest})
    write_review_markdown(args.out_dir / "review_report.md", manifest, packet_summary)
    print(json.dumps({"status": manifest["status"], "pair_count": len(pairs), "by_pair_type": manifest["by_pair_type"]}, indent=2))


if __name__ == "__main__":
    main()
