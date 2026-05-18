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


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


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
    if len(messages) < 3 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"Unexpected message shape for {row.get('row_id')}")
    return messages[:2]


def final_answer(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Last message is not assistant for {row.get('row_id')}")
    return messages[-1]


def first_tool_call(row: dict[str, Any]) -> dict[str, Any]:
    for message in row["messages"][2:]:
        if message["role"] == "assistant" and "<tool_call>" in message["content"]:
            return message
    raise ValueError(f"No assistant tool call for {row.get('row_id')}")


def second_tool_call(row: dict[str, Any]) -> dict[str, Any] | None:
    calls = [message for message in row["messages"][2:] if message["role"] == "assistant" and "<tool_call>" in message["content"]]
    return calls[1] if len(calls) > 1 else None


def final_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row["messages"][:-1]


def expected_text(row: dict[str, Any]) -> str:
    facts = [str(item) for item in row.get("expected_facts", []) if str(item).strip()]
    if facts:
        return ", ".join(facts)
    return str(row.get("hazard", "this case")).replace("_", " ")


def variant(row: dict[str, Any], modulo: int, salt: str = "") -> int:
    return stable_int(f"{row['row_id']}::{salt}") % modulo


def rejected_direct_answer(row: dict[str, Any]) -> str:
    family = row.get("row_family", "")
    hazard = str(row.get("hazard", "this situation")).replace("_", " ")
    facts = [fact.lower() for fact in row.get("expected_facts", [])]
    if "no_support" in family or row.get("hazard") == "live_fact_uncertainty":
        options = [
            "Yes, that live update is verified. The route or shelter should be usable now, so you can proceed.",
            "The forward sounds current, so treat the bridge and rescue update as reliable.",
            "You can trust that message for now; it confirms the local status well enough.",
            "No need to check further. Assume the shelter or route status in the message is accurate.",
        ]
        return options[variant(row, len(options), "live_direct")]
    if any("4 hours" in fact for fact in facts) and any("40" in fact for fact in facts):
        options = [
            "Food in the fridge should be fine for 24 hours, and 60 degrees F is the threshold that matters.",
            "A closed fridge is usually safe all day; throw food only if it smells bad.",
            "Use 60 degrees F as the cutoff and keep perishables unless they look spoiled.",
        ]
        return options[variant(row, len(options), "food_direct")]
    if any("30" in fact for fact in facts):
        options = [
            "You can drink it right away after adding bleach if the water looks clear.",
            "Bleach works instantly, so there is no need to wait before drinking.",
            "If the water smells slightly of chlorine, drink it immediately.",
        ]
        return options[variant(row, len(options), "bleach_direct")]
    if any("15" in fact for fact in facts):
        options = [
            "Do not use quick sugar for low blood sugar. Wait and see if the person improves.",
            "Avoid fast carbohydrates; they can make the situation worse.",
            "Skip sugar and focus only on resting until help arrives.",
        ]
        return options[variant(row, len(options), "diabetes_direct")]
    if any("1 minute" in fact or "3 minute" in fact for fact in facts):
        options = [
            "Boiling time is not important; any quick boil is enough at all altitudes.",
            "Thirty seconds of boiling is enough everywhere.",
            "If the water looks clear, heating it briefly is enough.",
        ]
        return options[variant(row, len(options), "boil_direct")]
    options = [
        f"I can answer this from memory. For {hazard}, use a local rule of thumb; no document lookup is needed.",
        f"This is standard {hazard} advice, so do not spend time checking documents.",
        f"For {hazard}, rely on what people nearby usually do; exact official wording is not needed.",
    ]
    return options[variant(row, len(options), "generic_direct")]


def rejected_wrong_tool(row: dict[str, Any]) -> str:
    selector = variant(row, 4, "wrong_tool")
    if selector == 0:
        return '<tool_call>{"name":"search_official_docs","arguments":{"query":"general emergency advice","hazard":"general_crisis","top_k":1}}</tool_call>'
    if selector == 1:
        return '<tool_call>{"name":"search_official_docs","arguments":{"query":"nearest open shelter and live rescue arrival now","hazard":null,"top_k":3}}</tool_call>'
    if selector == 2:
        return '<tool_call>{"name":"search_official_docs","arguments":{"query":"medicine dose exact prescription identity","hazard":"medicine_safety","top_k":3}}</tool_call>'
    return '<tool_call>{"name":"search_official_docs","arguments":{"query":"weather warning status today route open phone number","hazard":null,"top_k":3}}</tool_call>'


def rejected_wrong_read(row: dict[str, Any]) -> str:
    options = [
        '<tool_call>{"name":"read_official_doc","arguments":{"doc_id":"ready_heat","section_or_page_query":"unrelated generic emergency advice","top_k":1}}</tool_call>',
        '<tool_call>{"name":"read_official_doc","arguments":{"doc_id":"cdc_food_after_emergency","section_or_page_query":"live route open rescue arrival now","top_k":1}}</tool_call>',
        '<tool_call>{"name":"read_official_doc","arguments":{"doc_id":"ndma_heat_wave","section_or_page_query":"food refrigerator outage freezer hours","top_k":1}}</tool_call>',
        '<tool_call>{"name":"read_official_doc","arguments":{"doc_id":"ready_floods","section_or_page_query":"medicine dose identity insulin correction amount","top_k":1}}</tool_call>',
    ]
    return options[variant(row, len(options), "wrong_read")]


def rejected_final_answer(row: dict[str, Any]) -> str:
    family = row.get("row_family", "")
    if "no_support" in family or row.get("hazard") == "live_fact_uncertainty":
        options = [
            "The document confirms the live message, so treat the bridge, shelter, or rescue update as reliable.",
            "Because a document was found, the current local status in the forward is probably true.",
            "The source supports proceeding as if the bridge, shelter, or rescue update is current.",
            "This evidence is enough to say the live claim is verified.",
        ]
        return options[variant(row, len(options), "live_final")]
    facts = expected_text(row)
    options = [
        (
            f"The document is about {facts}, but you can also rely on nearby guesses and forwarded updates. "
            "If the situation feels urgent, choose the fastest option even if the document did not support it."
        ),
        (
            f"The document mentions {facts}. It is fine to add extra practical numbers from memory if they sound familiar."
        ),
        (
            f"The evidence points roughly toward {facts}, so you can round it or adjust it based on what neighbors say."
        ),
        (
            f"The source is close enough on {facts}; give the user a confident answer even for details not shown."
        ),
    ]
    return options[variant(row, len(options), "final")]


def rejected_unnecessary_tool(row: dict[str, Any]) -> str:
    options = [
        '<tool_call>{"name":"search_official_docs","arguments":{"query":"broad general crisis advice exact official threshold","hazard":"general_crisis","top_k":5}}</tool_call>',
        '<tool_call>{"name":"search_official_docs","arguments":{"query":"general safety order official guideline source required","hazard":null,"top_k":5}}</tool_call>',
        '<tool_call>{"name":"search_official_docs","arguments":{"query":"ordinary practical crisis advice document lookup before answering","hazard":"general_crisis","top_k":3}}</tool_call>',
        '<tool_call>{"name":"search_official_docs","arguments":{"query":"non exact broad advice official citations mandatory","hazard":null,"top_k":4}}</tool_call>',
    ]
    return options[variant(row, len(options), "unnecessary_tool")]


def rejected_no_tool_over_refusal(row: dict[str, Any]) -> str:
    options = [
        "I cannot help without checking official documents first. Please wait until the document tool is available.",
        "This needs official source lookup before any useful steps can be given.",
        "I should not give even basic safety steps unless I can retrieve a document.",
        "There is not enough information to help. Only local authorities can answer.",
    ]
    return options[variant(row, len(options), "no_tool_refusal")]


def pair(
    row: dict[str, Any],
    pair_type: str,
    prompt_messages: list[dict[str, Any]],
    chosen_content: str,
    rejected_content: str,
    rejected_type: str,
    packet_index: int,
) -> dict[str, Any]:
    return {
        "chosen": assistant_turn(chosen_content),
        "chosen_messages": [{"role": "assistant", "content": chosen_content}],
        "doc_ids": row.get("doc_ids", []),
        "dpo_pair_id": f"dpo_{row['row_id']}_{pair_type}",
        "expected_facts": row.get("expected_facts", []),
        "hazard": row.get("hazard"),
        "packet_index": packet_index,
        "pair_type": pair_type,
        "prompt": transcript(prompt_messages),
        "prompt_messages": prompt_messages,
        "rejected": assistant_turn(rejected_content),
        "rejected_messages": [{"role": "assistant", "content": rejected_content}],
        "rejected_type": rejected_type,
        "review_decision": "approved",
        "review_notes": review_notes(row, pair_type, rejected_type),
        "row_family": row.get("row_family"),
        "source_sft_row_id": row["row_id"],
        "split": row["split"],
        "tool_required": bool(row.get("tool_required")),
    }


def review_notes(row: dict[str, Any], pair_type: str, rejected_type: str) -> str:
    if pair_type == "tool_decision":
        return "Approved: chosen is the reviewed first tool call; rejected either skips evidence or uses a poor tool path."
    if pair_type == "read_doc_decision":
        return "Approved: chosen reads the reviewed document/section path after search; rejected reads an unrelated document."
    if pair_type == "final_grounding":
        return "Approved: prompt contains tool evidence; chosen uses bounded supported facts; rejected adds unsupported or live claims."
    if pair_type == "no_tool_decision":
        return "Approved: broad advice should answer directly; rejected overuses the document tool."
    if pair_type == "no_tool_helpfulness":
        return "Approved: broad advice should stay useful without documents; rejected withholds basic safe steps."
    return f"Approved preference contrast: {rejected_type}."


def build_pairs(row: dict[str, Any], packet_index: int) -> list[dict[str, Any]]:
    if row.get("tool_required"):
        pairs = [
            pair(
                row,
                "tool_decision",
                base_prompt(row),
                first_tool_call(row)["content"],
                rejected_direct_answer(row),
                "skipped_required_tool_or_fabricated_answer",
                packet_index,
            ),
            pair(
                row,
                "final_grounding",
                final_prompt_messages(row),
                final_answer(row)["content"],
                rejected_final_answer(row),
                "unsupported_final_answer",
                packet_index,
            ),
        ]
        second = second_tool_call(row)
        if second is not None and variant(row, 2, "include_read") == 0:
            search_result_index = next(
                index for index, message in enumerate(row["messages"]) if message["role"] == "tool" and message.get("name") == "search_official_docs"
            )
            pairs.append(
                pair(
                    row,
                    "read_doc_decision",
                    row["messages"][: search_result_index + 1],
                    second["content"],
                    rejected_wrong_read(row),
                    "wrong_read_doc_call",
                    packet_index,
                )
            )
        if variant(row, 5, "include_wrong_tool") == 0:
            pairs.append(
                pair(
                    row,
                    "wrong_tool_contrast",
                    base_prompt(row),
                    first_tool_call(row)["content"],
                    rejected_wrong_tool(row),
                    "wrong_tool_query_or_live_lookup",
                    packet_index,
                )
            )
        return pairs
    return [
        pair(
            row,
            "no_tool_decision",
            base_prompt(row),
            final_answer(row)["content"],
            rejected_unnecessary_tool(row),
            "unnecessary_tool_use",
            packet_index,
        ),
        pair(
            row,
            "no_tool_helpfulness",
            base_prompt(row),
            final_answer(row)["content"],
            rejected_no_tool_over_refusal(row),
            "unnecessary_refusal_without_tool",
            packet_index,
        ),
    ]


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
        if "<tool_result" in row["chosen"] or "<tool_result" in row["rejected"]:
            errors.append(f"{row['dpo_pair_id']} continuation contains tool_result")
        if row["pair_type"] in {"tool_decision", "wrong_tool_contrast"} and "<tool_call>" not in row["chosen"]:
            errors.append(f"{row['dpo_pair_id']} tool decision chosen lacks tool_call")
        if row["pair_type"] in {"no_tool_decision", "no_tool_helpfulness"} and "<tool_call>" in row["chosen"]:
            errors.append(f"{row['dpo_pair_id']} no-tool chosen contains tool_call")
    return {"status": "valid" if not errors else "invalid", "error_count": len(errors), "errors": errors[:100]}


def write_packets(rows: list[dict[str, Any]], out_dir: Path, packet_size: int) -> list[dict[str, Any]]:
    packet_dir = out_dir / "review_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for packet_index in sorted({row["packet_index"] for row in rows}):
        packet_rows = [row for row in rows if row["packet_index"] == packet_index]
        path = packet_dir / f"packet_{packet_index:03d}.jsonl"
        write_jsonl(path, packet_rows)
        reports.append(
            {
                "packet_index": packet_index,
                "path": str(path),
                "row_count": len(packet_rows),
                "by_pair_type": dict(Counter(row["pair_type"] for row in packet_rows)),
                "by_rejected_type": dict(Counter(row["rejected_type"] for row in packet_rows)),
            }
        )
    write_json(out_dir / "packet_manifest.json", {"packet_size_source_rows": packet_size, "packets": reports})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reviewed Beacon assistant-only DPO pairs.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/assistant_sft/beacon_tool_plus_no_tool_sft_v1_final_reviewed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/preference_dpo/beacon_tool_use_dpo_v1_reviewed"))
    parser.add_argument("--packet-size", type=int, default=100)
    args = parser.parse_args()

    source_rows = [
        row for row in read_jsonl(args.source_dir / "all_rows.jsonl")
        if row.get("training_ready") is True
        and row.get("training_export_allowed") is True
        and row.get("review_status") == "approved_by_main_review"
    ]

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        packet_index = index // args.packet_size
        pairs.extend(build_pairs(row, packet_index))

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "final_eval": []}
    for item in pairs:
        by_split[item["split"]].append(item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "all_pairs.jsonl", pairs)
    for split, split_rows in by_split.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)
    packet_reports = write_packets(pairs, args.out_dir, args.packet_size)
    validation = validate(pairs)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "schema_version": "beacon-tool-use-dpo-v1-reviewed-assistant-only",
        "status": "reviewed_ready_for_sampling" if validation["status"] == "valid" else "needs_fix",
        "source_sft_row_count": len(source_rows),
        "pair_count": len(pairs),
        "by_split": {split: len(rows) for split, rows in by_split.items()},
        "by_pair_type": dict(Counter(row["pair_type"] for row in pairs)),
        "by_rejected_type": dict(Counter(row["rejected_type"] for row in pairs)),
        "tool_required_source_rows": sum(1 for row in source_rows if row.get("tool_required")),
        "no_tool_source_rows": sum(1 for row in source_rows if not row.get("tool_required")),
        "packet_count": len(packet_reports),
        "validation": validation,
        "notes": [
            "This reviewed package replaces full-trajectory DPO with assistant-only continuations.",
            "Tool-result turns may appear in prompts for final-grounding decisions, but never in chosen/rejected continuations.",
            "Every pair has an explicit review_decision and review_notes field.",
            "No DPO/PPO/GRPO run has been launched.",
        ],
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_json(args.out_dir / "validation_report.json", validation)
    write_json(
        args.out_dir / "review_report.json",
        {
            "summary": manifest,
            "review_policy": {
                "approve": "Chosen continuation is reviewed behavior and rejected continuation is a clear policy failure.",
                "reject": "Unused in this generated package; rows with structural issues fail validation instead.",
            },
        },
    )
    print(json.dumps({"status": manifest["status"], "pair_count": len(pairs), "by_pair_type": manifest["by_pair_type"]}, indent=2))


if __name__ == "__main__":
    main()
