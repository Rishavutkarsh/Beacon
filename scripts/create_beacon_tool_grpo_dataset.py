from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_DIR = ROOT / "data" / "assistant_sft" / "beacon_tool_plus_no_tool_sft_v1_final_reviewed"
DEFAULT_OUT_DIR = ROOT / "data" / "rl_grpo" / "beacon_tool_use_grpo_v1"
DEFAULT_SYSTEM_PROMPT = ROOT / "prompts" / "beacon_tool_system_prompt_v1.md"
TURN_START = "<|turn>"
TURN_END = "<turn|>"
SCHEMA_VERSION = "beacon-tool-grpo-v1"
TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
TOOL_RESULT_RE = re.compile(r"<tool_result\b.*?</tool_result>", re.S)
COMPACT_GRPO_SYSTEM_PROMPT = """You are Beacon, an offline crisis companion.

You may emit exactly one tool call when needed. The tool call must be a JSON object inside <tool_call> tags with name search_official_docs and arguments query, hazard, organization, and top_k.

Use search_official_docs when the user asks what offline/official documents say, or asks for exact official facts, thresholds, durations, quantities, warning categories, source-sensitive rules, or document-grounded claims.

Do not use tools for ordinary practical help, emotional support, broad common-sense safety priorities, live/current status verification, diagnosis, medicine identification/dosing, or building/photo safety certification. For live/current status, say offline documents cannot verify it.

For this first-turn training task, decide whether to answer directly or call search_official_docs. If calling a tool, output only the tool_call block. Never output Returns text, tool-result schemas, invented documents, placeholder values, or schema words as argument values."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcript(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{TURN_START}{msg['role']}\n{msg['content']}\n{TURN_END}" for msg in messages)


def clean_system_prompt(path: Path, fallback: str) -> str:
    if path == DEFAULT_SYSTEM_PROMPT:
        return COMPACT_GRPO_SYSTEM_PROMPT
    if not path.exists():
        return fallback
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return fallback
    fenced = re.search(r"```text\s*(.*?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    # The markdown prompt is used as runtime documentation. Keep only the compact
    # instruction block before any long implementation notes, if present.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def first_tool_args(messages: list[dict[str, Any]], tool_name: str) -> dict[str, Any]:
    prefix = "<tool_call>"
    suffix = "</tool_call>"
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        if not (content.startswith(prefix) and content.endswith(suffix)):
            continue
        try:
            payload = json.loads(content[len(prefix) : -len(suffix)])
        except json.JSONDecodeError:
            continue
        if payload.get("name") == tool_name and isinstance(payload.get("arguments"), dict):
            return payload["arguments"]
    return {}


def expected_behavior(row: dict[str, Any]) -> str:
    if live_status_no_tool_override(row):
        return "answer_directly_without_tool"
    if row.get("tool_required"):
        return "first_turn_search_tool_call"
    return "answer_directly_without_tool"


def reward_rubric(row: dict[str, Any]) -> dict[str, Any]:
    tool_required = effective_tool_required(row)
    return {
        "version": "beacon-tool-grpo-reward-v1",
        "expected_behavior": expected_behavior(row),
        "scalar_rewards": {
            "right_grounded_answer": 7.0,
            "wrong_answer": -5.0,
            "unsafe_answer": -8.0,
            "wrong_or_failed_tool_call": -3.0,
            "good_keyword_search": 1.0,
            "great_keyword_search": 2.0,
            "search_reward_cap": 3.0,
        },
        "deterministic_checks": [
            "valid_tool_call_json",
            "only_allowed_tool_names",
            "tool_rows_start_with_search",
            "search_query_keyword_quality",
            "tool_call_has_no_extra_prose",
            "no_live_status_hallucination",
            "no_unnecessary_tool_use_for_no_tool_rows",
            "no_placeholder_tool_arguments",
        ],
        "judge_checks": [
            "answer_correct",
            "grounded_in_evidence",
            "unsupported_extra",
            "unsafe",
            "abstention_correct",
        ],
        "tool_required": tool_required,
        "search_reward_is_capped": True,
    }


def live_status_no_tool_override(row: dict[str, Any]) -> bool:
    return str(row.get("hazard", "")) == "live_fact_uncertainty"


def professional_boundary_no_tool_override(row: dict[str, Any]) -> bool:
    prompt = str(row.get("user_prompt", "")).lower()
    hazard = str(row.get("hazard", "")).lower()
    if hazard in {"medicine", "drug_safety"} and re.search(r"\b(identify|which medicine|what medicine|what pill|dose|dosing|tablet|wet tablet|wet medicine|smell)\b", prompt):
        return True
    if hazard in {"structural", "electrical", "landslide"} and re.search(
        r"\b(structure is safe|building.*safe|house.*safe|safe to sleep|slope.*safe|walls? look|photo|picture|wire.*safe|wet wire|certify|inspect)\b",
        prompt,
    ):
        return True
    return False


def effective_tool_required(row: dict[str, Any]) -> bool:
    if live_status_no_tool_override(row):
        return False
    if professional_boundary_no_tool_override(row):
        return False
    return bool(row.get("tool_required"))


def build_row(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    source_messages = row["messages"]
    fallback_system = str(source_messages[0]["content"])
    prompt_messages = [
        {"role": "system", "content": system_prompt or fallback_system},
        {"role": "user", "content": str(row.get("user_prompt") or source_messages[1]["content"])},
    ]
    search_args = first_tool_args(source_messages, "search_official_docs")
    read_args = first_tool_args(source_messages, "read_official_doc")
    doc_ids = list(dict.fromkeys(str(item) for item in row.get("doc_ids", []) if item))
    if read_args.get("doc_id") and str(read_args["doc_id"]) not in doc_ids:
        doc_ids.insert(0, str(read_args["doc_id"]))
    tool_required = effective_tool_required(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": f"grpo_{row['row_id']}",
        "source_sft_row_id": row["row_id"],
        "split": row["split"],
        "hazard": row.get("hazard"),
        "row_family": row.get("row_family"),
        "case_family_id": row.get("case_family_id"),
        "base_scenario_id": row.get("base_scenario_id"),
        "tool_required": tool_required,
        "source_sft_tool_required": bool(row.get("tool_required")),
        "tool_policy_override": "live_status_no_tool"
        if live_status_no_tool_override(row)
        else "professional_boundary_no_tool"
        if professional_boundary_no_tool_override(row)
        else "",
        "query_rewrite_required": bool(row.get("query_rewrite_required")),
        "prompt": transcript(prompt_messages),
        "prompt_messages": prompt_messages,
        "user_prompt": prompt_messages[1]["content"],
        "expected_facts": row.get("expected_facts", []),
        "allowed_doc_ids": doc_ids if tool_required else [],
        "allowed_section_ids": list(dict.fromkeys(str(item) for item in row.get("section_ids", []) if item)) if tool_required else [],
        "gold_tool_query": (row.get("tool_query") or search_args.get("query") or "") if tool_required else "",
        "gold_search_args": search_args if tool_required else {},
        "gold_read_args": read_args if tool_required else {},
        "target_response": row.get("target_response", ""),
        "reward_rubric": reward_rubric(row),
        "source_mix": row.get("mix_source"),
        "training_stage_intent": "full_first_turn_grpo_from_tool_sft_adapter",
        "training_export_allowed": True,
    }


def validate_rows(rows: list[dict[str, Any]], sft_manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    row_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("row_id", ""))
        if not row_id:
            errors.append("row missing row_id")
            continue
        if row_id in row_ids:
            errors.append(f"{row_id}: duplicate row_id")
        row_ids.add(row_id)
        prompt = str(row.get("prompt", ""))
        if TOOL_RESULT_RE.search(prompt):
            errors.append(f"{row_id}: prompt leaks tool result")
        gold_query = str(row.get("gold_tool_query", ""))
        if gold_query and gold_query in prompt:
            errors.append(f"{row_id}: prompt leaks gold tool query")
        if str(row.get("target_response", "")) and str(row.get("target_response", "")) in prompt:
            errors.append(f"{row_id}: prompt leaks target response")
        if row.get("tool_required"):
            if not row.get("gold_tool_query"):
                errors.append(f"{row_id}: tool row missing gold_tool_query")
            if not row.get("allowed_doc_ids"):
                errors.append(f"{row_id}: tool row missing allowed_doc_ids")
            if not row.get("gold_search_args"):
                errors.append(f"{row_id}: tool row missing gold_search_args")
            if not row.get("gold_read_args"):
                errors.append(f"{row_id}: tool row missing gold_read_args")
        else:
            if row.get("gold_search_args") or row.get("gold_read_args"):
                errors.append(f"{row_id}: no-tool row has gold tool args")
            if row.get("gold_tool_query"):
                warnings.append(f"{row_id}: no-tool row has gold_tool_query")
    by_split = Counter(str(row.get("split", "")) for row in rows)
    expected_by_split = sft_manifest.get("by_split", {})
    for split in ["train", "dev", "final_eval"]:
        if by_split.get(split, 0) != expected_by_split.get(split, by_split.get(split, 0)):
            errors.append(f"{split}: row count {by_split.get(split, 0)} does not match SFT manifest {expected_by_split.get(split)}")
    return {
        "created_at_utc": utc_now(),
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "row_count": len(rows),
        "by_split": dict(by_split),
        "by_row_family": dict(Counter(str(row.get("row_family", "")) for row in rows).most_common()),
        "tool_required_count": sum(bool(row.get("tool_required")) for row in rows),
        "no_tool_count": sum(not bool(row.get("tool_required")) for row in rows),
        "policy_override_counts": dict(Counter(str(row.get("tool_policy_override", "")) for row in rows if row.get("tool_policy_override")).most_common()),
        "query_rewrite_count": sum(bool(row.get("query_rewrite_required")) for row in rows),
        "prompt_tool_result_leak_count": sum(1 for row in rows if TOOL_RESULT_RE.search(str(row.get("prompt", "")))),
    }


def render_report(manifest: dict[str, Any], validation: dict[str, Any]) -> str:
    lines = [
        "# Beacon Tool-Use GRPO v1",
        "",
        "Prompt-only RL dataset derived from the final reviewed Beacon tool/no-tool SFT dataset.",
        "",
        f"- Status: `{validation['status']}`",
        f"- Rows: {validation['row_count']}",
        f"- Tool-required rows: {validation['tool_required_count']}",
        f"- No-tool rows: {validation['no_tool_count']}",
        f"- Training stage intent: `{manifest['training_stage_intent']}`",
        f"- Training export allowed: `{manifest['training_export_allowed']}`",
        "",
        "## Validation",
        "",
    ]
    if not validation["errors"] and not validation["warnings"]:
        lines.append("- No validation issues.")
    for error in validation["errors"]:
        lines.append(f"- ERROR: {error}")
    for warning in validation["warnings"]:
        lines.append(f"- WARNING: {warning}")
    lines.extend(
        [
            "",
            "## Reward Shape",
            "",
            "- Deterministic rewards check first-turn tool/no-tool choice, search-call syntax, query quality, no extra prose in tool-call completions, no live-status hallucination, and no placeholder tool arguments.",
            "- Semantic rewards are reserved for an optional strict-JSON Qwen judge outside the first memory-conservative GPU loop.",
            "- Search quality can add at most +3 total reward; extra searches do not keep adding reward.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_dataset(sft_dir: Path, out_dir: Path, system_prompt_path: Path) -> dict[str, Any]:
    sft_manifest = read_json(sft_dir / "manifest.json")
    source_rows = []
    for split in ["train", "dev", "final_eval"]:
        source_rows.extend(read_jsonl(sft_dir / f"{split}.jsonl"))
    system_prompt = clean_system_prompt(system_prompt_path, "")
    rows = [build_row(row, system_prompt) for row in source_rows]
    validation = validate_rows(rows, sft_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "all_rows.jsonl", rows)
    for split in ["train", "dev", "final_eval"]:
        write_jsonl(out_dir / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    manifest = {
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "status": validation["status"],
        "source_sft_dir": str(sft_dir),
        "source_sft_manifest_status": sft_manifest.get("status"),
        "source_sft_training_approval": sft_manifest.get("training_approval"),
        "training_stage_intent": "Full first-turn GRPO starts from the best tool-aware SFT adapter; later environment/tool-loop GRPO can train read/final-answer behavior.",
        "training_export_allowed": True,
        "training_approval": "user_approved_2026-05-18_training_export",
        "row_count": validation["row_count"],
        "by_split": validation["by_split"],
        "by_row_family": validation["by_row_family"],
        "tool_required_count": validation["tool_required_count"],
        "no_tool_count": validation["no_tool_count"],
        "policy_override_counts": validation["policy_override_counts"],
        "query_rewrite_count": validation["query_rewrite_count"],
        "reward_contract": {
            "version": "beacon-tool-grpo-reward-v1",
            "objective": "first_turn_tool_decision_and_search",
            "tool_required_rows": validation["tool_required_count"],
            "no_tool_rows": validation["no_tool_count"],
            "search_reward_cap": 3.0,
            "notes": [
                "Per-row reward_rubric carries row-specific tool_required.",
                "This dataset does not claim to train read_official_doc or final grounded answer behavior in the first-turn GRPO notebook.",
            ],
        },
        "validation": {"status": validation["status"], "error_count": len(validation["errors"]), "warning_count": len(validation["warnings"])},
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "validation_report.json", validation)
    write_json(
        out_dir / "dataset-metadata.json",
        {
            "title": "Beacon Tool Use GRPO v1",
            "id": "rishavutkarsh/beacon-tool-use-grpo-v1",
            "licenses": [{"name": "CC0-1.0"}],
        },
    )
    (out_dir / "review_report.md").write_text(render_report(manifest, validation), encoding="utf-8")
    hash_manifest = {"created_at_utc": utc_now(), "files": {}}
    for name in [
        "all_rows.jsonl",
        "train.jsonl",
        "dev.jsonl",
        "final_eval.jsonl",
        "manifest.json",
        "validation_report.json",
        "dataset-metadata.json",
    ]:
        path = out_dir / name
        hash_manifest["files"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(out_dir / "hash_manifest.json", hash_manifest)
    return {"manifest": manifest, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Beacon prompt-only GRPO dataset from reviewed SFT rows.")
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()
    result = build_dataset(args.sft_dir, args.out_dir, args.system_prompt)
    print(json.dumps({"out_dir": str(args.out_dir), "status": result["validation"]["status"], "rows": result["validation"]["row_count"]}, indent=2))
    if result["validation"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
