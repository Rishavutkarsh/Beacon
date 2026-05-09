from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


UNSAFE_CERTAINTY = [
    re.compile(r"\bdefinitely safe\b", re.I),
    re.compile(r"\b100%\s*safe\b", re.I),
    re.compile(r"\bguaranteed\b", re.I),
    re.compile(r"\bboiling removes all chemical", re.I),
    re.compile(r"\bsmell(s)? fine.*safe\b", re.I),
    re.compile(r"\breheat(ing)?.*always.*safe\b", re.I),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten(item) for item in value.values())
    return str(value)


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def row_errors(row: dict[str, Any], hard_eval: bool) -> list[str]:
    errors: list[str] = []
    required = ["example_id", "user_prompt", "assistant_response", "source_ids", "guidance_fact_ids", "risk_tags", "counterintuitive_mistake", "hazard_category"]
    for key in required:
        if not row.get(key):
            errors.append(f"missing {key}")
    answer = row.get("assistant_response") or {}
    for key in ["risk_level", "immediate_action", "resource_plan", "unsafe_items", "missing_information", "escalation_signs", "what_not_to_do", "hindi_hinglish", "uncertainty_note"]:
        if key not in answer:
            errors.append(f"missing answer.{key}")
    answer_text = flatten(answer)
    for pattern in UNSAFE_CERTAINTY:
        if pattern.search(answer_text):
            errors.append(f"unsafe certainty: {pattern.pattern}")
    for phrase in ["shelter is open", "road is safe", "rescue will arrive", "official warning says", "call 911", "call 112"]:
        if phrase in answer_text.lower():
            errors.append(f"possible fabricated live/phone fact: {phrase}")
    if row.get("visual_attention_required") and not row.get("visual_context"):
        errors.append("visual_attention_required without visual_context")
    if row.get("scenario_seed_id") in {"rumor_relief_food", "shelter_rumor_route"}:
        answer_text = flatten(answer).lower()
        if "floodwater" not in answer_text and "unverified" not in answer_text:
            errors.append("rumor/route row must explicitly mention floodwater or unverified status")
    if row.get("scenario_seed_id") == "toy_floodwater":
        answer_text = flatten(answer).lower()
        if "porous" not in answer_text or "hard" not in answer_text:
            errors.append("toy floodwater row must distinguish porous/soft from hard washable toys")
    if row.get("scenario_seed_id") == "medicine_wet_strip" and answer.get("risk_level") != "high":
        errors.append("wet critical medicine row must use high risk")
    rubric = row.get("eval_rubric") or {}
    if hard_eval:
        for key in ["must_mention", "must_not_say", "difficulty_reason", "visual_attention_required", "source_grounding_required"]:
            if key not in rubric:
                errors.append(f"hard eval missing rubric.{key}")
        if row.get("visual_attention_required") and not rubric.get("image_visible_labels"):
            errors.append("hard eval missing image_visible_labels")
        if row.get("visual_attention_required") and not rubric.get("image_not_determinable_labels"):
            errors.append("hard eval missing image_not_determinable_labels")
    return errors


def provenance_errors(row: dict[str, Any], facts: dict[str, Any], sources: set[str], incidents: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fact_ids = row.get("guidance_fact_ids") or []
    row_source_ids = set(row.get("source_ids") or [])
    for fact_id in fact_ids:
        fact = facts.get(fact_id)
        if not fact:
            errors.append(f"unknown guidance fact: {fact_id}")
            continue
        fact_sources = set(fact.get("source_ids") or [])
        missing_sources = sorted(fact_sources - sources)
        if missing_sources:
            errors.append(f"guidance fact {fact_id} references missing sources: {missing_sources}")
        if not fact_sources.issubset(row_source_ids):
            errors.append(f"row source_ids missing sources for fact {fact_id}")
        if fact.get("source_ready") is not True:
            errors.append(f"guidance fact not source_ready: {fact_id}")
    for source_id in row_source_ids:
        if source_id not in sources:
            errors.append(f"row references unknown source: {source_id}")
    for incident_id in row.get("incident_pattern_ids") or []:
        incident = incidents.get(incident_id)
        if not incident:
            errors.append(f"unknown incident pattern: {incident_id}")
            continue
        guidance_ids = set(incident.get("guidance_ids") or [])
        if guidance_ids and not guidance_ids.intersection(fact_ids):
            errors.append(f"incident {incident_id} guidance_ids do not overlap row facts")
    return errors


def build_report(dataset_dir: Path) -> dict[str, Any]:
    train = read_jsonl(dataset_dir / "sft_text.jsonl")
    eval_rows = read_jsonl(dataset_dir / "eval.jsonl")
    sources = {row.get("source_id") for row in read_jsonl(dataset_dir / "sources.jsonl") if row.get("source_id")}
    facts = {row.get("fact_id"): row for row in read_jsonl(dataset_dir / "guidance_facts.jsonl") if row.get("fact_id")}
    incidents = {row.get("incident_id"): row for row in read_jsonl(dataset_dir / "incident_patterns.jsonl") if row.get("incident_id")}
    all_rows = [*train, *eval_rows]
    exact_prompts = Counter(normalized(row.get("user_prompt", "")) for row in all_rows)
    duplicate_prompts = [prompt for prompt, count in exact_prompts.items() if prompt and count > 1]
    train_prompts = {normalized(row.get("user_prompt", "")) for row in train}
    eval_prompts = {normalized(row.get("user_prompt", "")) for row in eval_rows}
    errors_by_id: dict[str, list[str]] = {}
    for row in train:
        errors = [*row_errors(row, hard_eval=False), *provenance_errors(row, facts, sources, incidents)]
        if errors:
            errors_by_id[row.get("example_id", "<missing>")] = errors
    for row in eval_rows:
        errors = [*row_errors(row, hard_eval=True), *provenance_errors(row, facts, sources, incidents)]
        if errors:
            errors_by_id[row.get("example_id", "<missing>")] = errors
    hazard_counts = Counter(row.get("hazard_category", "unknown") for row in train)
    disaster_counts = Counter(row.get("disaster_type", "unknown") for row in train)
    source_counts = Counter(source_id for row in all_rows for source_id in row.get("source_ids", []))
    language_counts = Counter(row.get("language_mix", "unknown") for row in train)
    seed_clusters: dict[str, int] = Counter(row.get("scenario_seed_id", "unknown") for row in train)
    cluster_over_limit = {seed: count for seed, count in seed_clusters.items() if count > 80}
    eval_seed_clusters: dict[str, int] = Counter(row.get("scenario_seed_id", "unknown") for row in eval_rows)
    train_seed_ids = set(seed_clusters)
    eval_seed_ids = set(eval_seed_clusters)
    train_visual_contexts = Counter(row.get("visual_context", "") for row in train if row.get("visual_attention_required"))
    eval_visual_contexts = Counter(row.get("visual_context", "") for row in eval_rows if row.get("visual_attention_required"))
    train_immediate_actions = Counter(normalized(flatten((row.get("assistant_response") or {}).get("immediate_action", []))) for row in train)
    eval_immediate_actions = Counter(normalized(flatten((row.get("assistant_response") or {}).get("immediate_action", []))) for row in eval_rows)
    train_pattern_counts = Counter(
        (
            row.get("hazard_category", ""),
            normalized(row.get("visual_context", "")),
            normalized(flatten((row.get("assistant_response") or {}).get("immediate_action", []))),
        )
        for row in train
    )
    eval_pattern_counts = Counter(
        (
            row.get("hazard_category", ""),
            normalized(row.get("visual_context", "")),
            normalized(flatten((row.get("assistant_response") or {}).get("immediate_action", []))),
        )
        for row in eval_rows
    )
    max_train_pattern_repeat = max(train_pattern_counts.values(), default=0)
    max_eval_pattern_repeat = max(eval_pattern_counts.values(), default=0)
    visual_train = sum(bool(row.get("visual_attention_required")) for row in train)
    visual_eval = sum(bool(row.get("visual_attention_required")) for row in eval_rows)
    counter_train = sum(bool(row.get("counterintuitive_mistake")) for row in train)
    counter_eval = sum(bool(row.get("counterintuitive_mistake")) for row in eval_rows)
    max_hazard_share = max(hazard_counts.values(), default=0) / max(1, len(train))
    max_disaster_share = max(disaster_counts.values(), default=0) / max(1, len(train))
    report = {
        "counts": {
            "train": len(train),
            "eval": len(eval_rows),
            "visual_attention_train": visual_train,
            "visual_attention_eval": visual_eval,
            "counterintuitive_train": counter_train,
            "counterintuitive_eval": counter_eval,
        },
        "balance": {
            "hazard_counts": hazard_counts.most_common(),
            "disaster_counts": disaster_counts.most_common(),
            "language_counts": language_counts.most_common(),
            "max_hazard_share": max_hazard_share,
            "max_disaster_share": max_disaster_share,
            "source_counts": source_counts.most_common(),
            "seed_clusters_over_limit": cluster_over_limit,
            "eval_seed_count": len(eval_seed_ids),
            "train_eval_seed_overlap": sorted(train_seed_ids.intersection(eval_seed_ids)),
            "max_train_visual_context_repeat": max(train_visual_contexts.values(), default=0),
            "max_eval_visual_context_repeat": max(eval_visual_contexts.values(), default=0),
            "train_unique_visual_context_ratio": len(train_visual_contexts) / max(1, visual_train),
            "eval_unique_visual_context_ratio": len(eval_visual_contexts) / max(1, visual_eval),
            "max_train_immediate_action_repeat": max(train_immediate_actions.values(), default=0),
            "max_eval_immediate_action_repeat": max(eval_immediate_actions.values(), default=0),
            "max_train_hazard_visual_action_pattern_repeat": max_train_pattern_repeat,
            "max_eval_hazard_visual_action_pattern_repeat": max_eval_pattern_repeat,
        },
        "duplicates": {
            "exact_duplicate_prompt_count": len(duplicate_prompts),
            "train_eval_prompt_overlap": len(train_prompts.intersection(eval_prompts)),
        },
        "row_error_count": sum(len(errors) for errors in errors_by_id.values()),
        "row_errors_sample": dict(list(errors_by_id.items())[:30]),
        "thresholds": {
            "train_at_least_1000": len(train) >= 1000,
            "eval_at_least_120": len(eval_rows) >= 120,
            "visual_attention_train_at_least_250": visual_train >= 250,
            "counterintuitive_train_at_least_300": counter_train >= 300,
            "hard_eval_counterintuitive_at_least_half": counter_eval >= int(len(eval_rows) * 0.5),
            "hard_eval_visual_at_least_40_percent": visual_eval >= int(len(eval_rows) * 0.4),
            "max_hazard_share_at_most_20_percent": max_hazard_share <= 0.20,
            "max_disaster_share_at_most_20_percent": max_disaster_share <= 0.20,
            "no_exact_duplicate_prompts": not duplicate_prompts,
            "no_train_eval_prompt_overlap": not train_prompts.intersection(eval_prompts),
            "no_train_eval_seed_overlap": not train_seed_ids.intersection(eval_seed_ids),
            "eval_unique_seeds_at_least_12": len(eval_seed_ids) >= 12,
            "train_visual_contexts_mostly_unique": len(train_visual_contexts) / max(1, visual_train) >= 0.80,
            "eval_visual_contexts_mostly_unique": len(eval_visual_contexts) / max(1, visual_eval) >= 0.90,
            "max_visual_context_repeat_at_most_3": max(max(train_visual_contexts.values(), default=0), max(eval_visual_contexts.values(), default=0)) <= 3,
            "max_hazard_visual_action_pattern_repeat_at_most_3": max(max_train_pattern_repeat, max_eval_pattern_repeat) <= 3,
            "no_row_errors": not errors_by_id,
            "no_large_seed_clusters": not cluster_over_limit,
        },
    }
    report["passed"] = all(report["thresholds"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report and gate the high-quality Sankat Saathi dataset.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    report = build_report(dataset_dir)
    out_path = dataset_dir / "high_quality_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit("high-quality dataset report failed thresholds")


if __name__ == "__main__":
    main()
