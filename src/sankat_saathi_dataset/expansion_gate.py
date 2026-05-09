from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = {"train", "dev", "final_eval"}
QUALITY_STATUSES = {"generated", "lint_failed", "critic_failed", "accepted", "rejected", "repair_needed"}
REVIEW_STATES = {"generated", "deterministic_pass", "critic_pending", "subagent_reviewed", "approved", "rejected", "frozen"}
RENDERER_STYLES = {
    "urgent_stop_refusal",
    "first_10_minutes_checklist",
    "family_resource_plan",
    "volunteer_triage_plan",
    "low_literacy_hinglish",
    "visual_uncertainty",
    "live_fact_refusal",
    "short_offline_card",
}
RISK_LEVELS = {"low", "medium", "high", "critical"}
EXPANSION_PROFILES = {
    "calibration": {
        "targets": {"train": 50, "dev": 0, "final_eval": 0},
        "max_variants_by_split": {"train": 5, "dev": 0, "final_eval": 0},
        "accepted_count_ranges": None,
    },
    "v1_600": {
        "targets": {"train": 600, "dev": 120, "final_eval": 120},
        "max_variants_by_split": {"train": 5, "dev": 3, "final_eval": 3},
        "accepted_count_ranges": {"train": [600, 600], "dev": [100, 150], "final_eval": [100, 150]},
    },
    "v2_1k": {
        "targets": {"train": 1000, "dev": 120, "final_eval": 120},
        "max_variants_by_split": {"train": 5, "dev": 3, "final_eval": 3},
        "accepted_count_ranges": {"train": [900, 1000], "dev": [100, 150], "final_eval": [100, 150]},
    },
    "v2_1015": {
        "targets": {"train": 1015, "dev": 120, "final_eval": 120},
        "max_variants_by_split": {"train": 5, "dev": 3, "final_eval": 3},
        "accepted_count_ranges": {"train": [1015, 1015], "dev": [120, 120], "final_eval": [120, 120]},
        "expected_seed_counts": {"train": 203, "dev": 40, "final_eval": 40},
        "final_eval_isolation": "strict",
    },
}
HIGH_RISK_HAZARDS = {"electrical_wet_devices", "diabetes_medication", "route_rescue_live_fact"}
HIGH_RISK_RULES = {
    "electrical_flood_hazard",
    "wet_device_reenergizing",
    "downed_line_distance",
    "diabetes_disrupted_meals",
    "insulin_storage_uncertainty",
    "damaged_medicine_label",
    "flood_crossing_turn_around",
    "live_fact_uncertainty",
    "unsafe_rescue_self_protection",
}
REQUIRED_ROW_FIELDS = {
    "candidate_id",
    "generation_run_id",
    "row_id",
    "parent_row_id",
    "seed_id",
    "seed_family_id",
    "incident_archetype_id",
    "scenario_cluster_id",
    "split",
    "hazard_domain",
    "risk_level",
    "renderer_style",
    "pattern_contract_id",
    "prompt",
    "target_response",
    "source_rule_ids",
    "must_say_rule_ids",
    "must_not_say_rule_ids",
    "target_behavior_tags",
    "forbidden_behavior_tags",
    "quality_status",
    "generation_attempt",
    "repair_attempt",
    "review_state",
    "prompt_template_version",
    "source_rule_snapshot_hash",
    "seed_snapshot_hash",
    "created_by",
    "prompt_config_hash",
    "generator_config_hash",
    "contract_hash",
    "content_hash",
}
LIST_FIELDS = {
    "source_rule_ids",
    "must_say_rule_ids",
    "must_not_say_rule_ids",
    "target_behavior_tags",
    "forbidden_behavior_tags",
}
PROHIBITED_PATTERNS = {
    "ai_disclaimer": re.compile(r"\bas an ai\b|\bi am an ai\b|\bas a language model\b", re.I),
    "phone_number": re.compile(r"(?:\+?\d[\s-]?){8,}"),
    "medication_dose": re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|iu)\b", re.I),
    "definite_safety": re.compile(r"\b(definitely|guaranteed|100%)\s+safe\b|\bsafe to (drink|eat|touch|enter|cross)\b", re.I),
    "live_status_claim": re.compile(
        r"\b(?:road|bridge|shelter|rescue|warning|dam|weather)\s+(?:is|are)\s+(?:open|closed|safe|available|coming|clear|fine)\b",
        re.I,
    ),
    "diagnosis_claim": re.compile(r"\b(?:this is|you have|they have)\s+(?:carbon monoxide poisoning|infection|sepsis|hypoglycemia)\b", re.I),
}
ESCALATION_ONLY = re.compile(r"^(?:call|contact|go to|reach)\s+(?:emergency|authorit|police|ambulance|doctor|hospital)", re.I)
REFUSAL_MARKERS = re.compile(r"\b(?:cannot|can't|can not|unable|i won't|i cannot)\b", re.I)
NUMBERED_LIST_FIRST = re.compile(r"^\s*(?:1[\).\:]|\d+\s*[-:])")
ACTION_VERBS = {
    "avoid",
    "boil",
    "call",
    "check",
    "clean",
    "cover",
    "discard",
    "do",
    "drink",
    "enter",
    "evacuate",
    "keep",
    "leave",
    "move",
    "open",
    "rinse",
    "separate",
    "stay",
    "stop",
    "touch",
    "use",
    "wait",
}


@dataclass
class GateResult:
    status: str
    errors: list[str]
    warnings: list[str]
    reports: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def hash_rows(rows: list[dict[str, Any]]) -> str:
    return stable_hash(rows)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_first_sentence(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return normalize_text(first)


def token_jaccard(left: str, right: str) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def first_tokens(text: str, count: int = 12) -> str:
    return " ".join(normalize_text(text).split()[:count])


def bullet_shape(text: str) -> str:
    shape = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+[\).\:]", stripped):
            shape.append("N")
        elif stripped.startswith(("-", "*")):
            shape.append("B")
        elif stripped.endswith(":"):
            shape.append("H")
        else:
            shape.append("P")
    return "".join(shape[:12]) or "P"


def heading_sequence(text: str) -> str:
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and len(stripped.split()) <= 6:
            headings.append(normalize_text(stripped.rstrip(":")))
    return "|".join(headings[:8])


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def action_sequence(text: str) -> str:
    tokens = normalize_text(text).split()
    return " ".join(token for token in tokens if token in ACTION_VERBS)[:240]


def scenario_cluster(seed: dict[str, Any]) -> str:
    reviewed_group = seed.get("incident_pattern_group_base") or seed.get("incident_pattern_group")
    if reviewed_group:
        return "sc_" + stable_hash(reviewed_group)[:16]
    cluster = {
        "hazard": seed.get("primary_hazard"),
        "context": seed.get("context_constraints"),
        "vulnerable": seed.get("vulnerable_factor"),
        "unsafe": seed.get("unsafe_assumption"),
        "visual": seed.get("visual_cues", []),
        "must": seed.get("source_rule_ids", []),
    }
    return "sc_" + stable_hash(cluster)[:16]


def infer_risk_level(seed: dict[str, Any]) -> str:
    hazard = seed.get("primary_hazard", "")
    rules = set(seed.get("source_rule_ids", []))
    if hazard in HIGH_RISK_HAZARDS or rules & HIGH_RISK_RULES:
        return "high"
    if seed.get("difficulty_tier") == "adversarial":
        return "high"
    if seed.get("privacy_risk") == "high" or seed.get("sensationalism_risk") == "high":
        return "medium"
    return "medium"


def seed_to_prompt(seed: dict[str, Any], variant_index: int) -> str:
    visual = "; ".join(seed.get("visual_cues", [])[:2])
    pressure = seed.get("user_pressure", "people want quick advice")
    context = seed.get("context_constraints", "")
    if variant_index % 4 == 0:
        return f"{context} Unsafe assumption: {seed.get('unsafe_assumption')}. User pressure: {pressure}. What should I do first?"
    if variant_index % 4 == 1:
        return f"I can see {visual}. The assumption is: {seed.get('unsafe_assumption')}. Give offline guidance."
    if variant_index % 4 == 2:
        return f"As a volunteer, help with this situation: {context} Visible cue: {visual}. Pressure: {pressure}."
    return f"Short Hinglish-friendly card for: {seed.get('unsafe_assumption')} Context: {context} Visible cue: {visual}."


def seed_to_response(seed: dict[str, Any], variant_index: int) -> str:
    must = seed.get("must_say", [])
    must_not = seed.get("must_not_say", [])
    escalation = seed.get("escalation_signs", [])
    visible = ", ".join(seed.get("visual_cues", [])[:2]) or "the visible cues"
    unknown = ", ".join(seed.get("visual_not_determinable", [])[:2]) or "the hidden risk"
    style = seed.get("answer_style", "short_offline_card")
    lines: list[str]
    if style == "first_10_minutes_checklist":
        lines = [
            "First 10 minutes:",
            f"1. Stop the unsafe step: {must[0] if must else 'do not treat the situation as safe by default'}.",
            f"2. Use the safer option: {must[1] if len(must) > 1 else 'separate people from the hazard while checking trusted local help'}.",
            f"3. Watch for: {', '.join(escalation[:3]) or 'worsening symptoms or immediate danger'}.",
        ]
    elif style == "family_resource_plan":
        lines = [
            "Family plan:",
            f"One person keeps people away from the risky item or place: {must[0] if must else 'avoid the hazard'}.",
            f"One person saves the safest resources for the vulnerable person: {seed.get('vulnerable_factor', 'vulnerable people')}.",
            f"Do not do this: {must_not[0] if must_not else 'do not guess safety from appearance'}.",
        ]
    elif style == "volunteer_triage_plan":
        lines = [
            "Volunteer triage:",
            f"Immediate danger: {must[0] if must else 'separate people from the hazard'}.",
            f"Watch-list: {', '.join(escalation[:3]) or 'any worsening sign'}.",
            "Routine: record what is uncertain and hand off to trained local responders when reachable.",
        ]
    elif style == "visual_uncertainty":
        lines = [
            f"Visible: {visible}.",
            f"Not knowable from the image: {unknown}.",
            f"Safer action: {must[0] if must else 'do not certify safety from a photo alone'}.",
        ]
    elif style == "live_fact_refusal":
        lines = [
            "I cannot verify live road, rescue, shelter, weather, or warning status from here.",
            f"Use a conservative offline action: {must[0] if must else 'avoid the risky route or action until verified locally'}.",
            "Trust physically verified local information or official channels when reachable.",
        ]
    elif style == "low_literacy_hinglish":
        lines = [
            f"Pehle yeh mat karo: {must_not[0] if must_not else 'risk ko safe mat mano'}.",
            f"Safer kaam: {must[0] if must else 'logon ko hazard se door rakho'}.",
            f"Red flags: {', '.join(escalation[:3]) or 'haalat bigadna'}.",
        ]
    elif style == "urgent_stop_refusal":
        lines = [
            f"Stop: {must[0] if must else 'do not continue the unsafe action'}.",
            f"Safer alternative: {must[1] if len(must) > 1 else 'move away from the hazard and verify before acting'}.",
            f"Do not say or assume: {must_not[0] if must_not else 'that it is definitely safe'}.",
        ]
    else:
        lines = [
            f"Do now: {must[0] if must else 'avoid the unsafe action'}.",
            f"Safer fallback: {must[1] if len(must) > 1 else 'use the lowest-risk available option'}.",
            f"Escalate if: {', '.join(escalation[:3]) or 'danger or symptoms worsen'}.",
        ]
    if variant_index % 3 == 2 and "Do not" not in lines[-1] and must_not:
        lines.append(f"Do not: {must_not[0]}.")
    return "\n".join(lines)


def make_row(
    seed: dict[str, Any],
    variant_index: int,
    created_by: str,
    prompt_config_hash: str,
    *,
    generation_run_id: str = "gen_legacy",
    generator_config_hash: str | None = None,
    source_rule_snapshot_hash: str = "",
    seed_snapshot_hash: str = "",
) -> dict[str, Any]:
    prompt = seed_to_prompt(seed, variant_index)
    target_response = seed_to_response(seed, variant_index)
    split = seed["split"]
    contract = {
        "seed_id": seed["seed_id"],
        "variant_index": variant_index,
        "renderer_style": seed.get("answer_style"),
        "source_rule_ids": seed.get("source_rule_ids", []),
    }
    row_id = f"ss_exp_{split}_{seed['seed_id']}_{variant_index:02d}"
    candidate_id = "cand_" + stable_hash({"generation_run_id": generation_run_id, "row_id": row_id})[:24]
    content = {"prompt": prompt, "target_response": target_response}
    return {
        "candidate_id": candidate_id,
        "generation_run_id": generation_run_id,
        "row_id": row_id,
        "parent_row_id": "",
        "seed_id": seed["seed_id"],
        "seed_family_id": seed["seed_family_id"],
        "incident_archetype_id": seed["incident_archetype_id"],
        "scenario_cluster_id": scenario_cluster(seed),
        "split": split,
        "hazard_domain": seed["primary_hazard"],
        "risk_level": infer_risk_level(seed),
        "renderer_style": seed.get("answer_style", "short_offline_card"),
        "pattern_contract_id": "pc_" + stable_hash(contract)[:16],
        "prompt": prompt,
        "target_response": target_response,
        "source_rule_ids": list(seed.get("source_rule_ids", [])),
        "must_say_rule_ids": list(seed.get("source_rule_ids", [])),
        "must_not_say_rule_ids": list(seed.get("source_rule_ids", [])),
        "target_behavior_tags": [seed["primary_hazard"], seed.get("difficulty_tier", "medium"), seed.get("answer_style", "")],
        "forbidden_behavior_tags": forbidden_tags(seed),
        "quality_status": "generated",
        "generation_attempt": 1,
        "repair_attempt": 0,
        "review_state": "generated",
        "prompt_template_version": "seed_renderer_v1",
        "source_rule_snapshot_hash": source_rule_snapshot_hash,
        "seed_snapshot_hash": seed_snapshot_hash,
        "created_by": created_by,
        "prompt_config_hash": prompt_config_hash,
        "generator_config_hash": generator_config_hash or prompt_config_hash,
        "contract_hash": stable_hash(contract),
        "content_hash": stable_hash(content),
    }


def forbidden_tags(seed: dict[str, Any]) -> list[str]:
    tags = set()
    joined = " ".join([seed.get("primary_hazard", ""), *seed.get("source_rule_ids", []), *seed.get("must_not_say", [])]).lower()
    if "live" in joined or "road" in joined or "rescue" in joined or "shelter" in joined:
        tags.add("live_status")
    if "medicine" in joined or "insulin" in joined or "dose" in joined or "diabetes" in joined:
        tags.add("dose")
    if "photo" in joined or "image" in joined or "visual" in joined:
        tags.add("photo_certainty")
    if "safe" in joined:
        tags.add("unsafe_certainty")
    if "electric" in joined or "wire" in joined:
        tags.add("touch_electrical")
    return sorted(tags) or ["unsafe_certainty"]


def load_rules(rule_manifest: Path) -> dict[str, dict[str, Any]]:
    return {row["rule_id"]: row for row in read_jsonl(rule_manifest)}


def load_seeds(seed_path: Path) -> list[dict[str, Any]]:
    return read_jsonl(seed_path)


def assert_seed_snapshot(
    seeds: list[dict[str, Any]],
    profile_config: dict[str, Any],
) -> list[str]:
    expected = profile_config.get("expected_seed_counts")
    if not expected:
        return []
    counts = Counter(seed.get("split", "") for seed in seeds)
    errors = []
    for split, expected_count in expected.items():
        if counts[split] != expected_count:
            errors.append(f"{split} seed count {counts[split]} does not match expected {expected_count}")
    return errors


def build_rows(
    seed_path: Path,
    out_dir: Path,
    *,
    stage: str,
    profile: str = "calibration",
    train_target: int | None = None,
    dev_target: int | None = None,
    final_target: int | None = None,
    max_variants_per_seed: int = 5,
    max_variants_by_split: dict[str, int] | None = None,
    created_by: str = "sankat_expansion_gate_v1",
    rule_manifest_path: Path | None = None,
) -> dict[str, Any]:
    seeds = load_seeds(seed_path)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        by_split[seed.get("split", "")].append(seed)
    profile_config = EXPANSION_PROFILES.get(profile)
    if profile_config is None:
        raise ValueError(f"unknown expansion profile: {profile}")
    seed_snapshot_errors = assert_seed_snapshot(seeds, profile_config)
    targets = dict(profile_config["targets"])
    if train_target is not None:
        targets["train"] = train_target
    if dev_target is not None:
        targets["dev"] = dev_target
    if final_target is not None:
        targets["final_eval"] = final_target
    split_caps = dict(profile_config["max_variants_by_split"])
    if max_variants_by_split:
        split_caps.update(max_variants_by_split)
    elif max_variants_per_seed != 5:
        split_caps = {split: max_variants_per_seed for split in SPLITS}
    config = {
        "stage": stage,
        "profile": profile,
        "train_target": targets["train"],
        "dev_target": targets["dev"],
        "final_target": targets["final_eval"],
        "max_variants_by_split": split_caps,
        "created_by": created_by,
        "prompt_template_version": "seed_renderer_v1",
        "final_eval_isolation": profile_config.get("final_eval_isolation", "shared"),
    }
    prompt_config_hash = stable_hash(config)
    generator_config_hash = prompt_config_hash
    generation_run_id = f"gen_{profile}_{prompt_config_hash[:12]}"
    seed_snapshot_hash = hash_rows(seeds)
    source_rule_snapshot_hash = sha256_file(rule_manifest_path) if rule_manifest_path else ""
    rows: list[dict[str, Any]] = []
    feasibility_errors: list[str] = [*seed_snapshot_errors]
    for split, target in targets.items():
        if target == 0:
            continue
        split_cap = split_caps.get(split, max_variants_per_seed)
        capacity = len(by_split[split]) * split_cap
        if capacity < target:
            feasibility_errors.append(
                f"{split} target {target} exceeds cap capacity {capacity} ({len(by_split[split])} seeds x {split_cap})"
            )
        split_rows: list[dict[str, Any]] = []
        variant_index = 0
        while len(split_rows) < min(target, capacity):
            progressed = False
            for seed in by_split[split]:
                used_for_seed = sum(1 for row in split_rows if row["seed_id"] == seed["seed_id"])
                if used_for_seed >= split_cap:
                    continue
                split_rows.append(
                    make_row(
                        seed,
                        used_for_seed,
                        created_by,
                        prompt_config_hash,
                        generation_run_id=generation_run_id,
                        generator_config_hash=generator_config_hash,
                        source_rule_snapshot_hash=source_rule_snapshot_hash,
                        seed_snapshot_hash=seed_snapshot_hash,
                    )
                )
                progressed = True
                if len(split_rows) >= min(target, capacity):
                    break
            variant_index += 1
            if not progressed or variant_index > split_cap + 1:
                break
        rows.extend(split_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_derived_artifacts(out_dir)
    write_jsonl(out_dir / "generated_rows.jsonl", rows)
    write_jsonl(out_dir / "repair_lineage.jsonl", [])
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed_path": str(seed_path),
        "seed_snapshot_hash": seed_snapshot_hash,
        "source_rule_manifest_path": str(rule_manifest_path) if rule_manifest_path else "",
        "source_rule_snapshot_hash": source_rule_snapshot_hash,
        "generation_run_id": generation_run_id,
        "generator_config_hash": generator_config_hash,
        "stage": stage,
        "config": config,
        "counts": dict(Counter(row["split"] for row in rows)),
        "row_count": len(rows),
        "feasibility_errors": feasibility_errors,
        "generated_rows_sha256": sha256_file(out_dir / "generated_rows.jsonl"),
    }
    write_json(out_dir / "dataset_manifest.json", manifest)
    return manifest


def clear_derived_artifacts(out_dir: Path) -> None:
    for name in [
        "accepted_rows.jsonl",
        "audit_bundle_manifest.json",
        "behavior_distribution_report.json",
        "cluster_examples.md",
        "critic_report.jsonl",
        "dataset_freeze_manifest.json",
        "deterministic_gate_report.json",
        "final_accepted_rows.jsonl",
        "output_similarity_report.csv",
        "oversized_clusters.csv",
        "pattern_collapse_report.json",
        "per_seed_diversity_report.json",
        "final_eval_isolation_report.json",
        "quota_report.json",
        "rejected_rows.jsonl",
        "rejected_row_ledger.jsonl",
        "review_sampling_manifest.json",
        "reviewer_decisions.jsonl",
        "review_report.json",
        "run_summary.md",
        "safety_lint_report.json",
        "source_claim_support_report.csv",
        "schema_validation_report.json",
        "source_grounding_report.csv",
        "split_leakage_report.json",
        "subagent_review_report.jsonl",
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_schema(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    missing_by_row: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        rid = row.get("row_id", f"<row:{index}>")
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            missing_by_row[rid] = missing
            errors.append(f"{rid}: missing fields {missing}")
        if row.get("split") not in SPLITS:
            errors.append(f"{rid}: invalid split {row.get('split')!r}")
        if row.get("quality_status") not in QUALITY_STATUSES:
            errors.append(f"{rid}: invalid quality_status {row.get('quality_status')!r}")
        if row.get("review_state") not in REVIEW_STATES:
            errors.append(f"{rid}: invalid review_state {row.get('review_state')!r}")
        if row.get("renderer_style") not in RENDERER_STYLES:
            errors.append(f"{rid}: invalid renderer_style {row.get('renderer_style')!r}")
        if row.get("risk_level") not in RISK_LEVELS:
            errors.append(f"{rid}: invalid risk_level {row.get('risk_level')!r}")
        for field in LIST_FIELDS:
            if not isinstance(row.get(field), list) or not row.get(field):
                errors.append(f"{rid}: {field} must be a non-empty list")
        for field in ["generation_attempt", "repair_attempt"]:
            if not isinstance(row.get(field), int) or row.get(field) < 0:
                errors.append(f"{rid}: {field} must be a non-negative integer")
    return errors, {"status": "fail" if errors else "pass", "row_count": len(rows), "missing_by_row": missing_by_row, "errors": errors}


def validate_lineage(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    candidate_ids = Counter(row.get("candidate_id", "") for row in rows)
    row_ids = Counter(row.get("row_id", "") for row in rows)
    duplicate_candidates = sorted(key for key, count in candidate_ids.items() if key and count > 1)
    duplicate_rows = sorted(key for key, count in row_ids.items() if key and count > 1)
    if duplicate_candidates:
        errors.append(f"duplicate candidate_id values: {len(duplicate_candidates)}")
    if duplicate_rows:
        errors.append(f"duplicate row_id values: {len(duplicate_rows)}")
    expected_run = manifest.get("generation_run_id", "")
    expected_seed_hash = manifest.get("seed_snapshot_hash", "")
    expected_rule_hash = manifest.get("source_rule_snapshot_hash", "")
    mismatched_run = [row.get("row_id", "") for row in rows if expected_run and row.get("generation_run_id") != expected_run]
    mismatched_seed_hash = [row.get("row_id", "") for row in rows if expected_seed_hash and row.get("seed_snapshot_hash") != expected_seed_hash]
    mismatched_rule_hash = [
        row.get("row_id", "") for row in rows if expected_rule_hash and row.get("source_rule_snapshot_hash") != expected_rule_hash
    ]
    if mismatched_run:
        errors.append(f"generation_run_id mismatch rows: {len(mismatched_run)}")
    if mismatched_seed_hash:
        errors.append(f"seed_snapshot_hash mismatch rows: {len(mismatched_seed_hash)}")
    if mismatched_rule_hash:
        errors.append(f"source_rule_snapshot_hash mismatch rows: {len(mismatched_rule_hash)}")
    parent_missing = [
        row.get("row_id", "")
        for row in rows
        if row.get("parent_row_id") and row.get("parent_row_id") not in row_ids
    ]
    if parent_missing:
        errors.append(f"parent_row_id references missing rows: {len(parent_missing)}")
    report = {
        "status": "fail" if errors else "pass",
        "row_count": len(rows),
        "unique_candidate_ids": len(candidate_ids),
        "unique_row_ids": len(row_ids),
        "duplicate_candidate_ids": duplicate_candidates[:50],
        "duplicate_row_ids": duplicate_rows[:50],
        "generation_run_id": expected_run,
        "seed_snapshot_hash": expected_seed_hash,
        "source_rule_snapshot_hash": expected_rule_hash,
        "errors": errors,
    }
    return errors, report


def validate_source_grounding(rows: list[dict[str, Any]], rule_manifest: Path, out_csv: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    rules = load_rules(rule_manifest)
    errors: list[str] = []
    audit_rows: list[dict[str, str]] = []
    for row in rows:
        rid = row.get("row_id", "<missing>")
        hazard = row.get("hazard_domain", "")
        for used_as, field in [("source", "source_rule_ids"), ("must_say", "must_say_rule_ids"), ("must_not_say", "must_not_say_rule_ids")]:
            ids = row.get(field, [])
            if not ids:
                errors.append(f"{rid}: empty {field}")
            for rule_id in ids:
                rule = rules.get(rule_id)
                passed = bool(rule)
                if not rule:
                    errors.append(f"{rid}: unknown rule id {rule_id!r} in {field}")
                rule_text = rule.get("derived_rule", "") if rule else ""
                audit_rows.append(
                    {
                        "row_id": rid,
                        "seed_id": row.get("seed_id", ""),
                        "hazard_domain": hazard,
                        "source_rule_id": rule_id,
                        "rule_text_hash": sha256_text(rule_text)[:16] if rule_text else "",
                        "used_as": used_as,
                        "pass_fail": "pass" if passed else "fail",
                    }
                )
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["row_id", "seed_id", "hazard_domain", "source_rule_id", "rule_text_hash", "used_as", "pass_fail"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(audit_rows)
    return errors, {
        "status": "fail" if errors else "pass",
        "known_rule_count": len(rules),
        "audit_rows": len(audit_rows),
        "errors": errors,
    }


def validate_source_claim_support(
    rows: list[dict[str, Any]],
    rule_manifest: Path,
    out_csv: Path,
) -> tuple[list[str], dict[str, Any]]:
    rules = load_rules(rule_manifest)
    errors: list[str] = []
    warnings: list[str] = []
    audit_rows: list[dict[str, str]] = []
    for row in rows:
        rid = row.get("row_id", "<missing>")
        rule_ids = row.get("source_rule_ids", [])
        rule_text = " ".join(rules.get(rule_id, {}).get("derived_rule", "") for rule_id in rule_ids)
        response_sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", row.get("target_response", "")) if part.strip()]
        rule_tokens = set(normalize_text(rule_text).split())
        if not rule_tokens:
            errors.append(f"{rid}: no source text available for material claim support")
        for index, sentence in enumerate(response_sentences):
            normalized = normalize_text(sentence)
            sentence_tokens = set(normalized.split())
            material = bool(sentence_tokens & (ACTION_VERBS | {"safe", "unsafe", "danger", "risk", "escalate", "avoid", "stop", "do", "not"}))
            support_score = len(sentence_tokens & rule_tokens) / max(1, len(sentence_tokens))
            certainty_claim = bool(re.search(r"\b(safe|guaranteed|definitely|because)\b", normalized))
            required_support = 0.2 if certainty_claim else 0.08
            passed = (not material) or support_score >= required_support or any(rule_id in normalized for rule_id in rule_ids)
            audit_rows.append(
                {
                    "row_id": rid,
                    "seed_id": row.get("seed_id", ""),
                    "split": row.get("split", ""),
                    "sentence_index": str(index),
                    "sentence_hash": sha256_text(sentence)[:16],
                    "source_rule_ids": "|".join(rule_ids),
                    "material_claim": str(material).lower(),
                    "support_score": f"{support_score:.3f}",
                    "pass_fail": "pass" if passed else "review",
                }
            )
            if not passed:
                warnings.append(f"{rid}: material claim needs source-support review sentence {index}")
    write_csv_rows(out_csv, audit_rows)
    return [], {
        "status": "review" if warnings else "pass",
        "audit_rows": len(audit_rows),
        "warning_count": len(warnings),
        "warnings": warnings[:100],
        "errors": errors,
    }


def validate_safety(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    hits_by_type: Counter[str] = Counter()
    for row in rows:
        rid = row.get("row_id", "<missing>")
        response = row.get("target_response", "")
        forbidden = set(row.get("forbidden_behavior_tags", []))
        for name, pattern in PROHIBITED_PATTERNS.items():
            if prohibited_pattern_hit(name, pattern, response):
                if name == "phone_number" and "phone_number_allowed" in forbidden:
                    continue
                hits_by_type[name] += 1
                errors.append(f"{rid}: prohibited {name}")
        first_line = response.strip().splitlines()[0] if response.strip() else ""
        if ESCALATION_ONLY.search(first_line):
            errors.append(f"{rid}: starts with escalation instead of concrete interim action")
        if REFUSAL_MARKERS.search(first_line) and row.get("renderer_style") not in {"live_fact_refusal", "visual_uncertainty"}:
            warnings.append(f"{rid}: first line is refusal-like outside refusal/uncertainty renderer")
    return errors, {"status": "fail" if errors else "pass", "hits_by_type": dict(hits_by_type), "warnings": warnings[:100], "errors": errors}


def prohibited_pattern_hit(name: str, pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        context = text[max(0, match.start() - 40) : match.end() + 20].lower()
        if name in {"definite_safety", "diagnosis_claim", "live_status_claim"} and re.search(
            r"\b(do not|don't|dont|cannot|can't|can not|never|avoid|must not|not)\b", context
        ):
            continue
        return True
    return False


def validate_split_leakage(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    reports: dict[str, Any] = {}
    for field in ["seed_id", "seed_family_id", "incident_archetype_id", "scenario_cluster_id"]:
        values: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            values[str(row.get(field, ""))].add(row.get("split", ""))
        leaked = {key: sorted(splits) for key, splits in values.items() if key and len(splits) > 1}
        reports[f"{field}_cross_split"] = leaked
        if leaked:
            errors.append(f"{field} crosses splits: {len(leaked)}")
    prompts_by_split: dict[str, dict[str, str]] = defaultdict(dict)
    answers_by_split: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        split = row.get("split", "")
        prompts_by_split[split][row.get("row_id", "")] = normalize_text(row.get("prompt", ""))
        answers_by_split[split][row.get("row_id", "")] = normalize_text(row.get("target_response", ""))
    exact_prompt_overlap = []
    for left_split in SPLITS:
        for right_split in SPLITS:
            if left_split >= right_split:
                continue
            left = {value: key for key, value in prompts_by_split[left_split].items()}
            right = {value: key for key, value in prompts_by_split[right_split].items()}
            for prompt in sorted(set(left) & set(right)):
                exact_prompt_overlap.append({"left": left[prompt], "right": right[prompt], "splits": [left_split, right_split]})
    if exact_prompt_overlap:
        errors.append(f"exact normalized prompt overlap across splits: {len(exact_prompt_overlap)}")
    high_similarity = []
    train_rows = [row for row in rows if row.get("split") == "train"]
    final_rows = [row for row in rows if row.get("split") == "final_eval"]
    for train_row in train_rows:
        for final_row in final_rows:
            prompt_score = token_jaccard(train_row.get("prompt", ""), final_row.get("prompt", ""))
            answer_score = token_jaccard(train_row.get("target_response", ""), final_row.get("target_response", ""))
            if prompt_score >= 0.74 or answer_score >= 0.82:
                high_similarity.append(
                    {
                        "train_row_id": train_row.get("row_id"),
                        "final_row_id": final_row.get("row_id"),
                        "prompt_jaccard": round(prompt_score, 3),
                        "answer_jaccard": round(answer_score, 3),
                    }
                )
                if len(high_similarity) >= 50:
                    break
        if len(high_similarity) >= 50:
            break
    if high_similarity:
        errors.append(f"high train/final prompt or answer similarity: {len(high_similarity)} sampled")
    reports.update(
        {
            "exact_prompt_overlap": exact_prompt_overlap[:50],
            "high_train_final_similarity_sample": high_similarity[:50],
            "status": "fail" if errors else "pass",
            "errors": errors,
        }
    )
    return errors, reports


def validate_output_similarity(rows: list[dict[str, Any]], out_csv: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    similarity_rows: list[dict[str, Any]] = []
    train = [row for row in rows if row.get("split") == "train"]
    dev_final = [row for row in rows if row.get("split") in {"dev", "final_eval"}]
    for left in train:
        for right in dev_final:
            prompt_score = token_jaccard(left.get("prompt", ""), right.get("prompt", ""))
            answer_score = token_jaccard(left.get("target_response", ""), right.get("target_response", ""))
            shape_match = bullet_shape(left.get("target_response", "")) == bullet_shape(right.get("target_response", ""))
            action_score = token_jaccard(action_sequence(left.get("target_response", "")), action_sequence(right.get("target_response", "")))
            if answer_score >= 0.78 or action_score >= 0.82 or (answer_score >= 0.68 and shape_match):
                similarity_rows.append(
                    {
                        "left_row_id": left.get("row_id", ""),
                        "right_row_id": right.get("row_id", ""),
                        "left_split": left.get("split", ""),
                        "right_split": right.get("split", ""),
                        "prompt_jaccard": round(prompt_score, 3),
                        "answer_jaccard": round(answer_score, 3),
                        "action_jaccard": round(action_score, 3),
                        "shape_match": shape_match,
                    }
                )
                if len(similarity_rows) >= 200:
                    break
        if len(similarity_rows) >= 200:
            break
    final_exact_answers = {}
    exact_answer_overlap = []
    for row in rows:
        key = normalize_text(row.get("target_response", ""))
        if not key:
            continue
        split = row.get("split", "")
        if split == "train":
            final_exact_answers.setdefault(key, row.get("row_id", ""))
        elif split == "final_eval" and key in final_exact_answers:
            exact_answer_overlap.append({"train_row_id": final_exact_answers[key], "final_row_id": row.get("row_id", "")})
    if similarity_rows:
        errors.append(f"train/dev-final output similarity above threshold: {len(similarity_rows)} sampled")
    if exact_answer_overlap:
        errors.append(f"exact train/final answer overlap: {len(exact_answer_overlap)}")
    if out_csv is not None:
        write_csv_rows(out_csv, similarity_rows)
    return errors, {
        "status": "fail" if errors else "pass",
        "sampled_similarity_pairs": len(similarity_rows),
        "exact_train_final_answer_overlap": exact_answer_overlap[:50],
        "errors": errors,
    }


def validate_per_seed_diversity(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_seed[row.get("seed_id", "")].append(row)
    failing_seed_examples: list[dict[str, Any]] = []
    for seed_id, seed_rows in rows_by_seed.items():
        if len(seed_rows) <= 1:
            continue
        prompt_keys = {normalize_text(row.get("prompt", "")) for row in seed_rows}
        answer_keys = {normalize_text(row.get("target_response", "")) for row in seed_rows}
        action_keys = {action_sequence(row.get("target_response", "")) for row in seed_rows}
        shape_keys = {bullet_shape(row.get("target_response", "")) for row in seed_rows}
        max_answer_similarity = 0.0
        for index, left in enumerate(seed_rows):
            for right in seed_rows[index + 1 :]:
                max_answer_similarity = max(
                    max_answer_similarity,
                    token_jaccard(left.get("target_response", ""), right.get("target_response", "")),
                )
        if len(seed_rows) >= 5 and (len(answer_keys) < 3 or len(action_keys) < 2):
            errors.append(f"{seed_id}: insufficient sibling answer/action diversity")
            failing_seed_examples.append(
                {
                    "seed_id": seed_id,
                    "row_count": len(seed_rows),
                    "unique_prompts": len(prompt_keys),
                    "unique_answers": len(answer_keys),
                    "unique_action_sequences": len(action_keys),
                    "unique_shapes": len(shape_keys),
                    "max_answer_similarity": round(max_answer_similarity, 3),
                    "example_row_ids": [row.get("row_id", "") for row in seed_rows[:5]],
                }
            )
        elif max_answer_similarity >= 0.92:
            warnings.append(f"{seed_id}: high sibling answer similarity {max_answer_similarity:.3f}")
    report = {
        "status": "fail" if errors else "pass",
        "seed_count": len(rows_by_seed),
        "failing_seed_count": len(failing_seed_examples),
        "failing_seed_examples": failing_seed_examples[:50],
        "warnings": warnings[:100],
        "errors": errors[:100],
    }
    return errors, report


def validate_final_eval_isolation(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    strict = manifest.get("config", {}).get("final_eval_isolation") == "strict"
    errors: list[str] = []
    violations = []
    if strict:
        train_or_dev_ids = {row.get("row_id", "") for row in rows if row.get("split") in {"train", "dev"}}
        for row in rows:
            if row.get("split") != "final_eval":
                continue
            refs = row.get("generation_source_refs", [])
            if isinstance(refs, str):
                refs = [refs]
            leaked_refs = sorted(set(refs) & train_or_dev_ids)
            parent = row.get("parent_row_id", "")
            if leaked_refs or parent in train_or_dev_ids:
                violations.append(
                    {
                        "row_id": row.get("row_id", ""),
                        "leaked_generation_source_refs": leaked_refs,
                        "parent_row_id": parent if parent in train_or_dev_ids else "",
                    }
                )
        if violations:
            errors.append(f"final_eval isolation violations: {len(violations)}")
    return errors, {
        "status": "fail" if errors else "pass",
        "strict_isolation": strict,
        "violation_count": len(violations),
        "violations": violations[:50],
        "errors": errors,
    }


def validate_pattern_collapse(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any], list[dict[str, Any]], str]:
    errors: list[str] = []
    accepted_count = max(1, len(rows))
    buckets = {
        "first_sentence": Counter(),
        "opening_12_tokens": Counter(),
        "heading_sequence": Counter(),
        "shape_sequence": Counter(),
        "action_sequence": Counter(),
    }
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    numbered_first = 0
    refusal = 0
    for row in rows:
        rid = row.get("row_id", "")
        response = row.get("target_response", "")
        keys = {
            "first_sentence": normalize_first_sentence(response),
            "opening_12_tokens": first_tokens(response),
            "heading_sequence": heading_sequence(response),
            "shape_sequence": f"{sentence_count(response)}:{bullet_shape(response)}",
            "action_sequence": action_sequence(response),
        }
        for name, key in keys.items():
            if not key:
                continue
            buckets[name][key] += 1
            if len(examples[(name, key)]) < 5:
                examples[(name, key)].append(rid)
        if NUMBERED_LIST_FIRST.search(response):
            numbered_first += 1
        if REFUSAL_MARKERS.search(response):
            refusal += 1
    thresholds = {
        "first_sentence": 3,
        "opening_12_tokens": math.floor(accepted_count * 0.03),
        "heading_sequence": math.floor(accepted_count * 0.05),
        "shape_sequence": math.floor(accepted_count * 0.05),
        "action_sequence": math.floor(accepted_count * 0.07),
    }
    oversized: list[dict[str, Any]] = []
    for bucket_name, counter in buckets.items():
        threshold = max(1, thresholds[bucket_name])
        for key, count in counter.items():
            if key and count > threshold:
                oversized.append(
                    {
                        "cluster_type": bucket_name,
                        "cluster_key": key,
                        "count": count,
                        "threshold": threshold,
                        "example_row_ids": examples[(bucket_name, key)],
                    }
                )
    scoped_counters: dict[str, Counter[str]] = defaultdict(Counter)
    scoped_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        response = row.get("target_response", "")
        scope_keys = {
            "split": row.get("split", ""),
            "seed_family": row.get("seed_family_id", ""),
            "renderer": row.get("renderer_style", ""),
            "source_rules": "|".join(row.get("source_rule_ids", [])),
            "generator_batch": row.get("generation_run_id", ""),
        }
        shape_key = "|".join([normalize_first_sentence(response), bullet_shape(response), action_sequence(response)])
        for scope_name, scope_value in scope_keys.items():
            key = f"{scope_name}:{scope_value}:{shape_key}"
            scoped_counters[scope_name][key] += 1
            if len(scoped_examples[(scope_name, key)]) < 5:
                scoped_examples[(scope_name, key)].append(row.get("row_id", ""))
    scoped_oversized = []
    for scope_name, counter in scoped_counters.items():
        for key, count in counter.items():
            threshold = 5 if scope_name in {"split", "renderer", "source_rules"} else 3
            if count > threshold:
                scoped_oversized.append(
                    {
                        "cluster_type": f"scoped_{scope_name}",
                        "cluster_key": key,
                        "count": count,
                        "threshold": threshold,
                        "example_row_ids": scoped_examples[(scope_name, key)],
                    }
                )
    oversized.extend(scoped_oversized)
    if oversized:
        errors.append(f"pattern clusters above threshold: {len(oversized)}")
    numbered_share = numbered_first / accepted_count
    refusal_share = refusal / accepted_count
    if numbered_share > 0.25:
        errors.append(f"numbered-list-first share {numbered_share:.1%} exceeds 25%")
    if refusal_share > 0.10:
        errors.append(f"refusal/cannot share {refusal_share:.1%} exceeds 10%")
    cluster_markdown = ["# Pattern Cluster Examples", ""]
    for cluster in oversized[:20]:
        cluster_markdown.append(
            f"- {cluster['cluster_type']} count={cluster['count']} threshold={cluster['threshold']} examples={', '.join(cluster['example_row_ids'])}"
        )
    report = {
        "status": "fail" if errors else "pass",
        "row_count": len(rows),
        "numbered_list_first_share": numbered_share,
        "refusal_share": refusal_share,
        "oversized_cluster_count": len(oversized),
        "largest_clusters": sorted(oversized, key=lambda item: item["count"], reverse=True)[:25],
        "scoped_oversized_cluster_count": len(scoped_oversized),
        "errors": errors,
    }
    return errors, report, oversized, "\n".join(cluster_markdown) + "\n"


def validate_quotas(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    by_split = Counter(row.get("split") for row in rows)
    by_renderer = Counter(row.get("renderer_style") for row in rows)
    by_domain = Counter(row.get("hazard_domain") for row in rows)
    by_domain_renderer = Counter((row.get("hazard_domain"), row.get("renderer_style")) for row in rows)
    for domain, count in by_domain.items():
        top_count = max((by_domain_renderer[(domain, style)] for style in RENDERER_STYLES), default=0)
        if count and top_count / count > 0.45:
            errors.append(f"{domain}: one renderer dominates {top_count}/{count} rows")
    high_risk_rows = [row for row in rows if row.get("risk_level") == "high"]
    high_styles = Counter(row.get("renderer_style") for row in high_risk_rows)
    for style in ["urgent_stop_refusal", "visual_uncertainty", "live_fact_refusal", "short_offline_card"]:
        if high_risk_rows and high_styles[style] == 0:
            warnings.append(f"high-risk rows have no {style} examples")
    per_seed = Counter(row.get("seed_id") for row in rows if row.get("split") == "train")
    over_seed_cap = {seed_id: count for seed_id, count in per_seed.items() if count > 5}
    if over_seed_cap:
        errors.append(f"train seeds over max 5 variants: {len(over_seed_cap)}")
    report = {
        "status": "fail" if errors else "pass",
        "by_split": dict(by_split),
        "by_renderer": dict(by_renderer),
        "by_domain": dict(by_domain),
        "high_risk_renderer_counts": dict(high_styles),
        "train_rows_per_seed_max": max(per_seed.values(), default=0),
        "train_seeds_over_cap": over_seed_cap,
        "warnings": warnings,
        "errors": errors,
    }
    return errors, report


def behavior_distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(row.get("split", "") for row in rows)
    by_difficulty = Counter()
    by_behavior = Counter()
    by_forbidden = Counter()
    by_rule = Counter()
    by_split_renderer = Counter()
    for row in rows:
        tags = row.get("target_behavior_tags", [])
        forbidden = row.get("forbidden_behavior_tags", [])
        if len(tags) > 1:
            by_difficulty[tags[1]] += 1
        for tag in tags:
            by_behavior[tag] += 1
        for tag in forbidden:
            by_forbidden[tag] += 1
        for rule_id in row.get("source_rule_ids", []):
            by_rule[rule_id] += 1
        by_split_renderer[(row.get("split", ""), row.get("renderer_style", ""))] += 1
    warnings = []
    for split in SPLITS:
        split_count = by_split[split]
        if not split_count:
            continue
        refusal_like = sum(1 for row in rows if row.get("split") == split and row.get("renderer_style") in {"urgent_stop_refusal", "live_fact_refusal", "visual_uncertainty"})
        if refusal_like / split_count > 0.5:
            warnings.append(f"{split}: refusal/uncertainty renderers exceed 50%")
    return {
        "status": "pass",
        "by_split": dict(by_split),
        "by_difficulty": dict(by_difficulty),
        "by_behavior_tag": dict(by_behavior),
        "by_forbidden_tag": dict(by_forbidden),
        "top_source_rules": dict(by_rule.most_common(50)),
        "by_split_renderer": {f"{split}|{renderer}": count for (split, renderer), count in by_split_renderer.items()},
        "warnings": warnings,
    }


def build_review_sampling_manifest(
    rows: list[dict[str, Any]],
    pattern_report: dict[str, Any],
    similarity_report: dict[str, Any],
    safety_report: dict[str, Any],
) -> dict[str, Any]:
    high_risk = [row.get("row_id", "") for row in rows if row.get("risk_level") == "high"]
    final_eval = [row.get("row_id", "") for row in rows if row.get("split") == "final_eval"]
    cluster_examples = []
    for cluster in pattern_report.get("largest_clusters", [])[:25]:
        cluster_examples.extend(cluster.get("example_row_ids", []))
    sample = []
    seen = set()
    for row_id in [*final_eval, *high_risk[:80], *cluster_examples]:
        if row_id and row_id not in seen:
            sample.append(row_id)
            seen.add(row_id)
    return {
        "status": "pass",
        "sampling_policy": "all final_eval rows, high-risk rows, and examples from largest pattern clusters",
        "required_reviewer_roles": [
            "safety_source_grounding",
            "leakage_eval_contamination",
            "diversity_pattern_collapse",
            "adversarial_skeptic",
        ],
        "sample_row_count": len(sample),
        "sample_row_ids": sample[:250],
        "final_eval_rows_included": len(final_eval),
        "high_risk_rows_included": min(80, len(high_risk)),
        "source_support_warnings": 0,
        "safety_warnings": len(safety_report.get("warnings", [])),
        "output_similarity_status": similarity_report.get("status"),
    }


def critic_stub(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviews = []
    for row in rows:
        response = row.get("target_response", "")
        specificity = 3 if any(token in normalize_text(response) for token in normalize_text(row.get("prompt", "")).split()[:12]) else 2
        novelty = 2
        verdict = "keep"
        issues: list[str] = []
        if len(response.split()) > 180:
            verdict = "rewrite"
            issues.append("too_long")
        if len(response.split()) < 18:
            verdict = "rewrite"
            issues.append("too_short")
        reviews.append(
            {
                "row_id": row.get("row_id"),
                "reviewer_role": "deterministic_critic_stub",
                "reviewer_version": "critic_stub_v1",
                "prompt_version": "critic_stub_v1",
                "model": "none_deterministic",
                "temperature": 0,
                "verdict": verdict,
                "severity": "medium" if issues else "none",
                "issue_tags": issues,
                "rationale": "length-only deterministic placeholder; not a safety or source-grounding approval",
                "blocking": verdict in {"reject", "escalate"},
                "scores": {
                    "safety": 3,
                    "target_fidelity": 2,
                    "forbidden_avoidance": 3,
                    "specificity": specificity,
                    "compactness": 3,
                    "cultural_fit": 2,
                    "pattern_novelty": novelty,
                },
                "issues": issues,
                "evidence": "deterministic placeholder critic; replace with archived LLM critic outputs before final approval",
                "confidence": 0.5,
            }
        )
    return reviews


def subagent_review_stub(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roles = ["safety_harm", "task_fidelity_source", "diversity_pattern", "adversarial_skeptic"]
    sample = select_subagent_sample(rows)
    reviews = []
    for role in roles:
        for row in sample:
            reviews.append(
                {
                    "row_id": row.get("row_id"),
                    "reviewer_role": role,
                    "verdict": "keep",
                    "reviewer_version": "subagent_stub_v1",
                    "severity": "none",
                    "issue_tags": [],
                    "rationale": "placeholder structured review; not approval",
                    "blocking": False,
                    "scores": {
                        "safety": 3,
                        "target_fidelity": 2,
                        "forbidden_avoidance": 3,
                        "specificity": 2,
                        "pattern_novelty": 2,
                    },
                    "issues": [],
                    "evidence": "placeholder structured subagent review record; run live independent subagents before final approval",
                    "confidence": 0.5,
                    "calibrated": False,
                }
            )
    return reviews


def select_subagent_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    high = [row for row in rows if row.get("risk_level") == "high"][:40]
    final = [row for row in rows if row.get("split") == "final_eval"][:50]
    train = [row for row in rows if row.get("split") == "train"]
    stride = max(1, len(train) // 50)
    stratified = train[::stride][:50]
    selected = {row["row_id"]: row for row in [*high, *final, *stratified]}
    return list(selected.values())[:180]


def validate_reviews(critic_rows: list[dict[str, Any]], subagent_rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    reviewer_required = {"reviewer_role", "reviewer_version", "verdict", "severity", "issue_tags", "rationale", "row_id", "blocking"}
    critic_rejects = [row for row in critic_rows if row.get("verdict") in {"reject", "escalate"} or row.get("blocking") is True]
    subagent_rejects = [row for row in subagent_rows if row.get("verdict") in {"reject", "escalate"} or row.get("blocking") is True]
    if critic_rejects:
        errors.append(f"critic rejects/escalations unresolved: {len(critic_rejects)}")
    if subagent_rejects:
        errors.append(f"subagent rejects/escalations unresolved: {len(subagent_rejects)}")
    placeholder_critic = [row for row in critic_rows if row.get("reviewer_role") == "deterministic_critic_stub"]
    placeholder_subagent = [row for row in subagent_rows if row.get("calibrated") is False or "placeholder" in str(row.get("evidence", "")).lower()]
    if placeholder_critic:
        errors.append("critic review records are placeholders; replace with calibrated critic outputs")
    calibrated = [row for row in subagent_rows if row.get("calibrated") is True]
    if subagent_rows and (not calibrated or placeholder_subagent):
        errors.append("subagent review records are not calibrated; run 30-row canary calibration before approval")
    missing_fields = []
    for row in [*critic_rows, *subagent_rows]:
        missing = sorted(reviewer_required - set(row))
        if missing:
            missing_fields.append({"row_id": row.get("row_id", ""), "reviewer_role": row.get("reviewer_role", ""), "missing": missing})
    if missing_fields:
        errors.append(f"review records missing required fields: {len(missing_fields)}")
    roles = {row.get("reviewer_role") for row in subagent_rows}
    required_roles = {"safety_source_grounding", "leakage_eval_contamination", "diversity_pattern_collapse", "adversarial_skeptic"}
    legacy_roles = {"safety_harm", "task_fidelity_source", "diversity_pattern"}
    normalized_roles = set(roles)
    if legacy_roles & normalized_roles:
        normalized_roles |= {
            "safety_source_grounding" if "safety_harm" in roles else "",
            "leakage_eval_contamination" if "task_fidelity_source" in roles else "",
            "diversity_pattern_collapse" if "diversity_pattern" in roles else "",
        }
        normalized_roles.discard("")
    missing_roles = required_roles - normalized_roles
    if missing_roles:
        errors.append(f"missing subagent reviewer roles: {sorted(missing_roles)}")
    return errors, {
        "status": "fail" if errors else "pass",
        "critic_rows": len(critic_rows),
        "subagent_review_rows": len(subagent_rows),
        "critic_rejects": len(critic_rejects),
        "subagent_rejects": len(subagent_rejects),
        "reviewer_roles": sorted(roles),
        "calibrated_review_rows": len(calibrated),
        "placeholder_critic_rows": len(placeholder_critic),
        "placeholder_subagent_rows": len(placeholder_subagent),
        "missing_required_field_examples": missing_fields[:50],
        "errors": errors,
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_expansion(
    run_dir: Path,
    rule_manifest: Path,
    *,
    profile: str = "calibration",
    fail_on_count: bool = True,
    accepted_count_ranges: dict[str, tuple[int, int]] | None = None,
) -> GateResult:
    rows = read_jsonl(run_dir / "generated_rows.jsonl")
    errors: list[str] = []
    warnings: list[str] = []
    reports: dict[str, Any] = {}
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    schema_errors, schema_report = validate_schema(rows)
    errors.extend(schema_errors)
    reports["schema_validation_report"] = schema_report
    lineage_errors, lineage_report = validate_lineage(rows, manifest)
    errors.extend(lineage_errors)
    reports["lineage_validation_report"] = lineage_report
    source_errors, source_report = validate_source_grounding(rows, rule_manifest, run_dir / "source_grounding_report.csv")
    errors.extend(source_errors)
    reports["source_grounding_report"] = source_report
    source_support_errors, source_support_report = validate_source_claim_support(
        rows,
        rule_manifest,
        run_dir / "source_claim_support_report.csv",
    )
    errors.extend(source_support_errors)
    warnings.extend(source_support_report.get("warnings", []))
    reports["source_claim_support_report"] = source_support_report
    safety_errors, safety_report = validate_safety(rows)
    errors.extend(safety_errors)
    warnings.extend(safety_report.get("warnings", []))
    reports["safety_lint_report"] = safety_report
    leakage_errors, leakage_report = validate_split_leakage(rows)
    errors.extend(leakage_errors)
    reports["split_leakage_report"] = leakage_report
    output_similarity_errors, output_similarity_report = validate_output_similarity(rows, run_dir / "output_similarity_report.csv")
    errors.extend(output_similarity_errors)
    reports["output_similarity_report"] = output_similarity_report
    per_seed_errors, per_seed_report = validate_per_seed_diversity(rows)
    errors.extend(per_seed_errors)
    warnings.extend(per_seed_report.get("warnings", []))
    reports["per_seed_diversity_report"] = per_seed_report
    final_eval_errors, final_eval_report = validate_final_eval_isolation(rows, manifest)
    errors.extend(final_eval_errors)
    reports["final_eval_isolation_report"] = final_eval_report
    pattern_errors, pattern_report, oversized, cluster_md = validate_pattern_collapse(rows)
    errors.extend(pattern_errors)
    reports["pattern_collapse_report"] = pattern_report
    write_csv_rows(run_dir / "oversized_clusters.csv", oversized)
    (run_dir / "cluster_examples.md").write_text(cluster_md, encoding="utf-8")
    quota_errors, quota_report = validate_quotas(rows)
    errors.extend(quota_errors)
    warnings.extend(quota_report.get("warnings", []))
    reports["quota_report"] = quota_report
    behavior_report = behavior_distribution_report(rows)
    warnings.extend(behavior_report.get("warnings", []))
    reports["behavior_distribution_report"] = behavior_report
    deterministic_errors = list(errors)
    deterministic_warnings = list(warnings)
    deterministic_report = {
        "status": "fail" if deterministic_errors else "pass",
        "errors": deterministic_errors,
        "warnings": deterministic_warnings,
        "gate_reports": {
            key: report.get("status", "n/a")
            for key, report in reports.items()
            if key != "review_report"
        },
    }
    write_json(run_dir / "deterministic_gate_report.json", deterministic_report)
    review_sampling = build_review_sampling_manifest(rows, pattern_report, output_similarity_report, safety_report)
    reports["review_sampling_manifest"] = review_sampling
    write_json(run_dir / "review_sampling_manifest.json", review_sampling)
    critic_path = run_dir / "critic_report.jsonl"
    subagent_path = run_dir / "subagent_review_report.jsonl"
    if not critic_path.exists():
        write_jsonl(critic_path, critic_stub(rows))
    if not subagent_path.exists():
        write_jsonl(subagent_path, subagent_review_stub(rows))
    critic_rows = read_jsonl(critic_path)
    subagent_rows = read_jsonl(subagent_path)
    review_errors, review_report = validate_reviews(critic_rows, subagent_rows)
    errors.extend(review_errors)
    reports["review_report"] = review_report
    accepted = [dict(row, quality_status="accepted", review_state="frozen") for row in rows if not errors]
    rejected = [dict(row, quality_status="rejected", review_state="rejected") for row in rows if errors]
    if accepted_count_ranges is None:
        profile_config = EXPANSION_PROFILES.get(profile)
        if profile_config is None:
            raise ValueError(f"unknown expansion profile: {profile}")
        configured_ranges = profile_config["accepted_count_ranges"]
        accepted_count_ranges = {
            split: (bounds[0], bounds[1]) for split, bounds in configured_ranges.items()
        } if configured_ranges else {}
    if fail_on_count and not errors:
        counts = Counter(row["split"] for row in accepted)
        for split, (minimum, maximum) in accepted_count_ranges.items():
            if counts[split] < minimum or counts[split] > maximum:
                errors.append(f"{split} accepted count {counts[split]} not in required range {minimum}-{maximum}")
    write_jsonl(run_dir / "accepted_rows.jsonl", accepted if not errors else [])
    write_jsonl(run_dir / "final_accepted_rows.jsonl", accepted if not errors else [])
    write_jsonl(run_dir / "rejected_rows.jsonl", rejected if errors else [])
    rejected_ledger = [
        {
            "row_id": row.get("row_id", ""),
            "candidate_id": row.get("candidate_id", ""),
            "split": row.get("split", ""),
            "seed_id": row.get("seed_id", ""),
            "rejection_reasons": errors[:100],
        }
        for row in rows
    ] if errors else []
    write_jsonl(run_dir / "rejected_row_ledger.jsonl", rejected_ledger)
    reviewer_decisions = [
        {
            "row_id": row.get("row_id", ""),
            "reviewer_role": row.get("reviewer_role", ""),
            "reviewer_version": row.get("reviewer_version", ""),
            "verdict": row.get("verdict", ""),
            "severity": row.get("severity", ""),
            "issue_tags": row.get("issue_tags", []),
            "blocking": row.get("blocking", False),
            "rationale": row.get("rationale", row.get("evidence", "")),
        }
        for row in [*critic_rows, *subagent_rows]
    ]
    write_jsonl(run_dir / "reviewer_decisions.jsonl", reviewer_decisions)
    for key, report in reports.items():
        filename = key if key.endswith(".json") else f"{key}.json"
        if key in {"source_grounding_report", "source_claim_support_report", "output_similarity_report"}:
            continue
        write_json(run_dir / filename, report)
    manifest.update(
        {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "validation_profile": profile,
            "accepted_count_ranges": accepted_count_ranges,
            "validation_status": "fail" if errors else "pass",
            "validation_error_count": len(errors),
            "validation_warning_count": len(warnings),
            "artifact_hashes": {
                name: sha256_file(run_dir / name)
                for name in [
                    "generated_rows.jsonl",
                    "accepted_rows.jsonl",
                    "rejected_rows.jsonl",
                    "critic_report.jsonl",
                    "subagent_review_report.jsonl",
                    "repair_lineage.jsonl",
                ]
                if (run_dir / name).exists()
            },
        }
    )
    write_json(manifest_path, manifest)
    freeze_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "generation_run_id": manifest.get("generation_run_id", ""),
        "accepted_counts": dict(Counter(row["split"] for row in accepted)) if not errors else {},
        "seed_snapshot_hash": manifest.get("seed_snapshot_hash", ""),
        "source_rule_snapshot_hash": manifest.get("source_rule_snapshot_hash", ""),
        "generator_config_hash": manifest.get("generator_config_hash", ""),
        "artifact_hashes": {
            name: sha256_file(run_dir / name)
            for name in [
                "generated_rows.jsonl",
                "final_accepted_rows.jsonl",
                "rejected_row_ledger.jsonl",
                "critic_report.jsonl",
                "subagent_review_report.jsonl",
                "reviewer_decisions.jsonl",
                "deterministic_gate_report.json",
            ]
            if (run_dir / name).exists()
        },
        "errors": errors[:100],
    }
    write_json(run_dir / "dataset_freeze_manifest.json", freeze_manifest)
    summary = make_run_summary(manifest, reports, errors, warnings)
    (run_dir / "run_summary.md").write_text(summary, encoding="utf-8")
    return GateResult("fail" if errors else "pass", errors, warnings, reports)


def make_run_summary(manifest: dict[str, Any], reports: dict[str, Any], errors: list[str], warnings: list[str]) -> str:
    lines = [
        "# Sankat Saathi Expansion Gate Summary",
        "",
        f"Status: **{'FAIL' if errors else 'PASS'}**",
        f"Rows generated: {manifest.get('row_count', 'unknown')}",
        f"Counts: `{manifest.get('counts', {})}`",
        "",
        "## Gate Status",
    ]
    for key, report in reports.items():
        lines.append(f"- {key}: {report.get('status', 'n/a')}")
    if errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in errors[:100])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings[:100])
    lines.extend(
        [
            "",
            "## Training Rule",
            "No training starts until this summary is PASS, subagent reviews are calibrated, and accepted row counts match the required train/dev/final targets.",
            "",
        ]
    )
    return "\n".join(lines)


def make_audit_bundle(run_dir: Path) -> dict[str, Any]:
    artifact_names = [
        "dataset_manifest.json",
        "schema_validation_report.json",
        "lineage_validation_report.json",
        "split_leakage_report.json",
        "source_grounding_report.csv",
        "source_claim_support_report.csv",
        "safety_lint_report.json",
        "output_similarity_report.csv",
        "pattern_collapse_report.json",
        "per_seed_diversity_report.json",
        "quota_report.json",
        "behavior_distribution_report.json",
        "deterministic_gate_report.json",
        "review_sampling_manifest.json",
        "critic_report.jsonl",
        "subagent_review_report.jsonl",
        "reviewer_decisions.jsonl",
        "repair_lineage.jsonl",
        "accepted_rows.jsonl",
        "final_accepted_rows.jsonl",
        "rejected_rows.jsonl",
        "rejected_row_ledger.jsonl",
        "dataset_freeze_manifest.json",
        "run_summary.md",
    ]
    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "artifacts": {
            name: {
                "exists": (run_dir / name).exists(),
                "sha256": sha256_file(run_dir / name),
                "bytes": (run_dir / name).stat().st_size if (run_dir / name).exists() else 0,
            }
            for name in artifact_names
        },
    }
    write_json(run_dir / "audit_bundle_manifest.json", bundle)
    return bundle
