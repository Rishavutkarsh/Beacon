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
    "created_by",
    "prompt_config_hash",
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


def make_row(seed: dict[str, Any], variant_index: int, created_by: str, prompt_config_hash: str) -> dict[str, Any]:
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
    content = {"prompt": prompt, "target_response": target_response}
    return {
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
        "created_by": created_by,
        "prompt_config_hash": prompt_config_hash,
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
) -> dict[str, Any]:
    seeds = load_seeds(seed_path)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        by_split[seed.get("split", "")].append(seed)
    profile_config = EXPANSION_PROFILES.get(profile)
    if profile_config is None:
        raise ValueError(f"unknown expansion profile: {profile}")
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
    }
    prompt_config_hash = stable_hash(config)
    rows: list[dict[str, Any]] = []
    feasibility_errors: list[str] = []
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
                split_rows.append(make_row(seed, used_for_seed, created_by, prompt_config_hash))
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
        "cluster_examples.md",
        "critic_report.jsonl",
        "oversized_clusters.csv",
        "pattern_collapse_report.json",
        "quota_report.json",
        "rejected_rows.jsonl",
        "review_report.json",
        "run_summary.md",
        "safety_lint_report.json",
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
                "prompt_version": "critic_stub_v1",
                "model": "none_deterministic",
                "temperature": 0,
                "verdict": verdict,
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
    critic_rejects = [row for row in critic_rows if row.get("verdict") in {"reject", "escalate"}]
    subagent_rejects = [row for row in subagent_rows if row.get("verdict") in {"reject", "escalate"}]
    if critic_rejects:
        errors.append(f"critic rejects/escalations unresolved: {len(critic_rejects)}")
    if subagent_rejects:
        errors.append(f"subagent rejects/escalations unresolved: {len(subagent_rejects)}")
    calibrated = [row for row in subagent_rows if row.get("calibrated") is True]
    if subagent_rows and not calibrated:
        errors.append("subagent review records are not calibrated; run 30-row canary calibration before approval")
    roles = {row.get("reviewer_role") for row in subagent_rows}
    missing_roles = {"safety_harm", "task_fidelity_source", "diversity_pattern", "adversarial_skeptic"} - roles
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
        "errors": errors,
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
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
    schema_errors, schema_report = validate_schema(rows)
    errors.extend(schema_errors)
    reports["schema_validation_report"] = schema_report
    source_errors, source_report = validate_source_grounding(rows, rule_manifest, run_dir / "source_grounding_report.csv")
    errors.extend(source_errors)
    reports["source_grounding_report"] = source_report
    safety_errors, safety_report = validate_safety(rows)
    errors.extend(safety_errors)
    warnings.extend(safety_report.get("warnings", []))
    reports["safety_lint_report"] = safety_report
    leakage_errors, leakage_report = validate_split_leakage(rows)
    errors.extend(leakage_errors)
    reports["split_leakage_report"] = leakage_report
    pattern_errors, pattern_report, oversized, cluster_md = validate_pattern_collapse(rows)
    errors.extend(pattern_errors)
    reports["pattern_collapse_report"] = pattern_report
    write_csv_rows(run_dir / "oversized_clusters.csv", oversized)
    (run_dir / "cluster_examples.md").write_text(cluster_md, encoding="utf-8")
    quota_errors, quota_report = validate_quotas(rows)
    errors.extend(quota_errors)
    warnings.extend(quota_report.get("warnings", []))
    reports["quota_report"] = quota_report
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
    accepted = [dict(row, quality_status="accepted") for row in rows if not errors]
    rejected = [dict(row, quality_status="rejected") for row in rows if errors]
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
    write_jsonl(run_dir / "rejected_rows.jsonl", rejected if errors else [])
    for key, report in reports.items():
        filename = key if key.endswith(".json") else f"{key}.json"
        if key == "source_grounding_report":
            continue
        write_json(run_dir / filename, report)
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
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
        "split_leakage_report.json",
        "source_grounding_report.csv",
        "safety_lint_report.json",
        "pattern_collapse_report.json",
        "quota_report.json",
        "critic_report.jsonl",
        "subagent_review_report.jsonl",
        "repair_lineage.jsonl",
        "accepted_rows.jsonl",
        "rejected_rows.jsonl",
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
