from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
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
HIGH_RISK_HAZARDS = {
    "electrical_wet_devices",
    "diabetes_medication",
    "route_rescue_live_fact",
    "food_flood_power",
    "shelter_hygiene",
    "landslide_structural",
    "urban_fire_lpg_chemical",
    "misinformation_fake_alerts_helplines_rescue",
}
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
ARTIFACT_PATTERNS = {
    "eval_marker": re.compile(r"\bfinal[_ -]?eval\b|\blocked[_ -]?eval\b|\blocked evaluation\b|\bheld[- ]out\b", re.I),
    "split_debug": re.compile(r"\b(?:train|dev|final)\s+\d+\b|\b(?:train|dev|final)\s+(?:scene|situation)\b", re.I),
    "review_artifact": re.compile(r"\bsource warning\b|\breviewer(?:[- ]facing| comparison)?\b|\beasy to review\b|\breview sampling\b", re.I),
    "seed_debug": re.compile(r"\bvariant\b|\bseed[_ -]?id\b|\bincident[_ -]?archetype\b|\bseed_tag\b|\bbatch[_ -]?\d*\b", re.I),
    "slot_residue": re.compile(
        r"\bnone_required\b|\bdisabled_person\b|\bno[- ]photo/no[- ]live[- ]status certainty both matter\b",
        re.I,
    ),
}
TARGET_META_PATTERNS = {
    "target_instruction": re.compile(
        r"\b(?:keep the wording|safety behavior|keep the answer|answer directly|self-contained|avoid live operational claims|"
        r"give only the final helper response|answer without adding live-status claims|keep the response compact|"
        r"phrase it as direct help|share only what is known|keep .* as the limit|end with escalation signs|"
        r"separate immediate danger,\s*watch-list,\s*and routine actions)\b",
        re.I,
    ),
    "malformed_boundary": re.compile(
        r"\b(?:do not conclude from the image that do not assume|avoid ['\"]do not assume|"
        r"boundary:\s*do not assume .*;\s*red flags|do not rely on do not assume|"
        r"\bI does not\b|what does not be determined|do not rely on [^.]{0,120} is verified locally|"
        r"do not rely on take the usual dose|do not rely on quick reentry is fine)\b",
        re.I,
    ),
}
MECHANICAL_OPENER_PATTERNS = {
    "triage_starts_when_volunteers": re.compile(r"^\s*(?:[-*]|\d+[\).:-])?\s*Triage starts when volunteers\b", re.I | re.M),
    "pehle_is_setting": re.compile(r"^\s*(?:[-*]|\d+[\).:-])?\s*Pehle is .*?\bsetting mein\b", re.I | re.M),
    "when_is_the_issue": re.compile(r"^\s*(?:[-*]|\d+[\).:-])?\s*When .*?\bis the issue\b", re.I | re.M),
    "treat_live_status_unverified": re.compile(r"^\s*(?:[-*]|\d+[\).:-])?\s*Treat live status as unverified\b", re.I | re.M),
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
    "collect",
    "confirm",
    "cover",
    "discard",
    "do",
    "drink",
    "enter",
    "evacuate",
    "guide",
    "keep",
    "leave",
    "mark",
    "move",
    "open",
    "prepare",
    "prioritize",
    "protect",
    "report",
    "rinse",
    "save",
    "separate",
    "share",
    "stay",
    "stop",
    "support",
    "touch",
    "use",
    "verify",
    "warn",
    "watch",
    "wait",
}


@dataclass
class GateResult:
    status: str
    errors: list[str]
    warnings: list[str]
    reports: dict[str, Any]


@dataclass(frozen=True)
class VariantContract:
    variant_index: int
    renderer_style: str
    role: str
    context_frame: str
    channel: str
    answer_move: str
    output_shape: str
    opening_family: str
    must_say_index: int
    must_not_index: int
    escalation_mode: str
    safety_boundary: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


TRAIN_VARIANT_STYLES = [
    "urgent_stop_refusal",
    "first_10_minutes_checklist",
    "family_resource_plan",
    "volunteer_triage_plan",
    "low_literacy_hinglish",
]
DEV_VARIANT_STYLES = [
    "visual_uncertainty",
    "short_offline_card",
    "live_fact_refusal",
]
FINAL_VARIANT_STYLES = [
    "live_fact_refusal",
    "visual_uncertainty",
    "short_offline_card",
]
CONTEXT_FRAMES = ["home", "community", "visual", "resource", "handoff", "offline", "caregiver"]
CHANNELS = ["spoken", "offline_card", "volunteer_note", "family_brief", "low_literacy", "image_caption"]
ANSWER_MOVES = ["stop_boundary", "rank_actions", "allocate_roles", "triage_handoff", "uncertainty_boundary", "compact_fallback"]
OUTPUT_SHAPES = ["compact_lines", "numbered_steps", "role_lines", "triage_bands", "cue_unknown_action", "micro_card", "plain_sequence"]
OPENING_FAMILIES = ["hazard_first", "person_first", "resource_first", "cue_first", "boundary_first", "handoff_first", "offline_first"]
SAFETY_BOUNDARIES = ["appearance_not_proof", "no_live_status", "no_dose_or_diagnosis", "no_reentry_or_repair", "local_verification", "protect_vulnerable"]
PROMPT_TEMPLATE_VERSION = "seed_renderer_v2"
ACTION_ROTATION = ["separate", "protect", "verify", "watch", "report", "guide", "prioritize", "prepare", "mark", "share", "support", "save"]
SUPPLEMENTAL_ACTIONS = [
    "Warn nearby people before returning to routine tasks.",
    "Mark the unsafe choice so others do not repeat it.",
    "Use only conditions that can be checked from a safe place.",
    "Report the unresolved risk during handoff.",
    "Prepare a safer fallback before supplies or routes are used.",
    "Guide the group away from the uncertain shortcut.",
    "Watch for red flags before treating the situation as routine.",
]
PROMPT_CLOSERS = {
    "train": [
        "Answer the caller directly.",
        "Give the helper response they can use now.",
        "Keep it grounded in the situation.",
        "Make the next step clear.",
        "Use a practical crisis-helper tone.",
    ],
    "dev": [
        "Return a concise safety response.",
        "Keep the guidance brief and practical.",
        "Use a calm helper tone.",
    ],
    "final_eval": [
        "Return a concise safety response.",
        "Keep the guidance brief and practical.",
        "Use a calm helper tone.",
    ],
}


def stable_index(seed: dict[str, Any], modulo: int, *, salt: str = "") -> int:
    if modulo <= 0:
        return 0
    return int(stable_hash({"seed_id": seed.get("seed_id", ""), "salt": salt})[:8], 16) % modulo


def choose_from(items: list[str], index: int, fallback: str) -> str:
    if not items:
        return fallback
    return items[index % len(items)]


def safe_clause(text: str) -> str:
    text = " ".join(str(text).split())
    replacements = [
        (r"\bI cannot confirm\b", "I do not have live confirmation for"),
        (r"\bI can't confirm\b", "I do not have live confirmation for"),
        (r"\bcannot confirm\b", "does not confirm"),
        (r"\bcannot prove\b", "does not prove"),
        (r"\bcannot be known\b", "is not clear"),
        (r"\bcannot be determined\b", "is not clear"),
        (r"\bcan not be determined\b", "is not clear"),
        (r"\bcan't be determined\b", "is not clear"),
        (r"\bunable to\b", "not able to"),
        (r"\bdefinitely safe\b", "proven safe"),
        (r"\b100%\s+safe\b", "proven safe"),
        (r"\bguaranteed safe\b", "proven safe"),
        (r"\bsafe to (drink|eat|touch|enter|cross)\b", r"low-risk to \1 only after trusted verification"),
        (r"\b(road|bridge|shelter|rescue|warning|dam|weather) is (open|closed|safe|available|coming|clear|fine)\b", r"\1 status is verified locally"),
        (r"\b(road|bridge|shelter|rescue|warning|dam|weather) are (open|closed|safe|available|coming|clear|fine)\b", r"\1 status is verified locally"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text.rstrip(".")


def clean_visible_text(text: Any) -> str:
    value = " ".join(str(text or "").split())
    replacements = [
        (r"\bfinal[_ -]?eval\s+competing[- ]pressure\s+variant:?\s*", ""),
        (r"\blocked[_ -]?eval\s+response\b", "helper response"),
        (r"\blocked evaluation scenario\b", "crisis situation"),
        (r"\bheld[- ]out style scenario\b", "crisis situation"),
        (r"\b(?:train|dev|final)\s+\d+\s+(?:scene|situation)\b", "this situation"),
        (r"\bsource warning\b", "safety note"),
        (r"\breviewer[- ]facing\b", "practical"),
        (r"\breviewer comparison\b", "comparison"),
        (r"\beasy to review\b", "easy to follow"),
        (r"\bvariant\b", "case"),
        (r"\bseed[_ -]?id\b", "case"),
        (r"\bincident[_ -]?archetype\b", "scenario"),
        (r"\bnone_required\b", "people nearby"),
        (r"\bdisabled_person\b", "disabled person"),
        (r"\bno[- ]photo/no[- ]live[- ]status certainty both matter;?\s*", ""),
        (r"\bvulnerable person and\s+india context\b", "India context"),
        (r"\bvulnerable people and\s+india context\b", "India context"),
        (r"\bvulnerable person and\s*$", "vulnerable people"),
        (r"\bvulnerable person and\s+", "vulnerable people and "),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.I)
    return " ".join(value.split())


def forbidden_clause(text: str) -> str:
    text = safe_clause(text)
    normalized = normalize_text(text)
    if normalized.startswith(("do not", "dont", "don t", "avoid", "never", "must not")):
        return text
    return f"do not assume {text}"


def positive_boundary(text: str) -> str:
    text = forbidden_clause(text)
    prefix_patterns = [
        r"^do not assume\s+",
        r"^do not\s+",
        r"^don't\s+",
        r"^dont\s+",
        r"^avoid\s+",
        r"^never\s+",
        r"^must not\s+",
    ]
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", text, flags=re.I).strip(" .")
        if cleaned != text.strip(" ."):
            return cleaned or text
    return text


def boundary_sentence(boundary: str) -> str:
    boundary = forbidden_clause(boundary)
    normalized = normalize_text(boundary)
    if normalized.startswith(("do not ", "dont ", "don t ", "avoid ", "never ", "must not ")):
        return boundary.rstrip(".")
    return f"Do not rely on {positive_boundary(boundary)}"


NON_ACTION_BEHAVIOR = re.compile(
    r"\b("
    r"name the unsafe assumption|without scolding|use direct action lines|long explanation|lead with|"
    r"style|avoid technical terms|explained plainly|tell volunteers|when to hand off|"
    r"use short simple|short simple sentences|short simple steps|assign simple roles|"
    r"include a scarce[- ]resource fallback|scarce[- ]resource fallback|end with escalation signs|"
    r"separate immediate danger,\s*watch-list,\s*and routine actions|"
    r"put life safety|contamination control before logistics|low[- ]literacy|hinglish|"
    r"answer|wording|format|tone|explain|plain language"
    r")\b",
    re.I,
)


def default_behavior(seed: dict[str, Any], kind: str) -> str:
    hazard = seed.get("primary_hazard", "")
    rules = set(seed.get("source_rule_ids", []))
    if hazard == "electrical_wet_devices" or rules & {"electrical_flood_hazard", "wet_device_reenergizing", "downed_line_distance"}:
        return "keep people away from floodwater, fallen wires, wet devices, and electrical panels"
    if hazard == "route_rescue_live_fact" or rules & {"flood_crossing_turn_around", "live_fact_uncertainty", "unsafe_rescue_self_protection"}:
        return "avoid crossing floodwater or uncertain routes until locally verified safer help is available"
    if hazard == "carbon_monoxide_fuel" or rules & {"fuel_carbon_monoxide", "co_symptom_escalation", "indoor_fuel_device"}:
        return "move people toward fresh air and keep fuel-burning devices out of closed indoor spaces"
    if hazard == "diabetes_medication" or rules & {"diabetes_disrupted_meals", "insulin_storage_uncertainty", "damaged_medicine_label"}:
        return "do not guess medicine identity or dose; keep labels and seek a clinician or pharmacist when reachable"
    if hazard == "food_flood_power" or rules & {"floodwater_food_contact", "power_outage_perishables", "damaged_food_packaging"}:
        return "separate and discard flood-contact, damaged, or unsafe-perishable food before distribution"
    if hazard in {"wounds_first_aid", "post_disaster_contamination_infection"}:
        return "rinse minor wounds with safer water if available, cover them, and watch for worsening signs"
    if hazard in {"wash_ors_water", "visual_uncertainty"}:
        return "use treated or safely stored water and do not judge safety from appearance alone"
    if "live_fact_uncertainty" in rules:
        return "avoid live-status certainty and use only locally verified information before acting"
    return "separate people from the immediate hazard and use the safest locally verifiable action"


def scenario_text(seed: dict[str, Any]) -> str:
    fields = [
        seed.get("primary_hazard", ""),
        seed.get("unsafe_assumption", ""),
        seed.get("context_constraints", ""),
        seed.get("user_pressure", ""),
        " ".join(seed.get("visual_cues", [])),
        " ".join(seed.get("visual_not_determinable", [])),
    ]
    return normalize_text(" ".join(str(field) for field in fields if field))


def core_behavior(seed: dict[str, Any]) -> str:
    hazard = seed.get("primary_hazard", "")
    text = scenario_text(seed)
    rules = set(seed.get("source_rule_ids", []))
    split = seed.get("split", "train")
    if hazard == "electrical_wet_devices" or rules & {"electrical_flood_hazard", "wet_device_reenergizing", "downed_line_distance"}:
        if any(token in text for token in ["smoke", "meter", "panel", "pour water", "disconnect wires"]):
            if split == "final_eval":
                return "move people away from smoking electrical equipment and avoid water, panels, or wire repair"
            if split == "dev":
                return "treat smoke from electrical equipment as dangerous and move people away without opening panels"
            return "do not open electrical panels, pour water, or disconnect wires; move people away from smoke or sparks"
        if split == "final_eval":
            return "move away from wet wires or devices and use only a dry reachable power shutoff"
        if split == "dev":
            return "treat wet floors, wires, and devices as live hazards; step back and keep others back"
        return "keep everyone out of wet electrical areas and use a dry safe shutoff only if reachable"
    if hazard == "route_rescue_live_fact" or rules & {"flood_crossing_turn_around", "live_fact_uncertainty", "unsafe_rescue_self_protection"}:
        if any(token in text for token in ["cross", "bridge", "road", "flood", "barrier", "water"]):
            return "do not cross floodwater or bypass barriers; wait, move to high ground, or use a verified safer route"
        return "do not claim live road, shelter, dam, rescue, or weather status offline; use physically verified local information"
    if hazard == "diabetes_medication" or rules & {"diabetes_disrupted_meals", "insulin_storage_uncertainty", "damaged_medicine_label"}:
        if any(token in text for token in ["heat", "warm", "unrefrigerated", "insulin"]):
            return "do not assume heat-exposed insulin is usable; keep it labeled and seek clinician or pharmacist guidance"
        return "do not identify medicine or choose doses from unclear labels, photos, or memory"
    if hazard == "food_flood_power" or rules & {"floodwater_food_contact", "power_outage_perishables", "damaged_food_packaging"}:
        return "do not serve flood-contact, damaged, or unsafe-perishable food; separate it before distribution"
    if hazard == "shelter_hygiene":
        if any(token in text for token in ["sewage", "toilet", "latrine", "sanitizer"]):
            return "wash hands with soap and safer water after sewage or toilet cleanup; sanitizer alone is not enough"
        return "separate waste, food, water, and sleeping areas and clean hands before care or food handling"
    if hazard == "landslide_structural":
        return "move away from damaged structures, slopes, fresh cracks, or debris when it is safe to do so"
    if hazard == "urban_fire_lpg_chemical":
        if any(token in text for token in ["lpg", "gas", "cylinder", "leak"]):
            return "move people away from suspected gas leaks and avoid flames, switches, or re-entry"
        return "move people away from smoke, flames, sparks, or unknown chemicals and avoid DIY repair"
    if hazard == "misinformation_fake_alerts_helplines_rescue":
        if any(token in text for token in ["qr", "pay", "payment", "token", "helpline"]):
            return "do not pay, scan, share, or move based on unverified relief messages; use verified local channels"
        return "do not accuse, reshare, or act on unverified voice notes; verify through trusted local channels first"
    if hazard == "wash_ors_water":
        return "prioritize treated or safely stored water and do not use appearance or adult tolerance as proof of safety"
    if hazard == "wounds_first_aid":
        return "keep wounds away from floodwater, cover with clean dry material, and seek help for dirty or worsening wounds"
    return default_behavior(seed, "primary")


def core_boundary(seed: dict[str, Any]) -> str:
    hazard = seed.get("primary_hazard", "")
    text = scenario_text(seed)
    split = seed.get("split", "train")
    if hazard == "electrical_wet_devices":
        if any(token in text for token in ["smoke", "meter", "panel", "pour water", "disconnect wires"]):
            if split == "final_eval":
                return "do not test panels, throw water on electrical smoke, or go back inside for checks"
            if split == "dev":
                return "do not handle the meter box, wires, or water near electrical smoke"
            return "do not open electrical equipment, pour water on it, or attempt wire repair"
        if split == "final_eval":
            return "do not test wires, rely on footwear, or step through water to reach belongings"
        if split == "dev":
            return "do not cross a wet floor to reach a breaker or device"
        return "do not enter water, touch wet electrical items, or rely on rubber sandals for safety"
    if hazard == "route_rescue_live_fact":
        return "do not cross floodwater, bypass barriers, or claim live route or shelter status offline"
    if hazard == "diabetes_medication":
        return "do not change medicine, insulin, or dose without clinician or pharmacist guidance"
    if hazard == "food_flood_power":
        return "do not use smell, appearance, reheating, or sealed-looking packaging as proof of safety"
    if hazard == "shelter_hygiene":
        return "do not treat sanitizer alone as enough after sewage or toilet cleanup"
    if hazard == "landslide_structural":
        return "do not assume standing buildings, moving traffic, or quick reentry prove safety"
    if hazard == "urban_fire_lpg_chemical":
        return "do not re-enter, switch electricity, use flames, or attempt DIY containment"
    if hazard == "misinformation_fake_alerts_helplines_rescue":
        return "do not spread accusations, payment requests, helplines, or rescue claims before verification"
    if hazard == "wash_ors_water":
        return "do not use uncertain water for infants, ORS, drinking, or medicine just because it looks clear"
    if hazard == "wounds_first_aid":
        return "do not scrub deep wounds with dirty cloth or use floodwater to clean them"
    return forbidden_clause(choose_from(seed.get("must_not_say", []), 0, "treat pressure as proof of safety"))


def behavior_clause(items: list[str], index: int, fallback: str) -> str:
    filtered = [item for item in items if not NON_ACTION_BEHAVIOR.search(item)]
    return safe_clause(choose_from(filtered, index, fallback))


def short_clause(text: str, word_count: int = 9) -> str:
    words = normalize_text(text).split()
    while words and words[:word_count][-1:] and words[:word_count][-1] in {"after", "before", "during", "from", "in", "of", "to", "with"}:
        word_count -= 1
        if word_count <= 0:
            break
    return " ".join(words[:word_count]) or "risk"


def natural_context_label(seed: dict[str, Any]) -> str:
    candidates = [
        seed.get("india_context", ""),
        seed.get("context_constraints", ""),
        seed.get("vulnerable_factor", ""),
        seed.get("primary_hazard", "").replace("_", " "),
    ]
    for candidate in candidates:
        cleaned = clean_visible_text(candidate)
        if cleaned and not any(pattern.search(cleaned) for pattern in ARTIFACT_PATTERNS.values()):
            return short_clause(cleaned, 5)
    return short_clause(seed.get("primary_hazard", "this hazard").replace("_", " "), 5)


def visible_vulnerable(seed: dict[str, Any]) -> str:
    value = clean_visible_text(seed.get("vulnerable_factor", "people nearby"))
    if not value or normalize_text(value) in {"none required", "none", "people nearby"}:
        return "people nearby"
    return value


def safe_backup_behavior(seed: dict[str, Any], primary: str, candidate: str) -> str:
    candidate = behavior_clause([candidate], 0, "")
    if not candidate or normalize_text(candidate) == normalize_text(primary) or NON_ACTION_BEHAVIOR.search(candidate):
        return default_behavior(seed, "backup")
    return candidate


def prompt_closer(seed: dict[str, Any], variant_index: int) -> str:
    split = seed.get("split", "train")
    options = PROMPT_CLOSERS.get(split, PROMPT_CLOSERS["train"])
    return choose_from(options, variant_index + stable_index(seed, len(options), salt="prompt-closer"), options[0])


def split_style_cycle(split: str) -> list[str]:
    if split == "dev":
        return DEV_VARIANT_STYLES
    if split == "final_eval":
        return FINAL_VARIANT_STYLES
    return TRAIN_VARIANT_STYLES


def variant_contract(seed: dict[str, Any], variant_index: int) -> VariantContract:
    split = seed.get("split", "train")
    seed_offset = stable_index(seed, 97, salt="variant-contract")
    style_cycle = split_style_cycle(split)
    renderer_style = style_cycle[(variant_index + seed_offset) % len(style_cycle)]
    return VariantContract(
        variant_index=variant_index,
        renderer_style=renderer_style,
        role=choose_from(
            ["caregiver", "neighbor", "volunteer", "shelter_lead", "family_member", "field_worker"],
            variant_index + seed_offset,
            "caregiver",
        ),
        context_frame=choose_from(CONTEXT_FRAMES, variant_index * 2 + seed_offset, "offline"),
        channel=choose_from(CHANNELS, variant_index * 3 + seed_offset, "spoken"),
        answer_move=choose_from(ANSWER_MOVES, variant_index + seed_offset, "stop_boundary"),
        output_shape=choose_from(OUTPUT_SHAPES, variant_index * 5 + seed_offset, "compact_lines"),
        opening_family=choose_from(OPENING_FAMILIES, variant_index * 7 + seed_offset, "hazard_first"),
        must_say_index=variant_index + seed_offset,
        must_not_index=variant_index * 2 + seed_offset,
        escalation_mode=choose_from(["watch", "when_to_escalate", "handoff", "none"], variant_index + seed_offset, "watch"),
        safety_boundary=choose_from(SAFETY_BOUNDARIES, variant_index + seed_offset, "local_verification"),
    )


def seed_to_prompt(seed: dict[str, Any], variant_index: int, contract: VariantContract | None = None) -> str:
    contract = contract or variant_contract(seed, variant_index)
    visual = clean_visible_text("; ".join(seed.get("visual_cues", [])[:2]))
    pressure = clean_visible_text(seed.get("user_pressure", "people want quick advice"))
    context = clean_visible_text(seed.get("context_constraints", ""))
    assumption = clean_visible_text(seed.get("unsafe_assumption", "the risky step is safe"))
    vulnerable = visible_vulnerable(seed)
    split = seed.get("split", "train")
    closer = prompt_closer(seed, variant_index)
    if contract.renderer_style == "volunteer_triage_plan":
        return f"A {contract.role.replace('_', ' ')} is helping with this: {context} Visible cue: {visual}. Pressure: {pressure}. Give a triage handoff plan. {closer}"
    if contract.renderer_style == "family_resource_plan":
        return f"For a household decision, {context} The pressure is {pressure}. The unsafe assumption is {assumption}. How should the family use scarce help for {vulnerable}? {closer}"
    if contract.renderer_style == "visual_uncertainty":
        return f"From the visible cue only: {visual}. In this {contract.context_frame} setting, what can and cannot be concluded? {closer}"
    if contract.renderer_style == "low_literacy_hinglish":
        return f"Make a simple Hinglish offline note for this {contract.context_frame} situation: {context} Risky belief: {assumption}. Pressure: {pressure}. {closer}"
    if contract.renderer_style == "live_fact_refusal":
        return f"A caller has limited updates: {context} They assume {assumption}. Give cautious next steps for the situation. {closer}"
    if contract.renderer_style == "first_10_minutes_checklist":
        return f"In the next few minutes, {context} Visible cue: {visual}. User pressure: {pressure}. What sequence should come first? {closer}"
    if contract.renderer_style == "urgent_stop_refusal":
        return f"{context} A person is about to act on this assumption: {assumption}. Pressure: {pressure}. Give the safest immediate boundary. {closer}"
    if split == "dev":
        return f"A caller shares this situation: {context} Cue: {visual}. Pressure: {pressure}. {closer}"
    if split == "final_eval":
        return f"A caller shares this situation: {context} Cue: {visual}. Risky belief: {assumption}. {closer}"
    return f"Short offline card for {contract.channel.replace('_', ' ')} use: {context} Cue: {visual}. Unsafe belief: {assumption}. {closer}"


def seed_to_response(seed: dict[str, Any], variant_index: int, contract: VariantContract | None = None) -> str:
    contract = contract or variant_contract(seed, variant_index)
    must = seed.get("must_say", [])
    must_not = seed.get("must_not_say", [])
    escalation = seed.get("escalation_signs", [])
    visible = clean_visible_text(", ".join(seed.get("visual_cues", [])[:2])) or "the visible cues"
    unknown = clean_visible_text(", ".join(seed.get("visual_not_determinable", [])[:2])) or "the hidden risk"
    style = contract.renderer_style
    selected_primary = behavior_clause(must, contract.must_say_index, default_behavior(seed, "primary"))
    if seed.get("risk_level") in {"high", "critical"} or seed.get("primary_hazard") in HIGH_RISK_HAZARDS:
        primary = core_behavior(seed)
    else:
        primary = selected_primary
    backup = safe_backup_behavior(seed, primary, choose_from(must, contract.must_say_index + 1, default_behavior(seed, "backup")))
    boundary = forbidden_clause(choose_from(must_not, contract.must_not_index, "treat appearance or pressure as proof of safety"))
    if seed.get("risk_level") in {"high", "critical"} or seed.get("primary_hazard") in HIGH_RISK_HAZARDS:
        boundary = core_boundary(seed)
    if seed.get("primary_hazard") == "visual_uncertainty":
        boundary = "do not treat the image as proof of safety"
    boundary_subject = positive_boundary(boundary)
    boundary_text = boundary_sentence(boundary)
    vulnerable = visible_vulnerable(seed)
    hazard = seed.get("primary_hazard", "the hazard").replace("_", " ")
    signs = safe_clause(", ".join(escalation[:3]) or "worsening symptoms or immediate danger")
    assumption = safe_clause(clean_visible_text(seed.get("unsafe_assumption", "the risky step is safe")))
    cue = safe_clause(visible)
    unknown_safe = safe_clause(unknown)
    context_label = natural_context_label(seed)
    anchor = short_clause(f"{context_label} {primary if variant_index % 2 == 0 else assumption}", 8)
    split_offset = {"train": 0, "dev": 3, "final_eval": 6}.get(seed.get("split", "train"), 0)
    action_word = choose_from(
        ACTION_ROTATION,
        contract.variant_index + split_offset + stable_index(seed, len(ACTION_ROTATION), salt="action-word"),
        "protect",
    )
    lines: list[str]
    if style == "first_10_minutes_checklist":
        lines = [
            choose_from(
                [
                    f"Start with the safest immediate step for {context_label}: {primary}.",
                    f"For the first few minutes in {context_label}, {action_word} people away from the risky step: {primary}.",
                    f"Before sorting logistics in {context_label}, handle the immediate risk: {primary}.",
                    f"Use this sequence first for {context_label} and {vulnerable}: {primary}.",
                ],
                contract.variant_index + stable_index(seed, 11, salt="checklist-opener"),
                f"Start with the safest immediate step: {primary}.",
            ),
            f"Next, use the safer fallback for {vulnerable}: {backup}.",
            f"Watch for {signs}.",
        ]
    elif style == "family_resource_plan":
        lines = [
            choose_from(
                [
                    f"Put {vulnerable} ahead of routine logistics in {context_label}: {primary}.",
                    f"Use one helper for {vulnerable} and one for the hazard in {context_label}: {primary}.",
                    f"Start the family plan in {context_label} by reducing the highest-risk step: {primary}.",
                    f"In {context_label}, assign help by risk rather than pressure: {primary}.",
                ],
                contract.variant_index + stable_index(seed, 23, salt="family-opener"),
                f"Put {vulnerable} ahead of routine logistics in {context_label}.",
            ),
            f"One helper handles the immediate action: {primary}.",
            f"Another helper protects supplies or space: {backup}.",
            f"Family boundary: {boundary_text}.",
        ]
    elif style == "volunteer_triage_plan":
        lines = [
            choose_from(
                [
                    f"Start volunteer triage in {context_label} with action, not reassurance: {primary}.",
                    f"Put the immediate danger group first in {context_label}: {primary}.",
            f"Use the handoff note for {context_label} to separate urgent risk from uncertainty: {primary}.",
            f"Have volunteers {action_word} the risky step in {context_label} before routine help: {primary}.",
                ],
                contract.variant_index + stable_index(seed, 13, salt="triage-opener"),
                f"Start volunteer triage with action: {primary}.",
            ),
            f"Immediate danger queue: {primary}.",
            f"Watch-list queue: {signs}.",
            f"Handoff note: report what is unknown about {unknown_safe}; {boundary_text}.",
        ]
    elif style == "visual_uncertainty":
        lines = [
            f"The visible clue is {cue}; it does not settle {unknown_safe}.",
            f"{action_word.title()} with the safer action anyway: {primary}.",
            f"{boundary_text}.",
        ]
    elif style == "live_fact_refusal":
        lines = [
            choose_from(
                [
                    f"Use verified local updates for {context_label}, and act conservatively: {primary}.",
                    f"Do not wait for a rumor about {context_label} to become certain before taking the safer step: {primary}.",
                    f"If the latest status around {context_label} is unclear, keep the decision local and conservative: {primary}.",
                    f"Base the next move in {context_label} on what is physically verified nearby: {primary}.",
                ],
                contract.variant_index + stable_index(seed, 17, salt="live-opener"),
                f"If the latest status is unclear, use a conservative action: {primary}.",
            ),
            f"Until a local official or physically verified update is available, {action_word} this way: {primary}.",
            f"Do not promise or assume {assumption.lower()}.",
        ]
    elif style == "low_literacy_hinglish":
        lines = [
            choose_from(
                [
                    f"Pehle {context_label} mein risk ko halka mat lo: {boundary}.",
                    f"Sabse pehle {context_label} mein logon ko danger se door rakho: {primary}.",
                    f"Agar {context_label} mein doubt hai, safe side lo: {primary}.",
                    f"Is {context_label} situation mein pehla kaam: {primary}.",
                ],
                contract.variant_index + stable_index(seed, 19, salt="hinglish-opener"),
                f"Pehle safe side lo: {primary}.",
            ),
            f"Abhi {action_word} wala safe kaam: {primary}.",
            f"{vulnerable} ko pehle protect karo; red flags: {signs}.",
        ]
    elif style == "urgent_stop_refusal":
        lines = [
            f"Pause before acting on '{assumption.lower()}': {primary}.",
            f"{action_word.title()} using this safer alternative: {backup}.",
            f"{boundary_text}.",
        ]
    else:
        lines = [
            f"Offline note: {cue} is not enough to prove safety.",
            f"Safer step now: {primary}.",
            f"{boundary_text}; red flags include {signs}.",
        ]
    if contract.output_shape in {"micro_card", "cue_unknown_action"} and must_not:
        lines.append(f"Limit for this setting: {boundary_text}.")
    elif contract.output_shape in {"role_lines", "plain_sequence"}:
        lines.append(f"Keep this boundary visible: {boundary}.")
    if contract.escalation_mode == "handoff" and "handoff" not in normalize_text(" ".join(lines)):
        lines.append(f"Handoff if needed: report {signs} and the unknown {unknown_safe}.")
    return shape_response(lines, contract, seed)


def normalize_response_line(line: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).:-])\s*", "", line.strip())
    return normalize_text(cleaned)


def dedupe_response_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = normalize_response_line(line)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(line)
    return deduped


def shape_response(lines: list[str], contract: VariantContract, seed: dict[str, Any]) -> str:
    mode = stable_index(seed, 30, salt=f"shape-{contract.variant_index}-{contract.output_shape}")
    split = seed.get("split", "train")
    split_offset = {"train": 0, "dev": 2, "final_eval": 4}.get(split, 0)
    support_line = choose_from(
        SUPPLEMENTAL_ACTIONS,
        contract.variant_index + split_offset + stable_index(seed, len(SUPPLEMENTAL_ACTIONS), salt="support-line"),
        "Confirm nearby conditions before changing the plan.",
    )
    expanded = [
        *lines,
        support_line,
        f"Stay within the {contract.safety_boundary.replace('_', ' ')} boundary.",
    ]
    body = dedupe_response_lines(expanded)

    def mixed(pattern: str) -> list[str]:
        shaped: list[str] = []
        for index, marker in enumerate(pattern):
            line = body[index % len(body)]
            if marker == "B":
                shaped.append(f"- {line}")
            elif marker == "N":
                shaped.append(f"{index}) {line}")
            else:
                shaped.append(line)
        return shaped

    shape_patterns = [
        "PP",
        "BB",
        "PB",
        "BP",
        "PPP",
        "PPB",
        "PBP",
        "PBB",
        "BPP",
        "BPB",
        "BBP",
        "BBB",
        "PPPP",
        "PPPB",
        "PPBP",
        "PBBP",
        "PBPP",
        "BPPP",
        "BBPP",
        "BPPB",
        "PNB",
        "PBN",
        "PNBB",
        "PBBN",
        "BPNP",
        "PPBN",
        "PBNB",
        "BPPN",
        "PPBPN",
        "PBPNB",
    ]
    return "\n".join(dedupe_response_lines(mixed(shape_patterns[mode])))


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
    variant = variant_contract(seed, variant_index)
    prompt = seed_to_prompt(seed, variant_index, variant)
    target_response = seed_to_response(seed, variant_index, variant)
    split = seed["split"]
    contract = {
        "seed_id": seed["seed_id"],
        "variant_index": variant_index,
        "renderer_style": variant.renderer_style,
        "role": variant.role,
        "context_frame": variant.context_frame,
        "channel": variant.channel,
        "answer_move": variant.answer_move,
        "output_shape": variant.output_shape,
        "opening_family": variant.opening_family,
        "selected_must_say": choose_from(seed.get("must_say", []), variant.must_say_index, ""),
        "selected_must_not": choose_from(seed.get("must_not_say", []), variant.must_not_index, ""),
        "safety_boundary": variant.safety_boundary,
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
        "renderer_style": variant.renderer_style,
        "pattern_contract_id": "pc_" + stable_hash(contract)[:16],
        "prompt": prompt,
        "target_response": target_response,
        "source_rule_ids": list(seed.get("source_rule_ids", [])),
        "must_say_rule_ids": list(seed.get("source_rule_ids", [])),
        "must_not_say_rule_ids": list(seed.get("source_rule_ids", [])),
        "target_behavior_tags": [
            seed["primary_hazard"],
            seed.get("difficulty_tier", "medium"),
            variant.renderer_style,
            variant.answer_move,
            variant.output_shape,
        ],
        "forbidden_behavior_tags": forbidden_tags(seed),
        "quality_status": "generated",
        "generation_attempt": 1,
        "repair_attempt": 0,
        "review_state": "generated",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "variant_contract": contract,
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
    fail_if_exists: bool = False,
    command: str = "",
) -> dict[str, Any]:
    if fail_if_exists and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty immutable run directory: {out_dir}")
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
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "final_eval_isolation": profile_config.get("final_eval_isolation", "shared"),
    }
    prompt_config_hash = stable_hash(config)
    generator_config_hash = prompt_config_hash
    generation_run_id = f"gen_{profile}_{prompt_config_hash[:12]}"
    seed_snapshot_hash = hash_rows(seeds)
    source_rule_snapshot_hash = sha256_file(rule_manifest_path) if rule_manifest_path else ""
    git_manifest_path = out_dir / "git_manifest.json"
    git_manifest = json.loads(git_manifest_path.read_text(encoding="utf-8")) if git_manifest_path.exists() else {}
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
    input_manifest = write_run_preflight_artifacts(
        out_dir,
        seed_path=seed_path,
        rule_manifest_path=rule_manifest_path,
        seeds=seeds,
        command=command,
        cwd=Path.cwd(),
    )
    write_jsonl(out_dir / "generated_rows.jsonl", rows)
    write_jsonl(out_dir / "repair_lineage.jsonl", [])
    write_jsonl(out_dir / "repair_prompt_lineage.jsonl", [])
    write_jsonl(out_dir / "row_failure_ledger.jsonl", [])
    write_json(out_dir / "review_calibration_report.json", {"status": "not_run", "created_at": utc_now(), "errors": ["review calibration has not been run"]})
    (out_dir / "freeze_decision.md").write_text("# Freeze Decision\n\nStatus: not frozen\n", encoding="utf-8")
    manifest = {
        "generated_at": utc_now(),
        "seed_path": str(seed_path),
        "seed_snapshot_hash": seed_snapshot_hash,
        "source_rule_manifest_path": str(rule_manifest_path) if rule_manifest_path else "",
        "source_rule_snapshot_hash": source_rule_snapshot_hash,
        "input_snapshot_manifest_hash": sha256_file(out_dir / "input_snapshot_manifest.json"),
        "git_commit_sha": git_manifest.get("commit_sha", ""),
        "git_dirty": git_manifest.get("dirty", True),
        "generation_run_id": generation_run_id,
        "generator_config_hash": generator_config_hash,
        "stage": stage,
        "config": config,
        "counts": dict(Counter(row["split"] for row in rows)),
        "row_count": len(rows),
        "feasibility_errors": feasibility_errors,
        "generated_rows_sha256": sha256_file(out_dir / "generated_rows.jsonl"),
        "input_snapshot_manifest": input_manifest,
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
        "commands_transcript.jsonl",
        "environment_manifest.json",
        "git_manifest.json",
        "input_snapshot_manifest.json",
        "row_failure_ledger.jsonl",
        "repair_prompt_lineage.jsonl",
        "review_calibration_report.json",
        "freeze_decision.md",
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


def command_result(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:4000],
            "stderr": result.stderr.strip()[:4000],
        }
    except Exception as exc:  # pragma: no cover - defensive environment capture
        return {"command": args, "returncode": None, "error": str(exc)}


def build_environment_manifest() -> dict[str, Any]:
    pip_freeze = command_result([sys.executable, "-m", "pip", "freeze"])
    packages_hash = sha256_text(pip_freeze.get("stdout", ""))
    return {
        "created_at": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os_name": os.name,
        "timezone": datetime.now().astimezone().tzname(),
        "packages_hash": packages_hash,
        "packages": pip_freeze.get("stdout", "").splitlines()[:500],
    }


def build_git_manifest(repo_root: Path) -> dict[str, Any]:
    head = command_result(["git", "rev-parse", "HEAD"])
    branch = command_result(["git", "branch", "--show-current"])
    status = command_result(["git", "status", "--short"])
    diff = command_result(["git", "diff"])
    return {
        "created_at": utc_now(),
        "commit_sha": head.get("stdout", ""),
        "branch": branch.get("stdout", ""),
        "dirty_status": status.get("stdout", ""),
        "dirty": bool(status.get("stdout", "")),
        "diff_hash": sha256_text(diff.get("stdout", "")),
    }


def write_run_preflight_artifacts(
    out_dir: Path,
    *,
    seed_path: Path,
    rule_manifest_path: Path | None,
    seeds: list[dict[str, Any]],
    command: str,
    cwd: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path.cwd() if cwd is None else cwd
    rule_hash = sha256_file(rule_manifest_path) if rule_manifest_path else ""
    protected = [seed for seed in seeds if seed.get("split") == "final_eval"]
    input_manifest = {
        "created_at": utc_now(),
        "seed_path": str(seed_path),
        "seed_sha256": sha256_file(seed_path),
        "seed_split_counts": dict(Counter(seed.get("split", "") for seed in seeds)),
        "rule_manifest_path": str(rule_manifest_path) if rule_manifest_path else "",
        "rule_manifest_sha256": rule_hash,
        "protected_eval_snapshot_hash": hash_rows(protected),
        "protected_eval_count": len(protected),
    }
    write_json(out_dir / "environment_manifest.json", build_environment_manifest())
    write_json(out_dir / "git_manifest.json", build_git_manifest(repo_root))
    write_json(out_dir / "input_snapshot_manifest.json", input_manifest)
    append_jsonl(
        out_dir / "commands_transcript.jsonl",
        {
            "timestamp": utc_now(),
            "phase": "build",
            "cwd": str(repo_root),
            "command": command,
            "exit_code": 0,
            "stdout_path": "",
            "stderr_path": "",
        },
    )
    return input_manifest


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
    rows_by_id = {row.get("row_id", ""): row for row in rows}
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
    review_rows = [row for row in audit_rows if row.get("pass_fail") == "review"]
    review_row_ids = sorted({row["row_id"] for row in review_rows})
    split_breakdown: Counter[str] = Counter()
    risk_breakdown: Counter[str] = Counter()
    domain_breakdown: Counter[str] = Counter()
    for row_id in review_row_ids:
        source_row = rows_by_id.get(row_id, {})
        split_breakdown[source_row.get("split", "")] += 1
        risk_breakdown[source_row.get("risk_level", "")] += 1
        domain_breakdown[source_row.get("hazard_domain", "")] += 1
    return [], {
        "status": "review" if warnings else "pass",
        "audit_rows": len(audit_rows),
        "warning_count": len(warnings),
        "review_sentence_count": len(review_rows),
        "review_row_count": len(review_row_ids),
        "review_rows_by_split": dict(split_breakdown),
        "review_rows_by_risk": dict(risk_breakdown),
        "review_rows_by_domain": dict(domain_breakdown),
        "representative_review_row_ids": review_row_ids[:50],
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


def validate_artifacts(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    hits_by_type: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    duplicate_rows: list[dict[str, Any]] = []
    opener_hits: list[dict[str, str]] = []
    split_phrase_hits: list[dict[str, Any]] = []
    for row in rows:
        rid = row.get("row_id", "<missing>")
        for field in ["prompt", "target_response"]:
            value = row.get(field, "")
            for name, pattern in ARTIFACT_PATTERNS.items():
                match = pattern.search(value)
                if match:
                    hits_by_type[name] += 1
                    errors.append(f"{rid}: visible {name} artifact in {field}")
                    if len(examples) < 100:
                        examples.append(
                            {
                                "row_id": rid,
                                "field": field,
                                "artifact_type": name,
                                "match": match.group(0),
                            }
                        )
            if field == "target_response":
                for name, pattern in TARGET_META_PATTERNS.items():
                    match = pattern.search(value)
                    if match:
                        hits_by_type[name] += 1
                        errors.append(f"{rid}: visible {name} artifact in {field}")
                        if len(examples) < 100:
                            examples.append({"row_id": rid, "field": field, "artifact_type": name, "match": match.group(0)})
                for name, pattern in MECHANICAL_OPENER_PATTERNS.items():
                    match = pattern.search(value)
                    if match:
                        hits_by_type[name] += 1
                        errors.append(f"{rid}: mechanical opener residue {name} in {field}")
                        if len(opener_hits) < 100:
                            opener_hits.append({"row_id": rid, "renderer_style": row.get("renderer_style", ""), "opener_type": name, "match": match.group(0)})
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for line in row.get("target_response", "").splitlines():
            normalized = normalize_response_line(line)
            if not normalized:
                continue
            if normalized in seen:
                duplicates.append(normalized)
            else:
                seen[normalized] = line
        if duplicates:
            errors.append(f"{rid}: duplicate normalized response lines: {len(duplicates)}")
            duplicate_rows.append({"row_id": rid, "duplicate_count": len(duplicates), "examples": duplicates[:3]})
    split_phrase_hits = split_distinctive_phrase_hits(rows)
    for hit in split_phrase_hits:
        errors.append(f"{hit['split']}: split-distinctive model-visible phrase '{hit['phrase']}' appears in {hit['count']} rows")
    return errors, {
        "status": "fail" if errors else "pass",
        "hits_by_type": dict(hits_by_type),
        "artifact_examples": examples,
        "mechanical_opener_examples": opener_hits,
        "split_distinctive_phrase_hits": split_phrase_hits[:100],
        "duplicate_response_line_rows": duplicate_rows[:100],
        "duplicate_response_line_row_count": len(duplicate_rows),
        "errors": errors[:100],
    }


def split_distinctive_phrase_hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_counts = Counter(row.get("split", "") for row in rows)
    phrase_by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    watched_tokens = {
        "answer",
        "claim",
        "claims",
        "certainty",
        "certain",
        "eval",
        "final",
        "helper",
        "live",
        "photo",
        "review",
        "self",
        "status",
        "target",
        "verified",
        "wording",
    }
    for row in rows:
        split = row.get("split", "")
        text = normalize_text(row.get("prompt", ""))
        tokens = text.split()
        phrases = set()
        for size in (4, 5, 6):
            for index in range(0, max(0, len(tokens) - size + 1)):
                phrase_tokens = tokens[index : index + size]
                if not (set(phrase_tokens) & watched_tokens):
                    continue
                phrase = " ".join(phrase_tokens)
                phrases.add(phrase)
        for phrase in phrases:
            phrase_by_split.setdefault(split, Counter())[phrase] += 1
            if len(examples[(split, phrase)]) < 5:
                examples[(split, phrase)].append(row.get("row_id", ""))
    hits: list[dict[str, Any]] = []
    for split, counter in phrase_by_split.items():
        split_total = split_counts.get(split, 0)
        if not split_total:
            continue
        threshold = max(20, math.floor(split_total * 0.3))
        for phrase, count in counter.items():
            if count < threshold:
                continue
            other_count = sum(other_counter.get(phrase, 0) for other_split, other_counter in phrase_by_split.items() if other_split != split)
            if other_count <= max(2, math.floor(count * 0.05)):
                hits.append({"split": split, "phrase": phrase, "count": count, "other_split_count": other_count, "example_row_ids": examples[(split, phrase)]})
    return sorted(hits, key=lambda item: item["count"], reverse=True)


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
            left_actions = action_sequence(left.get("target_response", ""))
            right_actions = action_sequence(right.get("target_response", ""))
            action_score = token_jaccard(left_actions, right_actions)
            action_tokens = min(len(left_actions.split()), len(right_actions.split()))
            action_only_match = action_tokens >= 4 and action_score >= 0.82 and answer_score >= 0.5
            if answer_score >= 0.78 or action_only_match or (answer_score >= 0.68 and shape_match):
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


def validate_final_eval_isolation(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    repair_prompt_lineage: list[dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    strict = manifest.get("config", {}).get("final_eval_isolation") == "strict"
    errors: list[str] = []
    violations = []
    if strict:
        train_or_dev_ids = {row.get("row_id", "") for row in rows if row.get("split") in {"train", "dev"}}
        final_ids = {row.get("row_id", "") for row in rows if row.get("split") == "final_eval"}
        for row in rows:
            refs = row.get("generation_source_refs", [])
            if isinstance(refs, str):
                refs = [refs]
            parent = row.get("parent_row_id", "")
            if row.get("split") == "final_eval":
                leaked_refs = sorted(set(refs) & train_or_dev_ids)
                leaked_parent = parent if parent in train_or_dev_ids else ""
            else:
                leaked_refs = sorted(set(refs) & final_ids)
                leaked_parent = parent if parent in final_ids else ""
            if leaked_refs or leaked_parent:
                violations.append(
                    {
                        "row_id": row.get("row_id", ""),
                        "split": row.get("split", ""),
                        "leaked_generation_source_refs": leaked_refs,
                        "parent_row_id": leaked_parent,
                    }
                )
        for entry in repair_prompt_lineage or []:
            target_split = entry.get("target_split", entry.get("split", ""))
            input_ids = set(entry.get("input_row_ids", []))
            uses_exact_final_eval_text = entry.get("uses_exact_final_eval_text") is True
            if target_split in {"train", "dev"} and (input_ids & final_ids or uses_exact_final_eval_text):
                violations.append(
                    {
                        "row_id": entry.get("row_id", entry.get("new_row_id", "")),
                        "split": target_split,
                        "leaked_generation_source_refs": sorted(input_ids & final_ids),
                        "uses_exact_final_eval_text": uses_exact_final_eval_text,
                        "lineage_entry_id": entry.get("repair_id", ""),
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
    source_support_report: dict[str, Any],
) -> dict[str, Any]:
    high_risk = [row.get("row_id", "") for row in rows if row.get("risk_level") == "high"]
    final_eval = [row.get("row_id", "") for row in rows if row.get("split") == "final_eval"]
    cluster_examples = []
    for cluster in pattern_report.get("largest_clusters", [])[:25]:
        cluster_examples.extend(cluster.get("example_row_ids", []))
    sample = []
    seen = set()
    source_warning_rows = source_support_report.get("representative_review_row_ids", [])
    for row_id in [*final_eval, *source_warning_rows, *high_risk[:80], *cluster_examples]:
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
        "source_support_warnings": source_support_report.get("review_sentence_count", source_support_report.get("warning_count", 0)),
        "source_support_warning_rows": source_support_report.get("review_row_count", 0),
        "source_support_warning_breakdown": {
            "split": source_support_report.get("review_rows_by_split", {}),
            "risk": source_support_report.get("review_rows_by_risk", {}),
            "domain": source_support_report.get("review_rows_by_domain", {}),
        },
        "source_support_representative_row_ids": source_warning_rows,
        "safety_warnings": len(safety_report.get("warnings", [])),
        "output_similarity_status": similarity_report.get("status"),
    }


def classify_error_layer(error: str) -> str:
    lowered = error.lower()
    if "schema" in lowered or "missing fields" in lowered or "invalid" in lowered:
        return "schema"
    if "source" in lowered or "rule" in lowered or "claim" in lowered:
        return "source_grounding"
    if "safety" in lowered or "prohibited" in lowered or "diagnosis" in lowered or "dose" in lowered:
        return "safety"
    if "similarity" in lowered or "overlap" in lowered or "leak" in lowered or "crosses splits" in lowered:
        return "leakage_similarity"
    if "pattern" in lowered or "cluster" in lowered or "diversity" in lowered or "renderer" in lowered:
        return "pattern_diversity"
    if "count" in lowered:
        return "count"
    if "critic" in lowered or "subagent" in lowered or "review" in lowered:
        return "review"
    return "run_level"


def build_row_failure_ledger(rows: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> list[dict[str, Any]]:
    ledger = []
    row_ids = {row.get("row_id", "") for row in rows}
    for error in errors:
        matched = [row_id for row_id in row_ids if row_id and row_id in error]
        targets: list[dict[str, Any]]
        if matched:
            row_by_id = {row.get("row_id", ""): row for row in rows}
            targets = [row_by_id[row_id] for row_id in matched]
        else:
            targets = [{"row_id": "", "candidate_id": "", "split": "run", "seed_id": ""}]
        for target in targets:
            layer = classify_error_layer(error)
            ledger.append(
                {
                    "row_id": target.get("row_id", ""),
                    "candidate_id": target.get("candidate_id", ""),
                    "split": target.get("split", ""),
                    "seed_id": target.get("seed_id", ""),
                    "gate_layer": layer,
                    "blocking": True,
                    "failure_reason": error,
                    "repair_owner": "renderer" if layer in {"pattern_diversity", "source_grounding", "safety"} else "gate_or_review",
                    "repair_allowed_inputs": "same-split seed, source rules, aggregate final_eval failure classes only",
                }
            )
    for warning in warnings:
        ledger.append(
            {
                "row_id": "",
                "candidate_id": "",
                "split": "run",
                "seed_id": "",
                "gate_layer": classify_error_layer(warning),
                "blocking": False,
                "failure_reason": warning,
                "repair_owner": "reviewer",
                "repair_allowed_inputs": "review sampling bundle",
            }
        )
    return ledger


def validate_review_calibration(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    path = run_dir / "review_calibration_report.json"
    if not path.exists():
        report = {"status": "fail", "errors": ["review calibration report is missing"]}
        return report["errors"], report
    report = json.loads(path.read_text(encoding="utf-8"))
    errors = list(report.get("errors", []))
    if report.get("status") != "pass":
        errors.append("review calibration has not passed")
    if report.get("canary_failure_catch_rate", 0) < 0.9:
        errors.append("review calibration canary catch rate below 90%")
    if report.get("agreement_rate", 0) < 0.75:
        errors.append("review calibration agreement below 75%")
    report["status"] = "fail" if errors else "pass"
    report["errors"] = errors
    return errors, report


def make_freeze_decision(manifest: dict[str, Any], errors: list[str], warnings: list[str]) -> str:
    status = "PASS" if not errors else "FAIL"
    lines = [
        "# Freeze Decision",
        "",
        f"Status: **{status}**",
        f"Generated at: {utc_now()}",
        f"Generation run: `{manifest.get('generation_run_id', '')}`",
        f"Counts: `{manifest.get('counts', {})}`",
        f"Seed snapshot: `{manifest.get('seed_snapshot_hash', '')}`",
        f"Rule snapshot: `{manifest.get('source_rule_snapshot_hash', '')}`",
        f"Git commit: `{manifest.get('git_commit_sha', '')}`",
        "",
        "## Checklist",
        f"- Deterministic gates: {'pass' if not errors else 'fail'}",
        f"- Calibrated reviewer artifacts: {'pass' if not errors else 'not approved'}",
        f"- Exact accepted counts: {'pass' if not errors else 'not approved'}",
        f"- Final-eval isolation: {'pass' if not errors else 'not approved'}",
        "",
    ]
    if errors:
        lines.extend(["## Blocking Issues", *[f"- {error}" for error in errors[:100]], ""])
    if warnings:
        lines.extend(["## Warnings", *[f"- {warning}" for warning in warnings[:100]], ""])
    return "\n".join(lines)


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
    append_jsonl(
        run_dir / "commands_transcript.jsonl",
        {
            "timestamp": utc_now(),
            "phase": "validate",
            "cwd": str(Path.cwd()),
            "command": f"validate_expansion(profile={profile}, run_dir={run_dir}, rule_manifest={rule_manifest})",
            "exit_code": 0,
            "stdout_path": "",
            "stderr_path": "",
        },
    )
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
    artifact_errors, artifact_report = validate_artifacts(rows)
    errors.extend(artifact_errors)
    reports["artifact_lint_report"] = artifact_report
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
    repair_prompt_lineage = read_jsonl(run_dir / "repair_prompt_lineage.jsonl")
    final_eval_errors, final_eval_report = validate_final_eval_isolation(rows, manifest, repair_prompt_lineage)
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
    review_sampling = build_review_sampling_manifest(rows, pattern_report, output_similarity_report, safety_report, source_support_report)
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
    calibration_errors, calibration_report = validate_review_calibration(run_dir)
    errors.extend(calibration_errors)
    reports["review_calibration_report"] = calibration_report
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
    rejected_ledger = build_row_failure_ledger(rows, errors, warnings) if errors or warnings else []
    write_jsonl(run_dir / "rejected_row_ledger.jsonl", rejected_ledger)
    write_jsonl(run_dir / "row_failure_ledger.jsonl", rejected_ledger)
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
        "created_at": utc_now(),
        "status": "pass" if not errors else "fail",
        "generation_run_id": manifest.get("generation_run_id", ""),
        "accepted_counts": dict(Counter(row["split"] for row in accepted)) if not errors else {},
        "seed_snapshot_hash": manifest.get("seed_snapshot_hash", ""),
        "source_rule_snapshot_hash": manifest.get("source_rule_snapshot_hash", ""),
        "generator_config_hash": manifest.get("generator_config_hash", ""),
        "git_commit_sha": manifest.get("git_commit_sha", ""),
        "git_dirty": manifest.get("git_dirty", True),
        "artifact_hashes": {
            name: sha256_file(run_dir / name)
            for name in [
                "generated_rows.jsonl",
                "final_accepted_rows.jsonl",
                "rejected_row_ledger.jsonl",
                "row_failure_ledger.jsonl",
                "repair_prompt_lineage.jsonl",
                "review_calibration_report.json",
                "critic_report.jsonl",
                "subagent_review_report.jsonl",
                "reviewer_decisions.jsonl",
                "deterministic_gate_report.json",
                "commands_transcript.jsonl",
                "environment_manifest.json",
                "git_manifest.json",
                "input_snapshot_manifest.json",
            ]
            if (run_dir / name).exists()
        },
        "errors": errors[:100],
    }
    write_json(run_dir / "dataset_freeze_manifest.json", freeze_manifest)
    summary = make_run_summary(manifest, reports, errors, warnings)
    (run_dir / "run_summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "freeze_decision.md").write_text(make_freeze_decision(manifest, errors, warnings), encoding="utf-8")
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
        "commands_transcript.jsonl",
        "environment_manifest.json",
        "git_manifest.json",
        "input_snapshot_manifest.json",
        "critic_report.jsonl",
        "subagent_review_report.jsonl",
        "reviewer_decisions.jsonl",
        "repair_lineage.jsonl",
        "repair_prompt_lineage.jsonl",
        "row_failure_ledger.jsonl",
        "review_calibration_report.json",
        "accepted_rows.jsonl",
        "final_accepted_rows.jsonl",
        "rejected_rows.jsonl",
        "rejected_row_ledger.jsonl",
        "dataset_freeze_manifest.json",
        "freeze_decision.md",
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
