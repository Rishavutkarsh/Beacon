from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANONICAL_SEED_FIELDS = {
    "seed_id",
    "seed_family_id",
    "primary_hazard",
    "secondary_hazards",
    "unsafe_assumption",
    "context_constraints",
    "user_pressure",
    "vulnerable_factor",
    "visual_cues",
    "visual_not_determinable",
    "must_say",
    "must_not_say",
    "escalation_signs",
    "source_ids",
    "source_rule_ids",
    "incident_archetype_id",
    "india_context",
    "difficulty_tier",
    "answer_style",
    "split",
    "privacy_risk",
    "sensationalism_risk",
    "review_status",
    "reject_reason",
}
PROPOSAL_EXTRA_FIELDS = {
    "generator_batch_id",
    "assignment_category",
    "core_decision_problem",
    "answer_archetype",
    "unsafe_assumption_type",
    "user_pressure_type",
    "setting_type",
    "hazard_subtype",
    "disaster_phase",
    "affected_population",
    "operational_decision",
    "unsafe_action_traps",
    "required_refusal_escalation_behavior",
    "source_basis",
    "source_grounding_need",
    "allowed_safe_guidance",
    "forbidden_claims",
    "safe_action_boundary",
    "similarity_risk_notes",
    "novelty_rationale",
    "intended_gap_filled",
    "do_not_overlap_with",
    "nearest_train_seed_ids",
    "nearest_dev_final_seed_ids",
    "why_not_dev_final_overlap",
    "why_this_is_train_only_safe",
    "real_incident_check",
}
REQUIRED_PROPOSAL_FIELDS = CANONICAL_SEED_FIELDS | PROPOSAL_EXTRA_FIELDS
LIST_FIELDS = {
    "secondary_hazards",
    "visual_cues",
    "visual_not_determinable",
    "must_say",
    "must_not_say",
    "escalation_signs",
    "source_ids",
    "source_rule_ids",
    "do_not_overlap_with",
    "nearest_train_seed_ids",
    "nearest_dev_final_seed_ids",
    "source_basis",
    "unsafe_action_traps",
}
SURFACE_NOVELTY_PATTERNS = [
    re.compile(r"\bdifferent (?:city|village|location|place|person|name|wording)\b", re.I),
    re.compile(r"\bonly (?:the )?(?:city|village|location|place|person|name|wording) (?:is|was|changed)\b", re.I),
]
OPERATIONAL_FACT_PATTERNS = {
    "helpline_or_phone": re.compile(r"(?:\+?\d[\s-]?){8,}"),
    "live_status": re.compile(r"\b(?:road|bridge|shelter|rescue|dam|warning|weather)\s+(?:is|are|will be)\s+(?:open|closed|available|safe|coming|clear|released)\b", re.I),
    "official_order": re.compile(r"\b(?:official|collector|police|ndma|sdma|government)\s+(?:said|ordered|announced|confirmed)\b", re.I),
    "casualty_count": re.compile(r"\b\d+\s+(?:dead|killed|missing|injured|casualties|trapped)\b", re.I),
    "dam_level": re.compile(r"\b(?:dam|reservoir)\s+(?:level|release|gate)\s+(?:is|at|opened|closed)\b", re.I),
    "timestamp": re.compile(r"\b(?:today|tomorrow|yesterday|tonight|this morning|at \d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", re.I),
    "named_institution": re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s+(?:Hospital|School|College|Police|Station|NGO|Trust|Company)\b"),
}
HAZARD_TARGETS = [
    "heatwave_cold_lightning_dust",
    "urban_fire_lpg_chemical",
    "dam_flash_flood_riverbank_coastal",
    "crowd_shelter_overcrowding",
    "post_disaster_contamination_infection",
    "accessibility_elder_disabled_pregnancy_child_language",
    "misinformation_fake_alerts_helplines_rescue",
    "infrastructure_power_telecom_road_transit",
]
DEFAULT_DIVERSITY_REJECTS = {
    "ss_exp_b05_001",
    "ss_exp_b05_007",
    "ss_exp_b05_009",
    "ss_exp_b05_010",
    "ss_seed_batch07_001",
    "ss_seed_batch07_002",
    "ss_seed_batch07_006",
    "ss_seed_batch07_008",
    "ss_seed_batch07_011",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def seed_text(seed: dict[str, Any]) -> str:
    parts = [
        seed.get("primary_hazard", ""),
        seed.get("unsafe_assumption", ""),
        seed.get("context_constraints", ""),
        seed.get("user_pressure", ""),
        seed.get("vulnerable_factor", ""),
        " ".join(seed.get("visual_cues", [])),
        " ".join(seed.get("must_say", [])),
        " ".join(seed.get("must_not_say", [])),
        seed.get("answer_style", ""),
        " ".join(seed.get("source_basis", [])),
        seed.get("operational_decision", ""),
        " ".join(seed.get("unsafe_action_traps", [])),
    ]
    return " ".join(parts)


def scenario_text(seed: dict[str, Any]) -> str:
    return " ".join(
        [
            seed.get("unsafe_assumption", ""),
            seed.get("context_constraints", ""),
            seed.get("user_pressure", ""),
            seed.get("vulnerable_factor", ""),
            " ".join(seed.get("visual_cues", [])),
        ]
    )


def decision_signature(seed: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            [
                seed.get("core_decision_problem", ""),
                seed.get("unsafe_assumption_type", ""),
                seed.get("user_pressure_type", ""),
                seed.get("answer_archetype", seed.get("answer_style", "")),
                " ".join(seed.get("source_rule_ids", [])),
            ]
        )
    )


def token_jaccard(left: str, right: str) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def overlap_ratio(left: list[str], right: list[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, min(len(a), len(b)))


def hazard_balance(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_split": dict(Counter(seed.get("split", "") for seed in seeds)),
        "by_primary_hazard": dict(Counter(seed.get("primary_hazard", "") for seed in seeds)),
        "by_answer_style": dict(Counter(seed.get("answer_style", "") for seed in seeds)),
        "by_difficulty": dict(Counter(seed.get("difficulty_tier", "") for seed in seeds)),
    }


def redacted_protected_cluster(seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": seed.get("incident_pattern_group_base") or seed.get("incident_pattern_group") or seed.get("seed_family_id"),
        "split": seed.get("split"),
        "primary_hazard": seed.get("primary_hazard"),
        "difficulty_tier": seed.get("difficulty_tier"),
        "answer_style": seed.get("answer_style"),
        "source_rule_ids": seed.get("source_rule_ids", []),
        "decision_shape": normalize_text(
            " ".join(
                [
                    seed.get("primary_hazard", ""),
                    seed.get("vulnerable_factor", ""),
                    seed.get("answer_style", ""),
                    " ".join(seed.get("source_rule_ids", [])),
                ]
            )
        ),
        "avoid_note": "Protected eval cluster. Do not recreate its decision problem, pressure dynamic, or answer move.",
    }


def target_gap_matrix(train_seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hazard_counts = Counter(seed.get("primary_hazard", "") for seed in train_seeds)
    style_counts = Counter(seed.get("answer_style", "") for seed in train_seeds)
    return [
        {
            "target_gap_id": target,
            "why_needed": "Underrepresented or absent disaster coverage relative to the current flood-heavy seed bank.",
            "avoid": "Do not invent live operational facts; keep source-grounded stable safety behavior.",
            "current_related_train_count": sum(count for hazard, count in hazard_counts.items() if any(part in hazard for part in target.split("_"))),
            "underused_styles": [style for style, count in style_counts.items() if count <= 10],
        }
        for target in HAZARD_TARGETS
    ]


def build_seed_expansion_packets(
    seed_path: Path,
    out_dir: Path,
    *,
    leakage_report_path: Path | None = None,
    pattern_report_path: Path | None = None,
) -> dict[str, Any]:
    seeds = read_jsonl(seed_path)
    train = [seed for seed in seeds if seed.get("split") == "train"]
    protected = [seed for seed in seeds if seed.get("split") in {"dev", "final_eval"}]
    leakage_report = json.loads(leakage_report_path.read_text(encoding="utf-8")) if leakage_report_path and leakage_report_path.exists() else {}
    pattern_report = json.loads(pattern_report_path.read_text(encoding="utf-8")) if pattern_report_path and pattern_report_path.exists() else {}
    generator_packet = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Generate train-only seed proposals without exact dev/final exposure.",
        "instructions": [
            "Use train seeds, schema, assigned themes, and generic quality rules only.",
            "Do not use dev/final-derived summaries, leakage reports, reviewer comments, or protected-specific rejection rationales.",
            "Do not try to infer or reconstruct protected dev/final examples.",
            "Novelty must be a different decision problem, not changed names, locations, or wording.",
            "Use generic or clearly fictionalized local details; do not invent live operational facts.",
            "Prefer source-grounded, plausible, operationally useful public-safety decisions over rare novelty.",
            "Do not invent official alerts, casualty numbers, helpline numbers, evacuation orders, medical treatment advice, or technical repair/containment instructions.",
        ],
        "train_seed_cards": train,
        "hazard_style_balance": hazard_balance(seeds),
        "target_gap_matrix": target_gap_matrix(train),
        "required_proposal_fields": sorted(REQUIRED_PROPOSAL_FIELDS),
        "coverage_targets": HAZARD_TARGETS,
    }
    reviewer_packet = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Review and gate train-only seed proposals with full leakage context.",
        "all_seed_cards": seeds,
        "protected_seed_ids": [seed["seed_id"] for seed in protected],
        "train_seed_ids": [seed["seed_id"] for seed in train],
        "leakage_report": leakage_report,
        "pattern_report": pattern_report,
        "hard_reject_rules": [
            "exact normalized seed/scenario match",
            "closest dev/final neighbor over threshold",
            "top 3 neighbors include 2 or more dev/final seeds",
            "same core decision problem as dev/final with actor/place/object swapped",
            "invented operational facts",
            "surface-only novelty rationale",
            "specific real incident resemblance",
        ],
    }
    assignments = make_subagent_assignments(generator_packet)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "generator_packet.json", generator_packet)
    write_json(out_dir / "reviewer_gate_packet.json", reviewer_packet)
    write_json(out_dir / "subagent_assignments.json", assignments)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_path": str(seed_path),
        "out_dir": str(out_dir),
        "train_seed_count": len(train),
        "protected_seed_count": len(protected),
        "generator_packet_sha256": sha256_text(json.dumps(generator_packet, ensure_ascii=False, sort_keys=True)),
        "reviewer_packet_sha256": sha256_text(json.dumps(reviewer_packet, ensure_ascii=False, sort_keys=True)),
        "assignment_count": len(assignments["assignments"]),
    }
    write_json(out_dir / "seed_expansion_packet_manifest.json", manifest)
    return manifest


def make_subagent_assignments(generator_packet: dict[str, Any]) -> dict[str, Any]:
    targets = generator_packet["target_gap_matrix"]
    assignments = []
    for index in range(0, len(targets), 2):
        slice_targets = targets[index : index + 2]
        assignments.append(
            {
                "assignment_id": f"seed_expansion_batch_{len(assignments) + 1:02d}",
                "proposal_target": "15-20",
                "split": "train",
                "target_gap_ids": [item["target_gap_id"] for item in slice_targets],
                "mini_quota_matrix": {
                    "setting_type": ["urban", "rural", "peri_urban", "transit", "informal_settlement", "institutional"],
                    "actor_type": ["ASHA_or_health_worker", "panchayat_or_ward_worker", "school_admin", "bus_or_transit_staff", "relief_volunteer", "family_caregiver"],
                    "decision_pressure": ["triage", "verification", "prioritization", "escalation", "communication", "resource_allocation"],
                    "vulnerable_group": ["none_required", "elder", "disabled_person", "pregnant_person", "child", "language_barrier"],
                    "answer_archetype": ["urgent_stop", "first_steps", "resource_plan", "volunteer_triage", "uncertainty_boundary", "short_offline_card"],
                },
                "required_answer_move_constraints": [
                    "at least 3 distinct answer_archetype values",
                    "at least 2 low-network or offline-safe scenarios",
                    "no live-status confirmation tasks unless the answer boundary is the core lesson",
                    "at least 3 distinct setting_type values",
                    "at least 3 distinct decision_pressure values",
                ],
                "forbidden": [
                    "do not use exact protected dev/final text",
                    "do not invent official orders, live road/shelter/rescue status, casualty counts, helplines, dam levels, or timestamps",
                    "do not satisfy novelty by changing names or places only",
                ],
            }
        )
    return {"assignments": assignments}


def load_rule_ids(rule_manifest: Path | None) -> set[str]:
    if not rule_manifest or not rule_manifest.exists():
        return set()
    return {row["rule_id"] for row in read_jsonl(rule_manifest)}


def validate_seed_proposals(
    proposal_path: Path,
    seed_path: Path,
    out_dir: Path,
    *,
    rule_manifest: Path | None = None,
    v1_rows_path: Path | None = None,
) -> dict[str, Any]:
    proposals = read_jsonl(proposal_path)
    existing = read_jsonl(seed_path)
    known_rule_ids = load_rule_ids(rule_manifest)
    v1_rows = read_jsonl(v1_rows_path) if v1_rows_path else []
    train = [seed for seed in existing if seed.get("split") == "train"]
    protected = [seed for seed in existing if seed.get("split") in {"dev", "final_eval"}]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    new_cluster_counts: Counter[str] = Counter()
    for proposal in proposals:
        result = evaluate_proposal(proposal, existing, train, protected, accepted, known_rule_ids, v1_rows, new_cluster_counts)
        neighbor_rows.extend(result["neighbors"])
        record = {**proposal, "gate_status": result["status"], "gate_reasons": result["reasons"], "gate_warnings": result["warnings"]}
        if result["status"] == "accepted":
            accepted.append(record)
            new_cluster_counts[result["cluster_key"]] += 1
        elif result["status"] == "review":
            review.append(record)
        else:
            rejected.append(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "accepted_seed_proposals.jsonl", accepted)
    write_jsonl(out_dir / "review_seed_proposals.jsonl", review)
    write_jsonl(out_dir / "rejected_seed_proposals.jsonl", rejected)
    write_csv(out_dir / "seed_proposal_neighbors.csv", neighbor_rows)
    metadata_rows = metadata_collision_rows(proposals)
    write_csv(out_dir / "seed_metadata_collision_report.csv", metadata_rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_count": len(proposals),
        "accepted_count": len(accepted),
        "review_count": len(review),
        "rejected_count": len(rejected),
        "status": "pass" if len(accepted) >= 80 and not hard_failures_in_accepted(accepted) else "fail",
        "acceptance_target": "at least 80 accepted train-only seed proposals",
        "cluster_counts": dict(new_cluster_counts),
        "metadata_collision_count": sum(1 for row in metadata_rows if row["count"] > 1),
    }
    write_json(out_dir / "seed_proposal_gate_report.json", report)
    return report


def evaluate_proposal(
    proposal: dict[str, Any],
    existing: list[dict[str, Any]],
    train: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    known_rule_ids: set[str],
    v1_rows: list[dict[str, Any]],
    new_cluster_counts: Counter[str],
) -> dict[str, Any]:
    seed_id = proposal.get("seed_id", "<missing>")
    reasons: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED_PROPOSAL_FIELDS - set(proposal))
    if missing:
        reasons.append(f"missing required fields: {missing}")
    if proposal.get("split") != "train":
        reasons.append("proposal split must be train")
    for field in LIST_FIELDS:
        if field in proposal and not isinstance(proposal.get(field), list):
            reasons.append(f"{field} must be a list")
    if known_rule_ids:
        unknown_rules = sorted(set(proposal.get("source_rule_ids", [])) - known_rule_ids)
        if unknown_rules:
            warnings.append(f"proposed new source_rule_ids need source-rule manifest review: {unknown_rules}")
    source_basis = proposal.get("source_basis", [])
    if not source_basis:
        reasons.append("source_basis is required")
    if any(str(item).strip().lower() in {"common knowledge", "general knowledge", "internet", "news"} for item in source_basis):
        reasons.append("source_basis is too vague")
    if not proposal.get("source_grounding_need") or not proposal.get("allowed_safe_guidance") or not proposal.get("safe_action_boundary"):
        reasons.append("source grounding boundary fields must be non-empty")
    novelty = " ".join([proposal.get("novelty_rationale", ""), proposal.get("why_not_dev_final_overlap", "")])
    if any(pattern.search(novelty) for pattern in SURFACE_NOVELTY_PATTERNS):
        reasons.append("novelty rationale appears surface-only")
    operational_hits = operational_fact_hits(proposal)
    if operational_hits:
        reasons.append(f"invented operational fact risk: {sorted(operational_hits)}")
    real_incident_check = normalize_text(proposal.get("real_incident_check", ""))
    if not any(token in real_incident_check for token in ["generic", "fictional", "not specific", "source bounded"]):
        reasons.append("real_incident_check must be generic_or_fictionalized, not_specific, or source_bounded")
    cluster_key = proposal_cluster_key(proposal)
    if new_cluster_counts[cluster_key] >= 5:
        reasons.append("new proposal expands an already full new-seed cluster")
    neighbors = nearest_neighbors(proposal, existing, accepted)
    protected_neighbors = [item for item in neighbors if item["neighbor_split"] in {"dev", "final_eval"}]
    top3 = neighbors[:3]
    if protected_neighbors and protected_neighbors[0]["overall_similarity"] >= 0.72:
        reasons.append("closest dev/final neighbor exceeds protected similarity threshold")
    if sum(1 for item in top3 if item["neighbor_split"] in {"dev", "final_eval"}) >= 2:
        reasons.append("top 3 neighbors include 2 or more protected dev/final seeds")
    if any(item["exact_scenario_match"] for item in neighbors[:5]):
        reasons.append("exact normalized scenario match")
    if same_decision_as_protected(proposal, protected):
        reasons.append("same core decision problem or pressure/action pattern as protected dev/final")
    if accepted and same_decision_as_protected(proposal, accepted):
        reasons.append("duplicates an already accepted new proposal decision problem")
    if v1_template_hit(proposal, v1_rows):
        warnings.append("similar to a known v1_600 generated template")
    if protected_neighbors and protected_neighbors[0]["overall_similarity"] >= 0.62:
        warnings.append("protected neighbor requires review")
    neighbor_report = [
        {
            "proposal_seed_id": seed_id,
            **item,
        }
        for item in neighbors[:5]
    ]
    status = "rejected" if reasons else "review" if warnings else "accepted"
    return {
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "neighbors": neighbor_report,
        "cluster_key": cluster_key,
    }


def operational_fact_hits(seed: dict[str, Any]) -> set[str]:
    text = (
        seed_text(seed)
        + " "
        + seed.get("source_grounding_need", "")
        + " "
        + " ".join(seed.get("source_basis", []))
    )
    hits = set()
    for name, pattern in OPERATIONAL_FACT_PATTERNS.items():
        if pattern.search(text):
            hits.add(name)
    return hits


def proposal_cluster_key(seed: dict[str, Any]) -> str:
    return "|".join(
        [
            normalize_text(seed.get("primary_hazard", "")),
            normalize_text(seed.get("core_decision_problem", "")),
            normalize_text(seed.get("answer_archetype", "")),
            normalize_text(seed.get("user_pressure_type", "")),
        ]
    )


def nearest_neighbors(proposal: dict[str, Any], existing: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_text = seed_text(proposal)
    candidate_scenario = scenario_text(proposal)
    candidate_decision = decision_signature(proposal)
    rows = []
    for neighbor in [*existing, *accepted]:
        text_similarity = token_jaccard(candidate_text, seed_text(neighbor))
        scenario_similarity = token_jaccard(candidate_scenario, scenario_text(neighbor))
        decision_similarity = token_jaccard(candidate_decision, decision_signature(neighbor))
        rule_overlap = overlap_ratio(proposal.get("source_rule_ids", []), neighbor.get("source_rule_ids", []))
        same_hazard = proposal.get("primary_hazard") == neighbor.get("primary_hazard")
        overall = max(text_similarity, scenario_similarity, decision_similarity * 0.95, rule_overlap * 0.55 if same_hazard else 0)
        rows.append(
            {
                "neighbor_seed_id": neighbor.get("seed_id", ""),
                "neighbor_split": neighbor.get("split", "new_train"),
                "text_similarity": round(text_similarity, 3),
                "scenario_similarity": round(scenario_similarity, 3),
                "decision_similarity": round(decision_similarity, 3),
                "rule_overlap": round(rule_overlap, 3),
                "same_hazard": same_hazard,
                "overall_similarity": round(overall, 3),
                "exact_scenario_match": normalize_text(candidate_scenario) == normalize_text(scenario_text(neighbor)),
            }
        )
    return sorted(rows, key=lambda item: item["overall_similarity"], reverse=True)


def same_decision_as_protected(proposal: dict[str, Any], protected: list[dict[str, Any]]) -> bool:
    proposal_decision = normalize_text(proposal.get("core_decision_problem", ""))
    proposal_pressure = normalize_text(proposal.get("user_pressure_type", ""))
    proposal_answer = normalize_text(proposal.get("answer_archetype", ""))
    proposal_unsafe = normalize_text(proposal.get("unsafe_assumption_type", proposal.get("unsafe_assumption", "")))
    for seed in protected:
        seed_decision = normalize_text(seed.get("core_decision_problem", seed.get("unsafe_assumption", "")))
        seed_pressure = normalize_text(seed.get("user_pressure_type", seed.get("user_pressure", "")))
        seed_answer = normalize_text(seed.get("answer_archetype", seed.get("answer_style", "")))
        seed_unsafe = normalize_text(seed.get("unsafe_assumption_type", seed.get("unsafe_assumption", "")))
        if token_jaccard(proposal_decision, seed_decision) >= 0.78:
            return True
        if (
            token_jaccard(proposal_unsafe, seed_unsafe) >= 0.72
            and token_jaccard(proposal_pressure, seed_pressure) >= 0.55
            and token_jaccard(proposal_answer, seed_answer) >= 0.55
        ):
            return True
    return False


def v1_template_hit(proposal: dict[str, Any], v1_rows: list[dict[str, Any]]) -> bool:
    if not v1_rows:
        return False
    proposal_norm = normalize_text(seed.get("answer_archetype", "") if (seed := proposal) else "")
    proposal_scenario = normalize_text(scenario_text(proposal))
    for row in v1_rows[:1200]:
        generated = normalize_text(" ".join([row.get("prompt", ""), row.get("target_response", "")]))
        if proposal_norm and proposal_norm in generated and token_jaccard(proposal_scenario, generated) >= 0.45:
            return True
    return False


def metadata_collision_rows(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for proposal in proposals:
        buckets[proposal_cluster_key(proposal)].append(proposal.get("seed_id", "<missing>"))
    return [
        {"cluster_key": key, "count": len(seed_ids), "seed_ids": "|".join(seed_ids)}
        for key, seed_ids in sorted(buckets.items())
    ]


def hard_failures_in_accepted(accepted: list[dict[str, Any]]) -> bool:
    return any(row.get("gate_reasons") for row in accepted)


def humanize_rule_id(rule_id: str) -> str:
    return " ".join(part for part in rule_id.replace("-", "_").split("_") if part)


def rule_manifest_additions(
    proposals: list[dict[str, Any]],
    existing_rules: list[dict[str, Any]],
    *,
    review_status: str = "pending_main_review",
) -> list[dict[str, Any]]:
    known = {row["rule_id"] for row in existing_rules}
    additions: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        for rule_id in proposal.get("source_rule_ids", []):
            if rule_id in known or rule_id in additions:
                continue
            source_basis = proposal.get("source_basis", [])
            source_ids = proposal.get("source_ids", [])
            must_say = proposal.get("must_say", [])
            basis = "; ".join(str(item) for item in source_basis[:2]) if isinstance(source_basis, list) else str(source_basis)
            behavior = "; ".join(str(item) for item in must_say[:2]) if isinstance(must_say, list) else str(must_say)
            additions[rule_id] = {
                "rule_id": rule_id,
                "derived_rule": f"Train-only expansion rule for {humanize_rule_id(rule_id)}. Source basis: {basis}. Expected behavior: {behavior}.",
                "source_ids": source_ids if isinstance(source_ids, list) else [],
                "jurisdiction_scope": "global public-safety guidance applied to India scenarios unless an India-specific source is listed",
                "review_status": review_status,
            }
    return [additions[key] for key in sorted(additions)]


def append_source_rule_additions(
    base_manifest_path: Path,
    proposals_path: Path,
    out_manifest_path: Path,
    additions_path: Path,
) -> dict[str, Any]:
    base_rules = read_jsonl(base_manifest_path)
    proposals = read_jsonl(proposals_path)
    additions = rule_manifest_additions(proposals, base_rules)
    combined = [*base_rules, *additions]
    write_jsonl(out_manifest_path, combined)
    write_jsonl(additions_path, additions)
    return {
        "base_rule_count": len(base_rules),
        "addition_count": len(additions),
        "combined_rule_count": len(combined),
        "out_manifest_path": str(out_manifest_path),
        "additions_path": str(additions_path),
    }


def reviewer_approved_seed_ids(
    proposals: list[dict[str, Any]],
    *,
    rejected_seed_ids: set[str] | None = None,
) -> set[str]:
    rejected_seed_ids = rejected_seed_ids or set()
    return {
        row["seed_id"]
        for row in proposals
        if row.get("gate_status") == "accepted" and row.get("split") == "train" and row.get("seed_id") not in rejected_seed_ids
    }


def promote_seed_proposals(
    base_seed_path: Path,
    accepted_proposals_path: Path,
    out_seed_path: Path,
    report_path: Path,
    *,
    rejected_seed_ids: set[str] | None = None,
) -> dict[str, Any]:
    base_seeds = read_jsonl(base_seed_path)
    proposals = read_jsonl(accepted_proposals_path)
    rejected_seed_ids = rejected_seed_ids or set()
    approved_ids = reviewer_approved_seed_ids(proposals, rejected_seed_ids=rejected_seed_ids)
    new_seeds = []
    seen = {seed["seed_id"] for seed in base_seeds}
    for proposal in proposals:
        seed_id = proposal.get("seed_id")
        if seed_id not in approved_ids:
            continue
        if seed_id in seen:
            raise ValueError(f"duplicate seed_id during promotion: {seed_id}")
        promoted = {
            key: value
            for key, value in proposal.items()
            if key not in {"gate_status", "gate_reasons", "gate_warnings"}
        }
        promoted["review_status"] = "accepted_train_only_seed"
        promoted["reject_reason"] = ""
        promoted["split"] = "train"
        new_seeds.append(promoted)
        seen.add(seed_id)
    combined = [*base_seeds, *new_seeds]
    split_counts = Counter(seed.get("split", "") for seed in combined)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_seed_count": len(base_seeds),
        "promoted_seed_count": len(new_seeds),
        "combined_seed_count": len(combined),
        "split_counts": dict(split_counts),
        "train_row_capacity_at_5_per_seed": split_counts.get("train", 0) * 5,
        "rejected_by_reviewer_count": len(rejected_seed_ids),
        "rejected_by_reviewer_seed_ids": sorted(rejected_seed_ids),
        "promoted_seed_ids": [seed["seed_id"] for seed in new_seeds],
        "out_seed_path": str(out_seed_path),
    }
    write_jsonl(out_seed_path, combined)
    write_json(report_path, report)
    return report
