from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sankat_saathi_dataset.expansion_gate import (
    EXPANSION_PROFILES,
    action_sequence,
    build_rows,
    bullet_shape,
    make_row,
    normalize_text,
    read_jsonl,
    validate_artifacts,
    validate_expansion,
    validate_final_eval_isolation,
    validate_output_similarity,
    validate_per_seed_diversity,
    validate_source_claim_support,
)


def scratch_dir(name: str) -> Path:
    out_dir = Path("data/expanded") / name
    shutil.rmtree(out_dir, ignore_errors=True)
    return out_dir


def test_full_target_is_infeasible_under_max_five_seed_cap() -> None:
    out_dir = scratch_dir("_test_expansion_gate_infeasible")
    manifest = build_rows(
        Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"),
        out_dir,
        stage="full",
        profile="v2_1k",
        max_variants_per_seed=5,
    )

    assert manifest["counts"]["train"] == 600
    assert any("train target 1000 exceeds cap capacity 600" in error for error in manifest["feasibility_errors"])
    shutil.rmtree(out_dir, ignore_errors=True)


def test_calibration_build_writes_schema_complete_rows() -> None:
    out_dir = scratch_dir("_test_expansion_gate_calibration")
    manifest = build_rows(Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"), out_dir, stage="calibration", profile="calibration")
    rows = read_jsonl(out_dir / "generated_rows.jsonl")

    assert manifest["counts"]["train"] == 50
    assert rows
    first = rows[0]
    assert first["candidate_id"].startswith("cand_")
    assert first["generation_run_id"].startswith("gen_")
    assert first["review_state"] == "generated"
    assert first["row_id"].startswith("ss_exp_train_")
    assert first["scenario_cluster_id"].startswith("sc_")
    assert first["source_rule_ids"]
    assert first["content_hash"]
    shutil.rmtree(out_dir, ignore_errors=True)


def test_v2_1015_profile_builds_full_capacity_from_expanded_seeds() -> None:
    out_dir = scratch_dir("_test_expansion_gate_v2_1015")
    manifest = build_rows(
        Path("data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl"),
        out_dir,
        stage="full",
        profile="v2_1015",
        rule_manifest_path=Path("data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl"),
    )

    assert manifest["counts"] == {"train": 1015, "dev": 120, "final_eval": 120}
    assert manifest["feasibility_errors"] == []
    assert manifest["seed_snapshot_hash"]
    assert manifest["source_rule_snapshot_hash"]
    rows = read_jsonl(out_dir / "generated_rows.jsonl")
    assert max(sum(1 for row in rows if row["seed_id"] == seed_id and row["split"] == "train") for seed_id in {row["seed_id"] for row in rows if row["split"] == "train"}) == 5
    shutil.rmtree(out_dir, ignore_errors=True)


def test_variant_renderer_produces_semantic_sibling_diversity() -> None:
    seed = next(row for row in read_jsonl(Path("data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl")) if row["split"] == "train")
    rows = [make_row(seed, index, "test", "hash", generation_run_id="gen_test") for index in range(5)]

    assert len({normalize_text(row["prompt"]) for row in rows}) == 5
    assert len({normalize_text(row["target_response"]) for row in rows}) == 5
    assert len({row["renderer_style"] for row in rows}) >= 3
    assert len({bullet_shape(row["target_response"]) for row in rows}) >= 2
    assert len({action_sequence(row["target_response"]) for row in rows}) >= 3
    assert all(row["prompt_template_version"] == "seed_renderer_v2" for row in rows)
    assert all("variant_contract" in row for row in rows)


def test_variant_renderer_keeps_prompt_response_styles_aligned() -> None:
    seed = next(row for row in read_jsonl(Path("data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl")) if row["split"] == "train")
    rows = [make_row(seed, index, "test", "hash", generation_run_id="gen_test") for index in range(5)]
    joined = {row["renderer_style"]: normalize_text(row["prompt"] + " " + row["target_response"]) for row in rows}

    assert any("triage" in text and "handoff" in text for style, text in joined.items() if style == "volunteer_triage_plan")
    assert any("family" in text or "household" in text for style, text in joined.items() if style == "family_resource_plan")
    assert any("hinglish" in text or "pehle" in text for style, text in joined.items() if style == "low_literacy_hinglish")
    assert all("i cannot verify" not in row["target_response"].lower() for row in rows)


def test_renderer_does_not_promote_style_notes_to_safety_actions() -> None:
    seeds = read_jsonl(Path("data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl"))
    problem_seed_ids = {"ss_seed_050", "ss_seed_074", "ss_seed_124"}
    rows = [make_row(seed, 0, "test", "hash", generation_run_id="gen_test") for seed in seeds if seed["seed_id"] in problem_seed_ids]
    joined = "\n".join(row["target_response"].lower() for row in rows)

    assert "assign simple roles" not in joined
    assert "use short simple" not in joined
    assert "scarce-resource fallback" not in joined
    assert "put life safety" not in joined
    assert any("fallen power line" in row["target_response"].lower() or "dry safe location" in row["target_response"].lower() for row in rows)
    assert any("fresh air" in row["target_response"].lower() or "fuel-burning" in row["target_response"].lower() for row in rows)
    assert any("high ground" in row["target_response"].lower() or "verified alternate route" in row["target_response"].lower() for row in rows)


def test_artifact_gate_catches_split_eval_debug_and_duplicate_lines() -> None:
    rows = [
        {
            "row_id": "bad_artifact",
            "prompt": "final-eval competing-pressure variant: keep the locked-eval response self-contained.",
            "target_response": "Pehle flooded basement breaker train 0 scene mein risk ko halka mat lo.\n- Move away.\n1) Move away.",
        }
    ]

    errors, report = validate_artifacts(rows)

    assert errors
    assert report["hits_by_type"]["eval_marker"] >= 1
    assert report["hits_by_type"]["split_debug"] >= 1
    assert report["hits_by_type"]["seed_debug"] >= 1
    assert report["duplicate_response_line_row_count"] == 1


def test_artifact_gate_allows_natural_train_word() -> None:
    rows = [
        {
            "row_id": "rail_context",
            "prompt": "A train passenger sees water near the platform. Give practical safety guidance.",
            "target_response": "Move away from the flooded edge.\nShare only locally verified platform updates.",
        }
    ]

    errors, report = validate_artifacts(rows)

    assert errors == []
    assert report["status"] == "pass"


def test_artifact_gate_catches_run006_mechanical_residue() -> None:
    rows = [
        {
            "row_id": "slot_residue",
            "split": "train",
            "renderer_style": "family_resource_plan",
            "prompt": "How should the family use scarce help for none_required?",
            "target_response": "Verify none_required first.\nOne helper handles the immediate action: move away.",
        },
        {
            "row_id": "target_meta",
            "split": "dev",
            "renderer_style": "visual_uncertainty",
            "prompt": "Give a brief practical answer.",
            "target_response": "Keep the wording compact so the safety behavior is easy to follow.\nMove away from the hazard.",
        },
        {
            "row_id": "triage_template",
            "split": "train",
            "renderer_style": "volunteer_triage_plan",
            "prompt": "A volunteer asks what to do.",
            "target_response": "Triage starts when volunteers guide around urban flood, not reassurance.\nImmediate danger queue: move back.",
        },
        {
            "row_id": "hinglish_template",
            "split": "train",
            "renderer_style": "low_literacy_hinglish",
            "prompt": "Make a simple note.",
            "target_response": "Pehle is relief camp setting mein risk wali risk ko halka mat lo: do not assume safe.",
        },
        {
            "row_id": "checklist_template",
            "split": "final_eval",
            "renderer_style": "first_10_minutes_checklist",
            "prompt": "In the next few minutes, no-photo/no-live-status certainty both matter.",
            "target_response": "When urban flood discard food is the issue, move people away.",
        },
        {
            "row_id": "live_template",
            "split": "final_eval",
            "renderer_style": "live_fact_refusal",
            "prompt": "Someone wants current status.",
            "target_response": "Treat live status as unverified while deciding about road status.",
        },
    ]

    errors, report = validate_artifacts(rows)

    assert errors
    assert report["hits_by_type"]["slot_residue"] >= 1
    assert report["hits_by_type"]["target_instruction"] >= 1
    assert report["hits_by_type"]["triage_starts_when_volunteers"] == 1
    assert report["hits_by_type"]["pehle_is_setting"] == 1
    assert report["hits_by_type"]["when_is_the_issue"] == 1
    assert report["hits_by_type"]["treat_live_status_unverified"] == 1


def test_v2_1015_scratch_build_passes_key_deterministic_renderer_gates() -> None:
    out_dir = scratch_dir("_test_expansion_gate_v2_1015_renderer_preflight")
    build_rows(
        Path("data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl"),
        out_dir,
        stage="full",
        profile="v2_1015",
        rule_manifest_path=Path("data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl"),
    )
    result = validate_expansion(
        out_dir,
        Path("data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl"),
        profile="v2_1015",
    )
    blocking = [
        error
        for error in result.errors
        if "review records are not calibrated" not in error
        and "review calibration status must pass" not in error
        and "review calibration has not been run" not in error
        and "review calibration has not passed" not in error
        and "review canary failure catch rate" not in error
        and "review calibration canary catch rate" not in error
        and "review agreement rate" not in error
        and "review calibration agreement rate" not in error
        and "review calibration agreement below" not in error
        and "review calibration report is missing" not in error
        and "placeholder" not in error
    ]

    assert not blocking
    source_support = result.reports["source_claim_support_report"]
    review_sampling = json.loads((out_dir / "review_sampling_manifest.json").read_text(encoding="utf-8"))
    assert review_sampling["source_support_warnings"] == source_support["review_sentence_count"]
    assert review_sampling["source_support_warning_rows"] == source_support["review_row_count"]
    shutil.rmtree(out_dir, ignore_errors=True)


def test_v2_1015_profile_requires_exact_train_count() -> None:
    bounds = EXPANSION_PROFILES["v2_1015"]["accepted_count_ranges"]["train"]

    assert bounds == [1015, 1015]
    assert not (bounds[0] <= 1000 <= bounds[1])
    assert not (bounds[0] <= 1016 <= bounds[1])


def test_v2_1015_seed_snapshot_assertion_fails_on_old_seed_bank() -> None:
    out_dir = scratch_dir("_test_expansion_gate_v2_1015_seed_counts")
    manifest = build_rows(
        Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"),
        out_dir,
        stage="full",
        profile="v2_1015",
    )

    assert any("seed count" in error for error in manifest["feasibility_errors"])
    shutil.rmtree(out_dir, ignore_errors=True)


def test_validation_writes_required_audit_artifacts_even_when_gate_fails() -> None:
    out_dir = scratch_dir("_test_expansion_gate_validation")
    build_rows(Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"), out_dir, stage="calibration", profile="calibration")

    result = validate_expansion(out_dir, Path("data/seed_cards/source_rule_manifest_v1.jsonl"), profile="calibration", fail_on_count=False)

    assert result.status == "fail"
    assert any("subagent review records are not calibrated" in error for error in result.errors)
    for name in [
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
        "final_eval_isolation_report.json",
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
    ]:
        assert (out_dir / name).exists(), name
    ledger = read_jsonl(out_dir / "row_failure_ledger.jsonl")
    assert ledger
    assert {"gate_layer", "blocking", "repair_owner", "repair_allowed_inputs"} <= set(ledger[0])
    shutil.rmtree(out_dir, ignore_errors=True)


def test_source_claim_support_report_flags_unsupported_claim() -> None:
    out_dir = scratch_dir("_test_expansion_gate_source_support")
    row = read_jsonl(Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"))[0]
    build_rows(Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"), out_dir, stage="calibration", profile="calibration")
    rows = read_jsonl(out_dir / "generated_rows.jsonl")[:1]
    rows[0]["target_response"] = "Drive through floodwater because it is safe if the engine is high."
    report_errors, report = validate_source_claim_support(rows, Path("data/seed_cards/source_rule_manifest_v1.jsonl"), out_dir / "source_claim_support_report.csv")

    assert report_errors == []
    assert report["warning_count"] >= 1
    assert (out_dir / "source_claim_support_report.csv").exists()
    assert row
    shutil.rmtree(out_dir, ignore_errors=True)


def test_per_seed_diversity_gate_catches_duplicate_variants() -> None:
    out_dir = scratch_dir("_test_expansion_gate_seed_diversity")
    build_rows(Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"), out_dir, stage="calibration", profile="calibration")
    rows = read_jsonl(out_dir / "generated_rows.jsonl")
    seed_rows = [row for row in rows if row["seed_id"] == rows[0]["seed_id"]]
    duplicated = []
    for index in range(5):
        clone = dict(seed_rows[0])
        clone["row_id"] = f"{clone['row_id']}_dup_{index}"
        clone["candidate_id"] = f"cand_dup_{index}"
        duplicated.append(clone)
    errors, report = validate_per_seed_diversity(duplicated)

    assert errors
    assert report["failing_seed_count"] == 1
    shutil.rmtree(out_dir, ignore_errors=True)


def test_output_similarity_gate_catches_train_final_answer_duplicate() -> None:
    train = {
        "row_id": "train_row",
        "split": "train",
        "prompt": "Prompt A",
        "target_response": "Stop the unsafe action. Move away from danger. Escalate if symptoms worsen.",
    }
    final = {
        "row_id": "final_row",
        "split": "final_eval",
        "prompt": "Prompt B",
        "target_response": "Stop the unsafe action. Move away from danger. Escalate if symptoms worsen.",
    }
    errors, report = validate_output_similarity([train, final])

    assert errors
    assert report["exact_train_final_answer_overlap"]


def test_final_eval_isolation_fails_on_train_reference() -> None:
    rows = [
        {"row_id": "train_row", "split": "train", "parent_row_id": ""},
        {"row_id": "final_row", "split": "final_eval", "parent_row_id": "", "generation_source_refs": ["train_row"]},
    ]
    errors, report = validate_final_eval_isolation(rows, {"config": {"final_eval_isolation": "strict"}})

    assert errors
    assert report["violation_count"] == 1


def test_final_eval_isolation_fails_on_train_row_referencing_final_eval() -> None:
    rows = [
        {"row_id": "final_row", "split": "final_eval", "parent_row_id": ""},
        {"row_id": "train_row", "split": "train", "parent_row_id": "", "generation_source_refs": ["final_row"]},
    ]
    errors, report = validate_final_eval_isolation(rows, {"config": {"final_eval_isolation": "strict"}})

    assert errors
    assert report["violation_count"] == 1


def test_final_eval_isolation_fails_on_repair_prompt_lineage_leak() -> None:
    rows = [
        {"row_id": "final_row", "split": "final_eval", "parent_row_id": ""},
        {"row_id": "train_row", "split": "train", "parent_row_id": ""},
    ]
    repair_lineage = [
        {
            "repair_id": "repair_001",
            "new_row_id": "train_row",
            "target_split": "train",
            "input_row_ids": ["final_row"],
            "uses_exact_final_eval_text": True,
        }
    ]
    errors, report = validate_final_eval_isolation(rows, {"config": {"final_eval_isolation": "strict"}}, repair_lineage)

    assert errors
    assert report["violation_count"] == 1


def test_build_expansion_refuses_non_empty_immutable_run_dir() -> None:
    out_dir = scratch_dir("_test_expansion_gate_fail_if_exists")
    out_dir.mkdir(parents=True)
    (out_dir / "marker.txt").write_text("existing", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_expansion.py",
            "--profile",
            "calibration",
            "--out-dir",
            str(out_dir),
            "--fail-if-exists",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    shutil.rmtree(out_dir, ignore_errors=True)


def test_build_expansion_cli_exits_nonzero_for_infeasible_full_target() -> None:
    out_dir = scratch_dir("_test_expansion_gate_cli")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_expansion.py",
            "--profile",
            "v2_1k",
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["counts"]["train"] == 600
    assert payload["feasibility_errors"]
    shutil.rmtree(out_dir, ignore_errors=True)


def test_v1_600_profile_builds_train_dev_final_without_feasibility_errors() -> None:
    out_dir = scratch_dir("_test_expansion_gate_v1_600")
    manifest = build_rows(
        Path("data/seed_cards/sankat_saathi_seed_cards_v1.jsonl"),
        out_dir,
        stage="full",
        profile="v1_600",
    )

    assert manifest["counts"] == {"train": 600, "dev": 120, "final_eval": 120}
    assert manifest["feasibility_errors"] == []
    rows = read_jsonl(out_dir / "generated_rows.jsonl")
    assert {row["split"] for row in rows} == {"train", "dev", "final_eval"}
    assert max(sum(1 for row in rows if row["seed_id"] == seed_id and row["split"] == "train") for seed_id in {row["seed_id"] for row in rows if row["split"] == "train"}) == 5
    assert max(sum(1 for row in rows if row["seed_id"] == seed_id and row["split"] == "dev") for seed_id in {row["seed_id"] for row in rows if row["split"] == "dev"}) == 3
    shutil.rmtree(out_dir, ignore_errors=True)
