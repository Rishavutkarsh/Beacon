from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sankat_saathi_dataset.expansion_gate import (
    EXPANSION_PROFILES,
    build_rows,
    read_jsonl,
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
