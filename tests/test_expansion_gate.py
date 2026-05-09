from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sankat_saathi_dataset.expansion_gate import build_rows, read_jsonl, validate_expansion


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
    assert first["row_id"].startswith("ss_exp_train_")
    assert first["scenario_cluster_id"].startswith("sc_")
    assert first["source_rule_ids"]
    assert first["content_hash"]
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
    ]:
        assert (out_dir / name).exists(), name
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
