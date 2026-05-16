from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from sankat_saathi_dataset.assistant_sft import build_rows, validate_bundle, validate_rows, write_bundle


def scratch_dir() -> Path:
    path = Path("data") / "_test_tmp" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def test_beacon_assistant_sft_rows_are_review_first_and_assistant_shaped() -> None:
    rows = build_rows()
    errors, report = validate_rows(rows, Path("data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl"))

    assert errors == []
    assert report["row_count"] == 12
    assert report["counts"]["split"] == {"train": 8, "dev": 2, "final_eval": 2}
    assert rows[0]["training_ready"] is False
    assert rows[0]["review_status"] == "pending"
    assert [message["role"] for message in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert "risk_level:" not in rows[0]["target_response"]
    assert "As an AI" not in rows[0]["target_response"]


def test_beacon_assistant_sft_bundle_validates_candidate_and_blocks_export() -> None:
    out_dir = scratch_dir()
    try:
        manifest = write_bundle(out_dir)
        errors, report = validate_bundle(out_dir, stage="candidate")

        assert errors == []
        assert manifest["training_export_allowed"] is False
        assert manifest["stage"] == "sft_draft_package_for_review"
        assert report["review_queue_count"] == 12
        assert (out_dir / "review_queue.csv").exists()
        assert (out_dir / "train.jsonl").exists()
        assert (out_dir / "dev.jsonl").exists()
        assert (out_dir / "final_eval.jsonl").exists()
        assert (out_dir / "source_rule_map.jsonl").exists()
        assert (out_dir / "dataset_design_note.md").exists()
        assert (out_dir / "review_report.json").exists()

        export_errors, _ = validate_bundle(out_dir, stage="export")
        assert any("training_export_allowed=true" in error for error in export_errors)
        assert any("all review checks approved" in error for error in export_errors)
    finally:
        cleanup(out_dir)


def test_beacon_assistant_sft_gate_rejects_live_fact_claim() -> None:
    rows = build_rows()
    rows[0] = dict(rows[0])
    rows[0]["target_response"] = "The bridge is safe and open. Call 123456 now."
    rows[0]["messages"] = [
        rows[0]["messages"][0],
        rows[0]["messages"][1],
        {"role": "assistant", "content": rows[0]["target_response"]},
    ]

    errors, _ = validate_rows(rows, Path("data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl"))

    assert any("fabricated_live_fact" in error for error in errors)


def test_beacon_assistant_sft_script_outputs_review_bundle() -> None:
    out_dir = scratch_dir()
    try:
        result = subprocess.run(
            [sys.executable, "scripts/build_beacon_assistant_sft.py", "--out-dir", str(out_dir)],
            text=True,
            encoding="utf-8",
            capture_output=True,
        )

        assert result.returncode == 0, result.stderr + result.stdout
        payload = json.loads(result.stdout)
        assert payload["errors"] == []
        assert (out_dir / "dataset_manifest.json").exists()
        assert (out_dir / "validation_report.json").exists()
        assert (out_dir / "train.jsonl").exists()
    finally:
        cleanup(out_dir)
