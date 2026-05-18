from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from sankat_saathi_dataset.local_grounding_research import (
    REQUIRED_HAZARD_FAMILIES,
    build_research_pack,
    validate_candidates,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def out_dir(name: str) -> Path:
    path = Path("test_runs") / "source_research" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_source_research_pack_builds_reviewable_outputs() -> None:
    target = out_dir("pack")
    result = build_research_pack(target)
    assert result.errors == []
    assert (target / "candidate_sources.jsonl").exists()
    assert (target / "downloaded_document_cards.jsonl").exists()
    assert (target / "rejected_sources.jsonl").exists()
    assert (target / "coverage_matrix.csv").exists()
    assert (target / "research_report.md").exists()
    assert (target / "manifest.json").exists()

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "source_research_only_no_grounding_cards"
    assert manifest["candidate_source_count"] >= 31
    assert manifest["accepted_document_count"] >= 20
    assert manifest["coverage_gap_count"] == 0


def test_source_research_required_hazard_families_are_covered() -> None:
    target = out_dir("coverage")
    build_research_pack(target)
    with (target / "coverage_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["hazard_family"] for row in rows} == set(REQUIRED_HAZARD_FAMILIES)
    assert all(row["status"] == "covered" for row in rows)
    assert any(int(row["india_source_count"]) > 0 for row in rows)


def test_source_research_validator_rejects_live_status_and_dose_claims() -> None:
    target = out_dir("validator")
    build_research_pack(target)
    candidates = read_jsonl(target / "candidate_sources.jsonl")
    bad = dict(candidates[0])
    bad["document_id"] = "bad_live_status"
    bad["title"] = "Bridge is open and take one tablet"
    errors = validate_candidates([*candidates, bad]).errors
    assert any("bridge is open" in error for error in errors)
    assert any("take one tablet" in error for error in errors)


def test_source_research_scripts_build_and_validate() -> None:
    target = out_dir("scripts")
    build = subprocess.run(
        [sys.executable, "scripts/build_beacon_source_research.py", "--out-dir", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    validate = subprocess.run(
        [sys.executable, "scripts/validate_beacon_source_research.py", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
