from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sankat_saathi_dataset.grounding_aware_sft import build_package, load_cards, retrieve_cards, validate_package
from sankat_saathi_dataset.local_grounding_cards import REVIEW_AXES, build_bundle, draft_cards


def out_dir(name: str) -> Path:
    path = Path("test_runs") / "grounding_aware_sft" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_approved_cards(target: Path) -> None:
    reviews = []
    for card in draft_cards():
        for reviewer_id in ["source_reviewer", "safety_reviewer"]:
            review = {
                "card_id": card["card_id"],
                "reviewer_id": reviewer_id,
                "recommendation": "approve",
                "issues": [],
                "notes": "test approval",
            }
            review.update({axis: "pass" for axis in REVIEW_AXES})
            reviews.append(review)
    review_path = target / "grounding_card_reviews.jsonl"
    review_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in reviews) + "\n", encoding="utf-8")
    build_bundle(target, review_path)


def test_local_retriever_finds_precision_food_card() -> None:
    cards_target = out_dir("cards")
    build_approved_cards(cards_target)
    cards = load_cards(cards_target, include_unapproved=True)
    hits = retrieve_cards("half full freezer 24 hours outage", cards, top_k=3)
    assert "precision_food_fridge_freezer_times_v1" in {hit.card_id for hit in hits}


def test_grounding_aware_sft_builds_blocked_candidate() -> None:
    cards_target = out_dir("cards_for_sft")
    package_target = out_dir("package")
    build_approved_cards(cards_target)
    result = build_package(package_target, cards_target)
    assert result.errors == []
    assert result.manifest["training_export_allowed"] is False
    assert result.manifest["langchain_used"] is False
    assert result.manifest["row_count"] == 1000
    assert result.manifest["unique_target_response_count"] >= 250
    assert result.errors == []
    rows = read_jsonl(package_target / "all_rows.jsonl")
    assert len(rows) == 1000
    assert {row["row_family"] for row in rows} >= {
        "retrieval_needed",
        "insufficient_grounding",
        "no_retrieval_needed",
        "negative_contrast",
    }
    assert (package_target / "final_eval.jsonl").exists()
    assert len(read_jsonl(package_target / "final_eval.jsonl")) == 100
    assert (package_target / "research_shadow_rows.jsonl").exists()
    assert all(not row["training_ready"] for row in rows)
    assert all(
        status == "approved"
        for row in rows
        for status in row.get("card_status_at_generation", {}).values()
    )
    assert all(
        not row["grounding_card_ids"]
        for row in rows
        if row["row_family"] == "no_retrieval_needed"
    )
    assert all(
        not row["blocking_reasons"]
        for row in rows
        if not row["grounding_card_ids"]
    )


def test_grounding_aware_validator_rechecks_retrieval() -> None:
    cards_target = out_dir("cards_for_validate")
    package_target = out_dir("package_validate")
    build_approved_cards(cards_target)
    build_package(package_target, cards_target)
    result = validate_package(package_target, cards_target)
    assert result.errors == []


def test_grounding_aware_scripts_allow_blocked() -> None:
    cards_target = out_dir("script_cards")
    package_target = out_dir("script_package")
    build_approved_cards(cards_target)
    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_beacon_grounding_aware_sft.py",
            "--cards-dir",
            str(cards_target),
            "--out-dir",
            str(package_target),
            "--allow-blocked",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    validate = subprocess.run(
        [
            sys.executable,
            "scripts/validate_beacon_grounding_aware_sft.py",
            str(package_target),
            "--cards-dir",
            str(cards_target),
            "--allow-blocked",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
