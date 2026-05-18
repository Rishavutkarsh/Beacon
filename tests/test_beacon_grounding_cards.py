from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sankat_saathi_dataset.local_grounding_cards import REVIEW_AXES, build_bundle, draft_cards, validate_cards
from sankat_saathi_dataset.local_grounding_research import REQUIRED_HAZARD_FAMILIES


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def out_dir(name: str) -> Path:
    path = Path("test_runs") / "grounding_cards" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def approval_reviews(cards: list[dict], reviewer_ids: tuple[str, str] = ("source_reviewer", "safety_reviewer")) -> list[dict]:
    reviews = []
    for row in cards:
        for reviewer_id in reviewer_ids:
            review = {
                "card_id": row["card_id"],
                "reviewer_id": reviewer_id,
                "recommendation": "approve",
                "issues": [],
                "notes": "test approval",
            }
            review.update({axis: "pass" for axis in REVIEW_AXES})
            reviews.append(review)
    return reviews


def test_grounding_card_draft_bundle_builds() -> None:
    target = out_dir("draft")
    result = build_bundle(target)
    assert result.manifest["draft_card_count"] == 56
    assert result.manifest["approved_card_count"] == 0
    assert (target / "draft_grounding_cards.jsonl").exists()
    assert (target / "approved_grounding_cards.jsonl").exists()
    assert (target / "coverage_matrix.csv").exists()


def test_grounding_cards_approve_with_two_reviews() -> None:
    target = out_dir("approved")
    cards = draft_cards()
    reviews = approval_reviews(cards)
    review_path = target / "grounding_card_reviews.jsonl"
    target.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in reviews) + "\n", encoding="utf-8")
    result = build_bundle(target, review_path)
    assert result.errors == []
    assert result.manifest["approved_card_count"] == 56
    approved = read_jsonl(target / "approved_grounding_cards.jsonl")
    assert len(approved) == 56
    assert set(row["hazard_family"] for row in approved) == set(REQUIRED_HAZARD_FAMILIES)


def test_grounding_card_validator_rejects_single_review_and_unsupported() -> None:
    cards = draft_cards()
    for row in cards:
        row["status"] = "approved"
    reviews = approval_reviews(cards, ("only_reviewer",))
    reviews[0]["source_support"] = "unsupported"
    result = validate_cards(cards, reviews)
    assert any("requires two passing reviewer approvals" in error for error in result.errors)
    assert any("unsupported source review" in error for error in result.errors)


def test_grounding_card_validator_rejects_live_and_medicine_claims() -> None:
    cards = draft_cards()
    bad = dict(cards[0])
    bad["card_id"] = "bad_claims"
    bad["answer_template"] = dict(bad["answer_template"])
    bad["answer_template"]["core_guidance"] = "The bridge is open and take one tablet now."
    result = validate_cards([*cards, bad], [])
    assert any("bridge is open" in error for error in result.errors)
    assert any("take one tablet" in error for error in result.errors)


def test_grounding_card_scripts_build_and_validate_draft() -> None:
    target = out_dir("scripts")
    build = subprocess.run(
        [sys.executable, "scripts/build_beacon_grounding_cards.py", "--out-dir", str(target), "--allow-draft-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    validate = subprocess.run(
        [sys.executable, "scripts/validate_beacon_grounding_cards.py", str(target), "--allow-draft-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
