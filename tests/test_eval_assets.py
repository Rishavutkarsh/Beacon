from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_gemma_sft_dataset import make_record


def test_frozen_eval_prompt_ids_are_unique_and_sufficient() -> None:
    path = Path("data/eval/sankat_saathi_e2b_smoke.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 20
    ids = [row["prompt_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert sum(bool(row.get("critical")) for row in rows) >= 10
    categories = {row["category"] for row in rows}
    assert {"contaminated_water", "food_floodwater", "diabetes_disruption", "electrical_flood"}.issubset(categories)


def test_gemma_sft_packaging_uses_gemma4_turn_template() -> None:
    row = {
        "example_id": "example_1",
        "user_prompt": "We have cloudy water and one elder has fever.",
        "assistant_response": {
            "risk_level": "high",
            "immediate_action": ["Do not drink untreated cloudy water."],
            "resource_plan": ["Use sealed water first."],
            "unsafe_items": ["cloudy water"],
            "missing_information": ["source of water"],
            "escalation_signs": ["fever"],
            "what_not_to_do": ["do not say definitely safe"],
            "hindi_hinglish": ["Paani ko safe mat maano."],
            "uncertainty_note": "Cannot determine safety from appearance alone.",
        },
        "source_ids": ["cdc_floodwater"],
        "guidance_fact_ids": ["fact_1"],
        "risk_tags": ["water"],
        "language_mix": "hinglish",
    }
    record = make_record(row)
    assert "<|turn>user" in record["prompt"]
    assert "<start_of_turn>" not in record["text"]
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["content"] == row["user_prompt"]
