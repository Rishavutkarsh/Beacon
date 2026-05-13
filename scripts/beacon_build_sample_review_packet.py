from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("reports/beacon_candidate_selection_eval_run018_preference")
SUMMARY = ROOT / "preference_selection_summary.json"
BUNDLE = ROOT / "preference_judge_bundle.jsonl"
OLD_MAP = Path("reports/beacon_candidate_selection_eval_run018/candidate_label_map.jsonl")
NEW_MAP = ROOT / "label_map.jsonl"
OUT = ROOT / "sample_review_packet.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    bundle_by_id = {row["example_id"]: row for row in read_jsonl(BUNDLE)}
    old_map = {row["example_id"]: row["label_map"] for row in read_jsonl(OLD_MAP)}
    new_map = {}
    for row in read_jsonl(NEW_MAP):
        if "label_map_to_original_bundle_label" in row:
            new_map[row["example_id"]] = {
                pref: old_map[row["example_id"]][old]
                for pref, old in row["label_map_to_original_bundle_label"].items()
            }
    records_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in summary["row_records"]:
        records_by_id[record["example_id"]].append(record)

    selected_ids: list[str] = []

    def add(condition, limit: int) -> None:
        added = 0
        for example_id, records in records_by_id.items():
            if example_id in selected_ids:
                continue
            if condition(example_id, records):
                selected_ids.append(example_id)
                added += 1
                if added >= limit:
                    return

    add(lambda _id, rs: any("base" in r["selected_candidates_after_veto"] for r in rs), 5)
    add(lambda _id, rs: any("checkpoint-175" in r["selected_candidates_after_veto"] for r in rs), 5)
    add(lambda _id, rs: any("checkpoint-400" in r["selected_candidates_after_veto"] or "checkpoint-448" in r["selected_candidates_after_veto"] for r in rs), 5)
    add(lambda _id, rs: len({tuple(r["selected_candidates_after_veto"]) for r in rs}) > 1, 5)
    add(lambda _id, rs: any(r["veto_candidates"] for r in rs), 6)
    add(lambda example_id, _rs: bundle_by_id[example_id].get("risk_level") == "high", 4)

    rows: list[dict[str, Any]] = []
    for example_id in selected_ids[:26]:
        bundle = bundle_by_id[example_id]
        label_to_candidate = new_map[example_id]
        named_answers = []
        for candidate in bundle["candidate_answers"]:
            named_answers.append(
                {
                    "candidate": label_to_candidate[candidate["label"]],
                    "response": candidate["response"],
                    "response_char_count": candidate["response_char_count"],
                }
            )
        rows.append(
            {
                "example_id": example_id,
                "hazard_domain": bundle.get("hazard_domain"),
                "risk_level": bundle.get("risk_level"),
                "hazard_bucket": bundle.get("hazard_bucket"),
                "prompt": bundle["prompt"],
                "expected_behavior_notes": bundle["expected_behavior_notes"],
                "judge_row_records": records_by_id[example_id],
                "named_candidate_answers": named_answers,
            }
        )
    write_jsonl(OUT, rows)
    print(f"wrote {OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
