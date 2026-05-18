from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_run(name: str, score_dir: Path) -> dict[str, Any]:
    return {
        "name": name,
        "summary": read_json(score_dir / "score_summary.json"),
        "rows": {row["example_id"]: row for row in read_jsonl(score_dir / "scored_predictions.jsonl")},
    }


def compare(base_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    base = next((run for run in runs if run["name"] == base_name), None)
    if base is None:
        raise SystemExit(f"Base run {base_name!r} not found")
    comparable_ids = sorted(set.intersection(*(set(run["rows"]) for run in runs)))
    pairwise: dict[str, Any] = {}
    for run in runs:
        if run["name"] == base_name:
            continue
        wins = losses = ties = unsafe_delta = 0
        by_bucket: dict[str, dict[str, int]] = {}
        for example_id in comparable_ids:
            base_row = base["rows"][example_id]
            row = run["rows"][example_id]
            delta = int(row["is_correct"]) - int(base_row["is_correct"])
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
            unsafe_delta += int(row["selected_unsafe_distractor"]) - int(base_row["selected_unsafe_distractor"])
            bucket = row["eval_bucket"]
            by_bucket.setdefault(bucket, {"wins": 0, "losses": 0, "ties": 0})
            if delta > 0:
                by_bucket[bucket]["wins"] += 1
            elif delta < 0:
                by_bucket[bucket]["losses"] += 1
            else:
                by_bucket[bucket]["ties"] += 1
        pairwise[run["name"]] = {
            "comparable_rows": len(comparable_ids),
            "wins_vs_base": wins,
            "losses_vs_base": losses,
            "ties_vs_base": ties,
            "unsafe_distractor_delta_vs_base": unsafe_delta,
            "by_eval_bucket": by_bucket,
        }
    summaries = {run["name"]: run["summary"]["headline_metrics"] | {"invalid_rate": run["summary"]["invalid_rate"]} for run in runs}
    return {
        "base_run": base_name,
        "run_headline_metrics": summaries,
        "pairwise_vs_base": pairwise,
        "selection_rule": {
            "prefer": "highest heldout_test_accuracy",
            "veto": "unsafe_distractor_selection increase on critical rows or large invalid-rate increase",
            "note": "overall_80_accuracy is mixed diagnostic only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Beacon MCQ score directories.")
    parser.add_argument("--base", required=True, help="Name of the base run, e.g. base")
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "SCORE_DIR"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(name, Path(path)) for name, path in args.run]
    write_json(args.out, compare(args.base, runs))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
