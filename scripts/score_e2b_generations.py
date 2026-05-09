from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.eval_rules import aggregate_scores, score_generation


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generation_by_prompt(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not prompt_id:
            raise SystemExit(f"generation row missing prompt_id: {row}")
        result[str(prompt_id)] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Sankat Saathi E2B generations against frozen eval prompts.")
    parser.add_argument("--prompts", default="data/eval/sankat_saathi_e2b_smoke.jsonl")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    prompts = read_jsonl(Path(args.prompts))
    generations = generation_by_prompt(read_jsonl(Path(args.generations)))
    missing = [prompt["prompt_id"] for prompt in prompts if prompt["prompt_id"] not in generations]
    if missing:
        raise SystemExit(f"missing generations for {len(missing)} prompt(s): {missing[:10]}")
    scores = [score_generation(prompt, generations[prompt["prompt_id"]]) for prompt in prompts]
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "judge_results.jsonl", [score.to_record() for score in scores])
    metrics = aggregate_scores(scores)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
