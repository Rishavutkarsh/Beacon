from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "beacon_low_lr_final_eval"
SUMMARY = REPORT_DIR / "final_eval_comparison_summary.json"
OUT = REPORT_DIR / "final_eval_comparison_summary.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    candidates = ["base", "attention_only_best_dev", "all_linear_best_dev"]
    metrics = ["critical", "major", "safety_veto", "best_safety", "best_preference"]
    totals = data["candidate_totals"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = range(len(candidates))
    width = 0.22
    for idx, metric in enumerate(["critical", "major", "safety_veto"]):
        axes[0].bar([item + (idx - 1) * width for item in x], [totals[c].get(metric, 0) for c in candidates], width, label=metric)
    axes[0].set_title("Safety flags lower is better")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(candidates, rotation=20, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for idx, metric in enumerate(["best_safety", "best_preference"]):
        axes[1].bar([item + (idx - 0.5) * width for item in x], [totals[c].get(metric, 0) for c in candidates], width, label=metric)
    axes[1].set_title("Judge wins higher is better")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(candidates, rotation=20, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"Beacon final_eval: {data['decision']}")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
