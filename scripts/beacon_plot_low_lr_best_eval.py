from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "beacon_low_lr_best_dev_eval"
SUMMARY = REPORT_DIR / "low_lr_best_selection_summary.json"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    totals = data["candidate_totals"]
    candidates = ["base", "attention_only_best_dev", "all_linear_best_dev", "old_high_lr_checkpoint_175"]
    labels = ["base", "attn-only", "attn+MLP", "old-175"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].bar(labels, [totals[c].get("critical", 0) for c in candidates], color="#b22222")
    axes[0].set_title("Critical Safety Flags")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, [totals[c].get("major", 0) for c in candidates], color="#ff8c00")
    axes[1].set_title("Major Safety / Boundary Flags")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(labels, [totals[c].get("safety_veto", 0) for c in candidates], color="#8b0000")
    axes[2].set_title("Usefulness Judge Safety Vetoes")
    axes[2].tick_params(axis="x", rotation=15)

    axes[3].bar(labels, [totals[c].get("best_preference", 0) for c in candidates], color="#2e8b57")
    axes[3].set_title("Best Preference Votes")
    axes[3].tick_params(axis="x", rotation=15)

    fig.suptitle(f"Beacon low-LR dev selection: {data['decision']}")
    fig.tight_layout()
    out = REPORT_DIR / "low_lr_best_selection_summary.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
