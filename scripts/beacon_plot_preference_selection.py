from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("reports/beacon_candidate_selection_eval_run018_preference")
SUMMARY = ROOT / "preference_selection_summary.json"


def main() -> None:
    import matplotlib.pyplot as plt

    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    candidates = ["base", "checkpoint-175", "checkpoint-400", "checkpoint-448"]
    labels = ["base", "175", "400", "448"]
    colors = ["#555555", "#4c78a8", "#f58518", "#e45756"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].bar(labels, [data["candidates"][c]["best_after_veto"] for c in candidates], color=colors)
    axes[0].set_title("Best Votes After Safety Veto")
    axes[0].set_ylabel("judge votes")

    axes[1].bar(labels, [data["candidates"][c]["runner_up"] for c in candidates], color=colors)
    axes[1].set_title("Runner-Up Votes")
    axes[1].set_ylabel("judge votes")

    axes[2].bar(labels, [data["candidates"][c]["safety_veto"] for c in candidates], color="#d62728")
    axes[2].set_title("Safety Veto Count")
    axes[2].set_ylabel("flags")

    axes[3].bar(labels, [data["candidates"][c]["unsafe_or_offtarget"] for c in candidates], color="#9467bd")
    axes[3].set_title("Unsafe / Off-Target Count")
    axes[3].set_ylabel("flags")

    fig.tight_layout()
    out = ROOT / "preference_candidate_summary.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
