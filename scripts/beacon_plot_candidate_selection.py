from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("reports/beacon_candidate_selection_eval_run018")
SUMMARY_PATH = ROOT / "candidate_selection_summary.json"
SFT_STATE_PATH = Path("kaggle_outputs/beacon_sft_full_guide_run018_v1/beacon_sft_full_guide_run018/trainer/checkpoint-448/trainer_state.json")


def main() -> None:
    import matplotlib.pyplot as plt

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    candidates = ["base", "checkpoint-175", "checkpoint-400", "checkpoint-448"]
    labels = ["base", "175", "400", "448"]
    data = summary["candidates"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].bar(labels, [data[c]["avg_score"] for c in candidates], color=["#555", "#4c78a8", "#f58518", "#e45756"])
    axes[0].set_title("Average Judge Score")
    axes[0].set_ylim(0, 5)
    axes[0].set_ylabel("score (1-5)")

    axes[1].bar(labels, [data[c]["first_place"] for c in candidates], color=["#555", "#4c78a8", "#f58518", "#e45756"])
    axes[1].set_title("First-Place Votes")
    axes[1].set_ylabel("votes across 2 judges x 95 rows")

    width = 0.35
    x = range(len(labels))
    axes[2].bar([i - width / 2 for i in x], [data[c]["critical_safety_failures"] for c in candidates], width=width, label="critical", color="#d62728")
    axes[2].bar([i + width / 2 for i in x], [data[c]["major_issues"] for c in candidates], width=width, label="major", color="#ff9896")
    axes[2].set_xticks(list(x), labels)
    axes[2].set_title("Judge Safety/Major Flags")
    axes[2].legend()

    axes[3].bar(labels, [data[c]["generic_template_flags"] for c in candidates], color="#9467bd")
    axes[3].set_title("Generic Template Flags")
    axes[3].set_ylabel("flags")

    fig.tight_layout()
    out = ROOT / "candidate_judge_summary.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    state = json.loads(SFT_STATE_PATH.read_text(encoding="utf-8"))
    logs = state.get("log_history", [])
    train = [(entry["step"], entry["loss"]) for entry in logs if "loss" in entry]
    dev = [(entry["step"], entry["eval_loss"]) for entry in logs if "eval_loss" in entry]
    fig, ax1 = plt.subplots(figsize=(11, 5))
    if train:
        ax1.plot([x for x, _ in train], [y for _, y in train], label="train loss", color="#4c78a8")
    if dev:
        ax1.plot([x for x, _ in dev], [y for _, y in dev], label="dev loss", color="#f58518", marker="o", markersize=3)
    for step in [175, 400, 448]:
        ax1.axvline(step, color="#777", linestyle="--", linewidth=0.8)
        ax1.text(step, ax1.get_ylim()[1], str(step), va="top", ha="center", fontsize=8)
    ax1.set_title("Beacon SFT Loss Curve With Evaluated Checkpoints")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("loss")
    ax1.legend()
    fig.tight_layout()
    out2 = ROOT / "sft_loss_curve_candidates.png"
    fig.savefig(out2, dpi=160)
    plt.close(fig)
    print(out)
    print(out2)


if __name__ == "__main__":
    main()
