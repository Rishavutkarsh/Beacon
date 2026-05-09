from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.seed_expansion import DEFAULT_DIVERSITY_REJECTS, promote_seed_proposals


def parse_rejects(raw: str) -> set[str]:
    if raw == "default_diversity_rejects":
        return set(DEFAULT_DIVERSITY_REJECTS)
    if raw in {"", "none", "None"}:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote reviewer-approved train-only seed proposals into a combined seed bank.")
    parser.add_argument("--base-seeds", default="data/seed_cards/sankat_saathi_seed_cards_v1.jsonl")
    parser.add_argument("--accepted-proposals", required=True)
    parser.add_argument("--out-seeds", default="data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl")
    parser.add_argument("--report", default="data/seed_cards/expansion_v2_train_only/promotion_report.json")
    parser.add_argument(
        "--reject-seed-ids",
        default="default_diversity_rejects",
        help="Comma-separated reviewer reject seed IDs, or default_diversity_rejects.",
    )
    args = parser.parse_args()

    report = promote_seed_proposals(
        ROOT / args.base_seeds,
        ROOT / args.accepted_proposals,
        ROOT / args.out_seeds,
        ROOT / args.report,
        rejected_seed_ids=parse_rejects(args.reject_seed_ids),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
