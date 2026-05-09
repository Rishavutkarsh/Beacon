from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.seed_expansion import append_source_rule_additions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source-rule manifest with train-only seed proposal additions.")
    parser.add_argument("--base-manifest", default="data/seed_cards/source_rule_manifest_v1.jsonl")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--out-manifest", default="data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl")
    parser.add_argument("--additions", default="data/seed_cards/source_rule_manifest_v2_train_seed_additions.jsonl")
    args = parser.parse_args()

    report = append_source_rule_additions(
        ROOT / args.base_manifest,
        ROOT / args.proposals,
        ROOT / args.out_manifest,
        ROOT / args.additions,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
