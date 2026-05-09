from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.seed_expansion import validate_seed_proposals


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate train-only seed proposals against existing Sankat Saathi seeds and locked eval.")
    parser.add_argument("proposal_jsonl")
    parser.add_argument("--seed-cards", default="data/seed_cards/sankat_saathi_seed_cards_v1.jsonl")
    parser.add_argument("--out-dir", default="data/seed_cards/expansion_v2_train_only/gate")
    parser.add_argument("--rule-manifest", default="data/seed_cards/source_rule_manifest_v1.jsonl")
    parser.add_argument("--v1-rows", default="data/expanded/sankat_expansion_v1_600/generated_rows.jsonl")
    args = parser.parse_args()

    report = validate_seed_proposals(
        ROOT / args.proposal_jsonl,
        ROOT / args.seed_cards,
        ROOT / args.out_dir,
        rule_manifest=ROOT / args.rule_manifest,
        v1_rows_path=ROOT / args.v1_rows,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
