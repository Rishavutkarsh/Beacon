from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.seed_expansion import build_seed_expansion_packets


def main() -> None:
    parser = argparse.ArgumentParser(description="Create contained packets for train-only Sankat Saathi seed expansion.")
    parser.add_argument("--seed-cards", default="data/seed_cards/sankat_saathi_seed_cards_v1.jsonl")
    parser.add_argument("--out-dir", default="data/seed_cards/expansion_v2_train_only")
    parser.add_argument("--leakage-report", default="data/expanded/sankat_expansion_v1_600/split_leakage_report.json")
    parser.add_argument("--pattern-report", default="data/expanded/sankat_expansion_v1_600/pattern_collapse_report.json")
    args = parser.parse_args()

    manifest = build_seed_expansion_packets(
        ROOT / args.seed_cards,
        ROOT / args.out_dir,
        leakage_report_path=ROOT / args.leakage_report,
        pattern_report_path=ROOT / args.pattern_report,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
