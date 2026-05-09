from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.expansion_gate import EXPANSION_PROFILES, build_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sankat Saathi expansion candidate rows with immutable provenance.")
    parser.add_argument("--seed-cards", default="data/seed_cards/sankat_saathi_seed_cards_v1.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--profile", choices=sorted(EXPANSION_PROFILES), default="calibration")
    parser.add_argument("--stage", choices=["calibration", "full"], default=None)
    parser.add_argument("--train-target", type=int, default=None)
    parser.add_argument("--dev-target", type=int, default=None)
    parser.add_argument("--final-target", type=int, default=None)
    parser.add_argument("--max-variants-per-seed", type=int, default=5)
    parser.add_argument("--rule-manifest", default="data/seed_cards/source_rule_manifest_v1.jsonl")
    args = parser.parse_args()
    out_dir = args.out_dir or f"data/expanded/sankat_expansion_{args.profile}"
    stage = args.stage or ("calibration" if args.profile == "calibration" else "full")

    manifest = build_rows(
        ROOT / args.seed_cards,
        ROOT / out_dir,
        stage=stage,
        profile=args.profile,
        train_target=args.train_target,
        dev_target=args.dev_target,
        final_target=args.final_target,
        max_variants_per_seed=args.max_variants_per_seed,
        rule_manifest_path=ROOT / args.rule_manifest,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    if manifest.get("feasibility_errors"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
