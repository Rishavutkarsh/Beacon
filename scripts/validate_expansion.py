from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.expansion_gate import EXPANSION_PROFILES, validate_expansion


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Sankat Saathi expansion run and write the full audit report set.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--rule-manifest", default="data/seed_cards/source_rule_manifest_v1.jsonl")
    parser.add_argument("--profile", choices=sorted(EXPANSION_PROFILES), default="calibration")
    parser.add_argument("--skip-final-count-gate", action="store_true", help="Use for calibration pilots; full approval keeps exact count gates on.")
    args = parser.parse_args()
    run_dir = args.run_dir or f"data/expanded/sankat_expansion_{args.profile}"

    result = validate_expansion(
        ROOT / run_dir,
        ROOT / args.rule_manifest,
        profile=args.profile,
        fail_on_count=not args.skip_final_count_gate,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "errors": result.errors[:50],
                "warnings": result.warnings[:50],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if result.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
