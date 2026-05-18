from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.grounding_aware_sft import DEFAULT_CARDS_DIR, DEFAULT_OUT_DIR, build_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Beacon grounding-aware SFT candidate package.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cards-dir", default=str(DEFAULT_CARDS_DIR))
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    result = build_package(Path(args.out_dir), Path(args.cards_dir))
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if result.errors:
        raise SystemExit(1)
    if result.warnings and not args.allow_blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
