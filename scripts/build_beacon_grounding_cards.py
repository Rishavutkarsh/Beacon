from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_grounding_cards import DEFAULT_OUT_DIR, build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Beacon local grounding card bundle.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--reviews", default="")
    parser.add_argument("--allow-draft-only", action="store_true")
    args = parser.parse_args()

    result = build_bundle(Path(args.out_dir), Path(args.reviews) if args.reviews else None)
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if result.errors and not args.allow_draft_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

