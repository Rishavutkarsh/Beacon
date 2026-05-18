from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_doc_tool import DEFAULT_OUT_DIR, build_indexes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Beacon offline official document lookup indexes.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    result = build_indexes(args.out_dir)
    print(json.dumps(result.manifest, indent=2, sort_keys=True))
    if result.errors and not args.allow_blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
