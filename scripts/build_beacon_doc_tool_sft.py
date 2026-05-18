from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_doc_tool import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DEFAULT_SFT_OUT_DIR,
    DEFAULT_TOOL_SFT_ROWS,
    build_tool_sft_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Beacon tool-use SFT rows for offline official document lookup.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SFT_OUT_DIR)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TOOL_SFT_ROWS)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    result = build_tool_sft_package(args.out_dir, args.index_dir, target_rows=args.target_rows)
    print(json.dumps(result.manifest, indent=2, sort_keys=True))
    if result.errors and not args.allow_blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
