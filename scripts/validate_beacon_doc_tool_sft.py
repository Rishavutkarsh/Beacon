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
    read_jsonl,
    validate_tool_sft_rows,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Beacon official-doc tool SFT rows.")
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_SFT_OUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.out_dir / "all_rows.jsonl")
    result = validate_tool_sft_rows(rows, args.index_dir)
    write_json(args.out_dir / "manifest.json", result.manifest)
    print(json.dumps(result.manifest, indent=2, sort_keys=True))
    if result.errors and not args.allow_blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
