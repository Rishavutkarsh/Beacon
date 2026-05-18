from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_grounding_research import DEFAULT_OUT_DIR, DEFAULT_SOURCE_CORPUS, build_research_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Beacon local-grounding source research pack.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    args = parser.parse_args()

    result = build_research_pack(Path(args.out_dir), Path(args.source_corpus))
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if result.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
