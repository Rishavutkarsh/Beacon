from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.expansion_gate import make_audit_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a hash manifest for a Sankat Saathi expansion audit bundle.")
    parser.add_argument("--run-dir", default="data/expanded/sankat_expansion_v1")
    args = parser.parse_args()
    bundle = make_audit_bundle(ROOT / args.run_dir)
    print(json.dumps(bundle, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
