from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.assistant_sft import DEFAULT_RULE_MANIFEST, validate_bundle


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate a Beacon assistant-SFT review bundle.")
    parser.add_argument("dataset_dir", nargs="?", default=str(ROOT / "data" / "assistant_sft" / "beacon_assistant_sft_v1_draft"))
    parser.add_argument("--stage", choices=["candidate", "export"], default="candidate")
    parser.add_argument("--rule-manifest", default=str(DEFAULT_RULE_MANIFEST))
    args = parser.parse_args()

    errors, report = validate_bundle(Path(args.dataset_dir), stage=args.stage, rule_manifest_path=Path(args.rule_manifest))
    print(json.dumps({"errors": errors, "report": report}, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
