from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.assistant_sft import DEFAULT_RULE_MANIFEST, validate_bundle, write_bundle


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build a Beacon assistant-SFT draft review bundle.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "assistant_sft" / "beacon_assistant_sft_v1_draft"))
    parser.add_argument("--rule-manifest", default=str(DEFAULT_RULE_MANIFEST))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rule_manifest = Path(args.rule_manifest)
    manifest = write_bundle(out_dir, rule_manifest)
    errors, report = validate_bundle(out_dir, stage="candidate", rule_manifest_path=rule_manifest)
    payload = {"manifest": manifest, "validation_report": report, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
