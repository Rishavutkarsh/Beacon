from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package frozen Sankat Saathi eval prompts for Kaggle.")
    parser.add_argument("--prompts", default="data/eval/sankat_saathi_e2b_smoke.jsonl")
    parser.add_argument("--out", default="kaggle/input/sankat-saathi-e2b-eval")
    args = parser.parse_args()
    prompt_path = Path(args.prompts)
    out_dir = Path(args.out)
    rows = read_jsonl(prompt_path)
    if len(rows) < 20:
        raise SystemExit(f"expected at least 20 frozen eval prompts, found {len(rows)}")
    ids = [row.get("prompt_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("prompt_id values must be unique")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prompt_path, out_dir / "prompts.jsonl")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_prompts": str(prompt_path),
        "prompt_count": len(rows),
        "categories": sorted({row.get("category") for row in rows}),
        "critical_count": sum(bool(row.get("critical")) for row in rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset_metadata = {
        "title": "Sankat Saathi E2B Eval",
        "id": "rishavutkarsh/sankat-saathi-e2b-eval",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out_dir}")


if __name__ == "__main__":
    main()
