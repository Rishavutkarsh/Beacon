from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_grounding_cards import DEFAULT_OUT_DIR, read_jsonl, validate_cards, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Beacon local grounding card bundle.")
    parser.add_argument("cards_dir", nargs="?", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--allow-draft-only", action="store_true")
    args = parser.parse_args()

    cards_dir = Path(args.cards_dir)
    cards = read_jsonl(cards_dir / "draft_grounding_cards.jsonl")
    reviews = read_jsonl(cards_dir / "grounding_card_reviews.jsonl")
    result = validate_cards(cards, reviews)
    write_json(cards_dir / "grounding_card_manifest.json", result.manifest)
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if result.errors and not args.allow_draft_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

