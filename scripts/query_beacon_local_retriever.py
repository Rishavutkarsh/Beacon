from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.grounding_aware_sft import DEFAULT_CARDS_DIR, load_cards, retrieve_cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Beacon local grounding-card retriever.")
    parser.add_argument("query")
    parser.add_argument("--cards-dir", default=str(DEFAULT_CARDS_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-unapproved", action="store_true")
    args = parser.parse_args()

    cards = load_cards(Path(args.cards_dir), include_unapproved=args.include_unapproved)
    hits = retrieve_cards(args.query, cards, top_k=args.top_k)
    print(json.dumps([hit.__dict__ for hit in hits], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
