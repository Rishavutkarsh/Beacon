from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_doc_tool import (  # noqa: E402
    DEFAULT_OUT_DIR,
    load_doc_index,
    load_section_index,
    read_official_doc,
    search_official_docs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Beacon's offline official-document lookup tool.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hazard", default="")
    parser.add_argument("--org", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--read-top-k", type=int, default=3)
    args = parser.parse_args()

    docs = load_doc_index(args.index_dir)
    sections = load_section_index(args.index_dir)
    doc_hits = search_official_docs(args.query, docs, args.hazard or None, args.org or None, args.top_k)
    payload = {"query": args.query, "documents": [], "sections": []}
    for hit in doc_hits:
        payload["documents"].append(hit.__dict__)
        for section_hit in read_official_doc(hit.doc_id, args.query, sections, args.read_top_k):
            payload["sections"].append(section_hit.__dict__)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
