from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.local_doc_tool import (  # noqa: E402
    DEFAULT_OUT_DIR,
    load_doc_index,
    load_section_index,
    read_official_doc,
    search_official_docs,
)


def emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if isinstance(payload, list):
        for row in payload:
            print(format_row(row))
        return
    print(format_row(payload))


def format_row(row: dict[str, Any]) -> str:
    if "snippet" in row:
        facts = ", ".join(row.get("key_facts", [])) or "none"
        return (
            f"{row.get('doc_id')} / {row.get('section_id')}\n"
            f"  score: {row.get('score')}\n"
            f"  title: {row.get('title')}\n"
            f"  key_facts: {facts}\n"
            f"  snippet: {row.get('snippet')}"
        )
    hazards = ", ".join(row.get("hazards", [])) or "none"
    return (
        f"{row.get('doc_id')}\n"
        f"  title: {row.get('title')}\n"
        f"  organization: {row.get('organization')}\n"
        f"  hazards: {hazards}"
    )


def list_docs(args: argparse.Namespace) -> None:
    docs = load_doc_index(args.index_dir)
    hits = search_official_docs(
        args.query,
        docs,
        hazard=args.hazard or None,
        organization=args.org or None,
        top_k=args.top_k,
    )
    emit([hit.__dict__ for hit in hits], args.json)


def show_doc(args: argparse.Namespace) -> None:
    docs = load_doc_index(args.index_dir)
    for row in docs:
        if row.get("doc_id") == args.doc_id:
            emit(row, args.json)
            return
    raise SystemExit(f"Unknown doc_id: {args.doc_id}")


def read_doc(args: argparse.Namespace) -> None:
    sections = load_section_index(args.index_dir)
    hits = read_official_doc(args.doc_id, args.section_or_query, sections, args.top_k)
    emit([hit.__dict__ for hit in hits], args.json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beacon offline official-doc CLI.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-docs", help="Search official offline docs.")
    list_parser.add_argument("--query", required=True)
    list_parser.add_argument("--hazard", default="")
    list_parser.add_argument("--org", default="")
    list_parser.add_argument("--top-k", type=int, default=5)
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=list_docs)

    show_parser = subparsers.add_parser("show-doc", help="Show one doc card by doc_id.")
    show_parser.add_argument("--doc-id", required=True)
    show_parser.add_argument("--json", action="store_true")
    show_parser.set_defaults(func=show_doc)

    read_parser = subparsers.add_parser("read-doc", help="Search sections inside one official doc.")
    read_parser.add_argument("--doc-id", required=True)
    read_parser.add_argument("--section-or-query", required=True)
    read_parser.add_argument("--top-k", type=int, default=3)
    read_parser.add_argument("--json", action="store_true")
    read_parser.set_defaults(func=read_doc)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
