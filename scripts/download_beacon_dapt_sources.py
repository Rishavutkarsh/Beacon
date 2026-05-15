from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


OUT_DIR = Path("data/dapt_corpus/beacon_crisis_v1_download")
SOURCE_LIST = OUT_DIR / "source_list.jsonl"
MIN_TEXT_CHARS = 900
CHUNK_WORDS = 1200
PDF_LINK_LIMIT_PER_HTML = 20
REQUEST_TIMEOUT = 45
USER_AGENT = "Beacon-DAPT-educational-corpus-builder/1.0"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self.skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"} and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        data = html.unescape(data).strip()
        if not data:
            return
        if self._in_title:
            self.title += data + " "
        if self.skip_depth:
            return
        self.parts.append(data + " ")

    def text(self) -> str:
        return clean_text("".join(self.parts))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?i)(cookie policy|privacy policy|terms of use|share this page)\s*", "", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def filename_for_url(url: str, suffix: str) -> str:
    parsed = urlparse(url)
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", Path(parsed.path).stem or parsed.netloc).strip("_")[:70]
    return f"{stem}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}{suffix}"


def fetch(session: requests.Session, url: str) -> tuple[bytes, str]:
    response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "")


def extract_html_links(raw: bytes, base_url: str) -> list[str]:
    text = raw.decode("utf-8", errors="ignore")
    hrefs = re.findall(r"""href=["']([^"']+)["']""", text, flags=re.IGNORECASE)
    pdfs = []
    for href in hrefs:
        absolute = urljoin(base_url, html.unescape(href))
        if ".pdf" in absolute.lower() and absolute not in pdfs:
            pdfs.append(absolute)
    return pdfs[:PDF_LINK_LIMIT_PER_HTML]


def extract_html(raw: bytes) -> tuple[str, str]:
    parser = TextExtractor()
    parser.feed(raw.decode("utf-8", errors="ignore"))
    return parser.text(), clean_text(parser.title)


def extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF extraction") from exc
    reader = PdfReader(BytesIO(raw))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n\n".join(pages))


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]
    chunks = []
    for start in range(0, len(words), CHUNK_WORDS):
        chunk = " ".join(words[start : start + CHUNK_WORDS]).strip()
        if len(chunk) >= MIN_TEXT_CHARS:
            chunks.append(chunk)
    return chunks


def doc_id(seed: dict[str, Any], url: str) -> str:
    base = seed.get("seed_id") or Path(urlparse(url).path).stem or seed.get("source_id") or "source"
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(base)).strip("_").lower() + "_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


def document_card(seed: dict[str, Any], url: str, raw: bytes, text: str, title: str, status: str, reason: str = "") -> dict[str, Any]:
    document_id = doc_id(seed, url)
    return {
        "document_id": document_id,
        "source_id": seed.get("source_id", "unknown"),
        "url": url,
        "title": title or seed.get("title") or document_id,
        "organization": seed.get("organization", seed.get("source_id", "unknown")),
        "language": seed.get("language", "unknown"),
        "hazards": seed.get("hazards", []),
        "document_type": "pdf" if url.lower().split("?")[0].endswith(".pdf") else seed.get("document_type", "html"),
        "retrieved_at": utc_now(),
        "status": status,
        "reject_reason": reason,
        "raw_sha256": sha256_bytes(raw) if raw else "",
        "text_sha256": sha256_text(text) if text else "",
        "char_count": len(text),
        "estimated_tokens": estimate_tokens(text) if text else 0,
    }


def dapt_rows_for_card(card: dict[str, Any], text: str) -> list[dict[str, Any]]:
    rows = []
    for index, chunk in enumerate(chunk_text(text)):
        rows.append({
            "text_id": f"{card['document_id']}_chunk_{index:04d}",
            "document_id": card["document_id"],
            "source_id": card["source_id"],
            "url": card["url"],
            "title": card["title"],
            "organization": card["organization"],
            "language": card["language"],
            "hazards": card["hazards"],
            "document_type": card["document_type"],
            "retrieved_at": card["retrieved_at"],
            "license": "recorded_for_educational_hackathon_use",
            "text": chunk,
            "estimated_tokens": estimate_tokens(chunk),
            "text_sha256": sha256_text(chunk),
        })
    return rows


def process_url(session: requests.Session, seed: dict[str, Any], url: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw, content_type = fetch(session, url)
    is_pdf = "pdf" in content_type.lower() or url.lower().split("?")[0].endswith(".pdf")
    if is_pdf:
        text = extract_pdf(raw)
        title = str(seed.get("title") or Path(urlparse(url).path).stem)
    else:
        text, title = extract_html(raw)
    card = document_card(seed, url, raw, text, title, "accepted" if len(text) >= MIN_TEXT_CHARS else "rejected", "too_short_or_unextractable" if len(text) < MIN_TEXT_CHARS else "")
    raw_suffix = ".pdf" if is_pdf else ".html"
    raw_path = OUT_DIR / "raw" / filename_for_url(url, raw_suffix)
    extracted_path = OUT_DIR / "extracted" / f"{card['document_id']}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    extracted_path.write_text(text, encoding="utf-8")
    if card["status"] == "rejected":
        return [], card
    return dapt_rows_for_card(card, text), card


def run(args: argparse.Namespace) -> None:
    global OUT_DIR
    OUT_DIR = Path(args.out_dir)
    seeds = read_jsonl(Path(args.source_list))
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    all_rows: list[dict[str, Any]] = read_jsonl(OUT_DIR / "dapt_all.jsonl") if args.append else []
    accepted_cards: list[dict[str, Any]] = read_jsonl(OUT_DIR / "document_cards.jsonl") if args.append else []
    rejected_cards: list[dict[str, Any]] = read_jsonl(OUT_DIR / "rejected_document_cards.jsonl") if args.append else []
    seen_urls: set[str] = set()
    if args.append:
        seen_urls.update(str(card.get("url")) for card in accepted_cards if card.get("url"))
    for seed_index, seed in enumerate(seeds, start=1):
        urls = [str(seed["url"])]
        try:
            raw, _content_type = fetch(session, urls[0])
            discovered = extract_html_links(raw, urls[0]) if not urls[0].lower().split("?")[0].endswith(".pdf") else []
            if discovered and args.follow_pdf_links:
                urls.extend(discovered)
        except Exception:
            pass
        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                rows, card = process_url(session, seed, url)
                all_rows.extend(rows)
                (accepted_cards if card["status"] == "accepted" else rejected_cards).append(card)
                print(f"[beacon-dapt-download] {seed_index}/{len(seeds)} {card['status']} rows={len(rows)} url={url}", flush=True)
            except Exception as exc:
                card = document_card(seed, url, b"", "", str(seed.get("title", "")), "rejected", f"{type(exc).__name__}:{exc}")
                rejected_cards.append(card)
                print(f"[beacon-dapt-download] {seed_index}/{len(seeds)} rejected error={type(exc).__name__} url={url}", flush=True)
            time.sleep(args.sleep)
    write_jsonl(OUT_DIR / "dapt_all.jsonl", all_rows)
    write_jsonl(OUT_DIR / "document_cards.jsonl", accepted_cards)
    write_jsonl(OUT_DIR / "rejected_document_cards.jsonl", rejected_cards)
    by_source = Counter(row["source_id"] for row in all_rows)
    by_hazard = Counter(hazard for row in all_rows for hazard in row.get("hazards", []))
    manifest = {
        "created_at_utc": utc_now(),
        "status": "complete",
        "source_list": str(Path(args.source_list)),
        "seed_count": len(seeds),
        "fetched_url_count": len(seen_urls),
        "accepted_document_count": len(accepted_cards),
        "rejected_document_count": len(rejected_cards),
        "dapt_row_count": len(all_rows),
        "estimated_tokens": sum(int(row["estimated_tokens"]) for row in all_rows),
        "by_source": dict(by_source.most_common()),
        "by_hazard": dict(by_hazard.most_common()),
    }
    write_json(OUT_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download curated Beacon DAPT source-list documents.")
    parser.add_argument("--source-list", default=str(SOURCE_LIST))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--follow-pdf-links", action="store_true", default=True)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
