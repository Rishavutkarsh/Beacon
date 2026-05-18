from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
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


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NEEDS_DOC_RE = re.compile(
    r"\b(official|cite|source|guidance|how long|how many|minutes?|hours?|days?|"
    r"temperature|degrees?|fridge|freezer|boil|bleach|disinfect|insulin|"
    r"generator|carbon monoxide|floodwater|evacuat|cyclone|threshold)\b",
    re.I,
)


QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(flood|floodwater).*\b(baby bottle|nipples?|pacifiers?|cutting boards?)\b|\b(baby bottle|nipples?|pacifiers?|cutting boards?).*\b(flood|floodwater)\b", re.I),
        "floodwater baby bottle nipples pacifiers wooden cutting boards throw out sanitizing methods are not effective food safety CDC",
    ),
    (re.compile(r"\b(generator|garage|carbon monoxide|co\b)\b", re.I), "generator carbon monoxide garage placement outdoors windows doors"),
    (re.compile(r"\b(fridge|refrigerator|freezer|food|power outage)\b", re.I), "power outage refrigerator freezer food safety 40 degrees 4 hours 48 hours"),
    (re.compile(r"\b(insulin|medicine|medication|drug)\b", re.I), "emergency insulin medicine refrigerated storage disaster"),
    (re.compile(r"\b(bleach|boil|disinfect|water)\b", re.I), "emergency drinking water disinfection boil bleach"),
    (
        re.compile(r"\b(flood|floodwater|baby bottle|nipples?|pacifiers?)\b", re.I),
        "floodwater baby bottle nipples pacifiers throw out sanitizing not effective food safety",
    ),
    (re.compile(r"\b(evacuat|cyclone|hurricane|storm)\b", re.I), "evacuation cyclone hurricane storm emergency guidance"),
)

FLOODWATER_BABY_ITEM_RE = re.compile(
    r"\b(flood|floodwater).*\b(baby bottle|nipples?|pacifiers?|cutting boards?)\b|"
    r"\b(baby bottle|nipples?|pacifiers?|cutting boards?).*\b(flood|floodwater)\b",
    re.I,
)
FLOODWATER_BABY_ITEM_QUERY = (
    "floodwater baby bottle nipples pacifiers wooden cutting boards throw out "
    "sanitizing methods are not effective food safety CDC FDA"
)


SYSTEM_PROMPT = """You are Beacon, a local-first crisis guidance assistant.
Use offline official documents for exact constants, thresholds, medical/storage rules, food/water safety rules, generator/carbon monoxide guidance, evacuation rules, and any source-sensitive claim.
If evidence is provided, answer only from that evidence and cite doc_id plus section_id.
If evidence is missing or weak, say the offline docs did not confirm the exact fact and give conservative safety steps.
If floodwater touched baby bottle nipples, pacifiers, or wooden cutting boards, do not recommend bleach sanitizing; official CDC evidence says to throw these items out because sanitizing methods are not effective for removing floodwater contaminants.
You may request a tool with exactly:
<tool_call>{"name":"search_official_docs","arguments":{"query":"..."}}</tool_call>
or
<tool_call>{"name":"read_official_doc","arguments":{"doc_id":"...","section_or_query":"..."}}</tool_call>
Do not invent citations."""


def ollama_generate(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_seconds: int,
    num_predict: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urllib.error.URLError) as exc:
        raise SystemExit(f"Could not reach Ollama at {url}: {exc}") from exc
    return clean_model_response(str(data.get("response", "")))


def clean_model_response(text: str) -> str:
    text = ANSI_RE.sub("", text).strip()
    for marker in ("...done thinking.", "…done thinking."):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    if text.startswith("Thinking...") and "\n\n" in text:
        text = text.split("\n\n", 1)[1].strip()
    return text


class DocTools:
    def __init__(self, index_dir: Path) -> None:
        self.docs = load_doc_index(index_dir)
        self.sections = load_section_index(index_dir)
        self.sections_by_id = {
            (str(section.get("doc_id")), str(section.get("section_id"))): section for section in self.sections
        }

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        hits = search_official_docs(query, self.docs, top_k=top_k)
        return {"documents": [hit.__dict__ for hit in hits]}

    def read(self, doc_id: str, section_or_query: str, top_k: int = 3) -> dict[str, Any]:
        hits = read_official_doc(doc_id, section_or_query, self.sections, top_k=top_k)
        sections: list[dict[str, Any]] = []
        for hit in hits:
            row = hit.__dict__.copy()
            source = self.sections_by_id.get((str(hit.doc_id), str(hit.section_id)), {})
            row["text"] = source.get("text", row.get("snippet", ""))
            sections.append(row)
        return {"sections": sections}

    def retrieve(self, query: str, doc_top_k: int = 3, section_top_k: int = 2) -> dict[str, Any]:
        if FLOODWATER_BABY_ITEM_RE.search(query):
            documents = [
                doc
                for doc in self.search(FLOODWATER_BABY_ITEM_QUERY, top_k=5)["documents"]
                if doc.get("doc_id") in {"cdc_food_after_emergency", "fda_food_water_floods"}
            ]
            sections: list[dict[str, Any]] = []
            sections.extend(
                self.read(
                    "cdc_food_after_emergency",
                    "throw out wooden cutting boards baby bottle nipples pacifiers floodwaters sanitizing methods are not effective",
                    top_k=1,
                )["sections"]
            )
            return {
                "query": query,
                "retrieval_query": FLOODWATER_BABY_ITEM_QUERY,
                "documents": documents,
                "sections": dedupe_sections(sections),
            }

        retrieval_query = expand_query(query)
        documents = self.search(retrieval_query, top_k=doc_top_k)["documents"]
        sections: list[dict[str, Any]] = []
        for doc in documents:
            sections.extend(self.read(str(doc["doc_id"]), retrieval_query, top_k=section_top_k)["sections"])
        sections = dedupe_sections(sections)
        return {"query": query, "retrieval_query": retrieval_query, "documents": documents, "sections": sections}


def expand_query(query: str) -> str:
    additions = [extra for pattern, extra in QUERY_EXPANSIONS if pattern.search(query)]
    if not additions:
        return query
    return query + " " + " ".join(dict.fromkeys(additions))


def dedupe_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for section in sections:
        key = (str(section.get("doc_id")), str(section.get("section_id")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(section)
    return deduped


def evidence_block(evidence: dict[str, Any]) -> str:
    lines = ["OFFLINE_DOC_EVIDENCE:"]
    for section in evidence.get("sections", [])[:4]:
        facts = ", ".join(section.get("key_facts", [])) or "none"
        evidence_text = str(section.get("text") or section.get("snippet") or "")
        lines.extend(
            [
                f"- citation: {section.get('doc_id')} / {section.get('section_id')}",
                f"  title: {section.get('title')}",
                f"  key_facts: {facts}",
                f"  text: {evidence_text}",
            ]
        )
    if len(lines) == 1:
        lines.append("- No matching sections found.")
    return "\n".join(lines)


def prompt_with_evidence(question: str, evidence: dict[str, Any]) -> str:
    return (
        f"{evidence_block(evidence)}\n\n"
        f"USER_QUESTION:\n{question}\n\n"
        "Answer with a short action-oriented response and a Citations section."
    )


def parse_tool_call(text: str) -> dict[str, Any] | None:
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("arguments"), dict):
        return payload
    return None


def run_tool_call(tools: DocTools, call: dict[str, Any]) -> dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments", {})
    if name == "search_official_docs":
        return tools.search(str(args.get("query", "")))
    if name == "read_official_doc":
        return tools.read(str(args.get("doc_id", "")), str(args.get("section_or_query", args.get("query", ""))))
    return {"error": f"Unsupported tool: {name}"}


def answer(args: argparse.Namespace) -> None:
    tools = DocTools(args.index_dir)
    question = args.question

    evidence = tools.retrieve(question) if args.force_docs or NEEDS_DOC_RE.search(question) else {"sections": []}
    if evidence.get("sections"):
        print(
            ollama_generate(
                args.ollama_url,
                args.model,
                prompt_with_evidence(question, evidence),
                args.temperature,
                args.timeout_seconds,
                args.num_predict,
            )
        )
        return

    first_prompt = (
        f"USER_QUESTION:\n{question}\n\n"
        "Answer if no official evidence is needed. If official evidence is needed, request exactly one tool call."
    )
    first = ollama_generate(
        args.ollama_url,
        args.model,
        first_prompt,
        args.temperature,
        args.timeout_seconds,
        args.num_predict,
    )
    call = parse_tool_call(first)
    if not call:
        print(first)
        return

    result = run_tool_call(tools, call)
    followup = (
        f"{first_prompt}\n\nASSISTANT_TOOL_REQUEST:\n{json.dumps(call, ensure_ascii=False)}\n\n"
        f"TOOL_RESULT:\n{json.dumps(result, ensure_ascii=False)}\n\n"
        "Now answer. Cite doc_id and section_id for any exact facts."
    )
    print(ollama_generate(args.ollama_url, args.model, followup, args.temperature, args.timeout_seconds, args.num_predict))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beacon Ollama controller with offline official-doc grounding.")
    parser.add_argument("question")
    parser.add_argument("--model", default="beacon-gemma4-current-best")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--force-docs", action="store_true", help="Retrieve local official docs before the first model call.")
    parser.set_defaults(func=answer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
