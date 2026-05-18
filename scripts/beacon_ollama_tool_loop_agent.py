from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
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


PROMPT_PATH = ROOT / "prompts" / "beacon_tool_system_prompt_v1.txt"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PLACEHOLDER_RE = re.compile(
    r"\b(DOC_ID|document_id|doc_id_from_search|doc_id_from_search_result|"
    r"FROM_SEARCH|FROM_READ|string|specific search query|official_guidance|\.\.\.)\b",
    re.I,
)
LIVE_STATUS_RE = re.compile(
    r"\b(open now|safe now|right now|today|tonight|current|currently|"
    r"available now|rescue.*(?:eta|minutes?|hours?)|which road|road.*open|"
    r"bridge.*open|shelter.*(?:space|beds?|available))\b",
    re.I,
)
FLOODWATER_BABY_ITEM_RE = re.compile(
    r"\b(flood|floodwater).*\b(baby bottle|nipples?|pacifiers?|cutting boards?)\b|"
    r"\b(baby bottle|nipples?|pacifiers?|cutting boards?).*\b(flood|floodwater)\b",
    re.I,
)
FLOODWATER_BABY_ITEM_QUERY = (
    "floodwater baby bottle nipples pacifiers wooden cutting boards throw out "
    "sanitizing methods are not effective food safety CDC"
)


def load_system_prompt() -> str:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        prompt
        + "\n\nUse concrete tool-call JSON, not placeholder JSON from schema examples. "
        'Example search: <tool_call>{"name":"search_official_docs","arguments":{"query":"food_safety floodwater baby bottle nipples pacifiers","hazard":"food_safety","organization":null,"top_k":4}}</tool_call> '
        'Example read after a real search result: <tool_call>{"name":"read_official_doc","arguments":{"doc_id":"cdc_food_after_emergency","section_or_page_query":"baby bottle nipples pacifiers floodwater","top_k":5}}</tool_call>'
        + "\n\nRuntime note: this local runner supports up to three sequential tool calls. "
        "If floodwater touched baby bottle nipples, pacifiers, or wooden cutting boards, "
        "do not recommend bleach sanitizing; official CDC evidence says to throw these items out "
        "because sanitizing methods are not effective for removing floodwater contaminants."
    )


SYSTEM_PROMPT = load_system_prompt()


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
        "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": num_predict},
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
    for marker in ("...done thinking.", "â€¦done thinking."):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    if text.startswith("Thinking...") and "\n\n" in text:
        text = text.split("\n\n", 1)[1].strip()
    return text


def parse_tool_call(text: str) -> dict[str, Any] | None:
    if len(TOOL_CALL_RE.findall(text)) != 1:
        return None
    match = TOOL_CALL_RE.fullmatch(text.strip())
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("arguments"), dict):
        return payload
    return None


def as_int(value: Any, default: int, minimum: int = 1, maximum: int = 8) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def contains_placeholder_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.search(value))
    if isinstance(value, dict):
        return any(contains_placeholder_value(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder_value(item) for item in value)
    return False


class DocTools:
    def __init__(self, index_dir: Path) -> None:
        self.docs = load_doc_index(index_dir)
        self.sections = load_section_index(index_dir)
        self.sections_by_id = {
            (str(section.get("doc_id")), str(section.get("section_id"))): section for section in self.sections
        }

    def search(
        self,
        query: str,
        hazard: str | None = None,
        organization: str | None = None,
        top_k: int = 4,
    ) -> dict[str, Any]:
        hits = search_official_docs(
            query=query,
            doc_index=self.docs,
            hazard=hazard,
            organization=organization,
            top_k=top_k,
        )
        return {"documents": [hit.__dict__ for hit in hits]}

    def read(self, doc_id: str, section_or_page_query: str, top_k: int = 5) -> dict[str, Any]:
        hits = read_official_doc(doc_id, section_or_page_query, self.sections, top_k=top_k)
        sections: list[dict[str, Any]] = []
        for hit in hits:
            row = hit.__dict__.copy()
            source = self.sections_by_id.get((str(hit.doc_id), str(hit.section_id)), {})
            row["text"] = source.get("text", row.get("snippet", ""))
            sections.append(row)
        return {"sections": sections}

    def retrieve(self, query: str) -> dict[str, Any]:
        if FLOODWATER_BABY_ITEM_RE.search(query):
            sections = self.read(
                "cdc_food_after_emergency",
                "throw out wooden cutting boards baby bottle nipples pacifiers floodwaters sanitizing methods are not effective",
                top_k=1,
            )["sections"]
            return {
                "query": query,
                "retrieval_query": FLOODWATER_BABY_ITEM_QUERY,
                "documents": self.search(FLOODWATER_BABY_ITEM_QUERY, hazard="food_safety", top_k=4)["documents"],
                "sections": sections,
            }
        documents = self.search(query, top_k=4)["documents"]
        sections: list[dict[str, Any]] = []
        for doc in documents[:3]:
            sections.extend(self.read(str(doc["doc_id"]), query, top_k=2)["sections"])
        return {"query": query, "retrieval_query": query, "documents": documents, "sections": dedupe_sections(sections)}


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


def validate_tool_call(call: dict[str, Any], state: dict[str, Any]) -> str | None:
    name = str(call.get("name", ""))
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return "Tool call arguments must be a JSON object."
    if contains_placeholder_value(args):
        return "Tool call contains placeholder values; use concrete query terms or a returned doc_id."
    if name not in {"search_official_docs", "read_official_doc"}:
        return f"Unsupported tool: {name}"
    if name == "search_official_docs":
        if set(args) != {"query", "hazard", "organization", "top_k"}:
            return "search_official_docs requires exactly query, hazard, organization, and top_k."
        if state.get("search_doc_ids"):
            return "After search_official_docs returns documents, call read_official_doc before searching again."
    if name == "read_official_doc" and not state.get("search_doc_ids"):
        return "Call search_official_docs before read_official_doc."
    if name == "read_official_doc":
        if set(args) != {"doc_id", "section_or_page_query", "top_k"}:
            return "read_official_doc requires exactly doc_id, section_or_page_query, and top_k."
        doc_id = str(args.get("doc_id", ""))
        if doc_id not in state.get("search_doc_ids", set()):
            return f"read_official_doc doc_id must come from search results, got {doc_id!r}."
        section_query = args.get("section_or_page_query")
        if not str(section_query or "").strip():
            return "read_official_doc requires section_or_page_query."
    if name == "search_official_docs" and not str(args.get("query", "")).strip():
        return "search_official_docs requires a non-empty query."
    return None


def run_tool_call(tools: DocTools, call: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments", {})
    if name == "search_official_docs":
        result = tools.search(
            query=str(args.get("query", "")),
            hazard=str(args["hazard"]) if args.get("hazard") is not None else None,
            organization=str(args["organization"]) if args.get("organization") is not None else None,
            top_k=as_int(args.get("top_k"), default=4),
        )
        state["search_doc_ids"] = {str(doc.get("doc_id")) for doc in result.get("documents", [])}
        return result
    if name == "read_official_doc":
        return tools.read(
            doc_id=str(args.get("doc_id", "")),
            section_or_page_query=str(args.get("section_or_page_query", "")),
            top_k=as_int(args.get("top_k"), default=5),
        )
    return {"error": f"Unsupported tool: {name}"}


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
        "Answer with a short action-oriented response and a Citations section. "
        "Use only the evidence above for exact facts."
    )


def initial_tool_loop_prompt(question: str, force_docs: bool = False) -> str:
    if force_docs:
        return (
            f"USER_QUESTION:\n{question}\n\n"
            "Offline official documents are required for this answer. "
            "Start by outputting exactly one search_official_docs tool_call."
        )
    return (
        f"USER_QUESTION:\n{question}\n\n"
        "Decide whether this needs offline official documents. "
        "If yes, output exactly one tool_call. If no, answer directly."
    )


def append_tool_turn(transcript: str, call: dict[str, Any], result: dict[str, Any]) -> str:
    return (
        f"{transcript}\n\n"
        f"ASSISTANT_TOOL_REQUEST:\n{json.dumps(call, ensure_ascii=False)}\n\n"
        f"TOOL_RESULT:\n{json.dumps(result, ensure_ascii=False)}\n\n"
        "Continue. If more document evidence is needed, output exactly one tool_call. "
        "Otherwise answer naturally with citations for exact facts."
    )


def append_tool_error(transcript: str, call: dict[str, Any], error: str) -> str:
    return (
        f"{transcript}\n\n"
        f"INVALID_TOOL_REQUEST:\n{json.dumps(call, ensure_ascii=False)}\n"
        f"TOOL_ERROR:\n{error}\n\n"
        "Correct the tool call if official evidence is still needed. Otherwise answer with a safe boundary."
    )


def live_status_boundary_response(question: str) -> str:
    return (
        "I cannot verify live/current status from offline documents, so I should not claim whether a route, "
        "bridge, shelter, rescue ETA, or warning is current right now.\n\n"
        "Safer next steps: check local authorities or emergency services, avoid relying on forwarded claims, "
        "and choose the lower-risk option until confirmed."
    )


def run_tool_loop(
    question: str,
    tools: DocTools,
    generate: Callable[[str], str],
    max_tool_calls: int = 3,
    force_docs: bool = False,
) -> str:
    transcript = initial_tool_loop_prompt(question, force_docs=force_docs)
    state: dict[str, Any] = {"search_doc_ids": set()}
    successful_tool_calls = 0
    for _ in range(max_tool_calls + 1):
        output = generate(transcript)
        call = parse_tool_call(output)
        if call is None:
            if "<tool_call" in output or "</tool_call>" in output:
                transcript = (
                    f"{transcript}\n\nINVALID_TOOL_REQUEST:\n{output}\n\n"
                    "Tool calls must be exactly one complete <tool_call> JSON block with no extra prose. "
                    "Correct the tool call or answer with a safe boundary."
                )
                max_tool_calls -= 1
                continue
            if force_docs and successful_tool_calls == 0:
                transcript = (
                    f"{transcript}\n\nINVALID_DIRECT_ANSWER:\n{output}\n\n"
                    "Offline official documents were required. Output exactly one search_official_docs tool_call first."
                )
                max_tool_calls -= 1
                continue
            return output
        if max_tool_calls <= 0:
            return "Offline tool-call limit reached before a final answer could be produced."
        error = validate_tool_call(call, state)
        if error:
            transcript = append_tool_error(transcript, call, error)
            max_tool_calls -= 1
            continue
        result = run_tool_call(tools, call, state)
        successful_tool_calls += 1
        transcript = append_tool_turn(transcript, call, result)
        max_tool_calls -= 1
    return "Offline tool-call loop ended without a final answer."


def answer(args: argparse.Namespace) -> None:
    tools = DocTools(args.index_dir)
    question = args.question

    if LIVE_STATUS_RE.search(question):
        print(live_status_boundary_response(question))
        return

    generate = lambda prompt: ollama_generate(  # noqa: E731
        args.ollama_url,
        args.model,
        prompt,
        args.temperature,
        args.timeout_seconds,
        args.num_predict,
    )

    if args.preload_docs:
        evidence = tools.retrieve(question)
        print(generate(prompt_with_evidence(question, evidence)))
        return

    print(run_tool_loop(question, tools, generate, max_tool_calls=args.max_tool_calls, force_docs=args.force_docs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beacon Ollama agent with canonical offline-doc tool loop.")
    parser.add_argument("question")
    parser.add_argument("--model", default="beacon-gemma4-current-best")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--preload-docs", action="store_true", help="Controller retrieves docs first for demo stability.")
    parser.add_argument(
        "--force-docs",
        action="store_true",
        help="Require the model to use the canonical tool loop instead of answering directly.",
    )
    parser.set_defaults(func=answer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
