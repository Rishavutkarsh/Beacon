from __future__ import annotations

import gc
import hashlib
import ast
import json
import math
import os
import re
import subprocess
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = Path("/kaggle/working/beacon_final_eval_tool_dpo_cpt_fullprompt_ckpt50_e2e_answers_v1")
RESULT_PATH = OUT_DIR / "answer_result.json"
CANDIDATE_ID = "tool_dpo_cpt_fullprompt_ckpt50_keyword_prompt_50_e2e"
EXPECTED_ROWS = 108
MAX_EVAL_ROWS = 50
TOOL_REQUIRED_EVAL_ROWS = 35
TOOL_NOT_REQUIRED_EVAL_ROWS = 15
EXPECTED_FINAL_EVAL_HASH = "09815854c2d5049371ec409492509846db89d9b598c2aed24dba2f9adfe74f3a"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 260
MAX_TOOL_CALLS = 3
PINNED = [
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
    "transformers==5.5.0",
    "peft==0.19.1",
    "accelerate==1.13.0",
    "bitsandbytes==0.49.2",
    "datasets==4.3.0",
    "sentencepiece",
]
DATA_CANDIDATES = [
    Path("/kaggle/input/beacon-tool-plus-no-tool-final-eval-v1"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-plus-no-tool-final-eval-v1"),
]
TOOL_INDEX_CANDIDATES = [
    Path("/kaggle/input/beacon-official-doc-tool-v1"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-official-doc-tool-v1"),
]
MODEL_CANDIDATES = [
    Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1"),
]
ADAPTER_CANDIDATES = [
    Path("/kaggle/input/beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter"),
    Path("/kaggle/input/datasets/nehak76044/beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter"),
]
TOKEN_RE = re.compile(r"[a-z0-9]+")
TOOL_CALL_RE = re.compile(
    r"(?:<tool_call>|<\|tool_call>)\s*(\{.*?\})\s*(?:</tool_call>|<tool_call\|>)",
    re.S,
)
TOOL_SYSTEM_CONTRACT = """You are Beacon, an offline crisis companion for India-relevant disaster situations.

You have offline lexical document tools. They do not infer synonyms well. Translate the user's meaning into the indexed hazard handles and retrieval keywords below.

Tools:

search_official_docs:
<tool_call>{"name":"search_official_docs","arguments":{"query":"...", "hazard":"...", "organization":null, "top_k":4}}</tool_call>

read_official_doc:
<tool_call>{"name":"read_official_doc","arguments":{"doc_id":"...", "section_or_page_query":"...", "top_k":5}}</tool_call>

Tool-call rules:
- When calling a tool, output exactly one complete <tool_call>...</tool_call> block and no other text.
- Use valid JSON with double quotes. No Markdown fences, comments, prose, multiple calls, or invented tool results.
- Never use placeholders such as "DOC_ID", "document_id", "FROM_SEARCH", "FROM_READ", "string", or "official_guidance".
- Never call read_official_doc until search_official_docs returns a concrete doc_id.
- Use section_or_page_query exactly.
- Use top_k from 3 to 5.

Indexed hazard handles:
boil_water, carbon_monoxide, chemical_contamination, children, coastal_evacuation, cold_wave, contamination, cyclone, dehydration, diabetes, diarrhoea, disinfection, drug_safety, electrical, emergencies, emergency_kit, evacuation, fire, flood, flood_cleanup, floodwater, food, food_safety, generators, heat_cold, heatwave, indoor_air, insulin, landslide, lightning, live_fact_uncertainty, medicine, medicine_disruption, misinformation, mold, ors, power_outage, preparedness, risk_communication, route_safety, sanitation, shelter, shelter_hygiene, smoke, storm, structural, symptoms, vulnerable_people, wash, water_safety, weather_alerts, winter_storm, wounds.

Search query dialect:
Use one indexed hazard handle plus 2-6 exact keywords. Do not paste the whole user question or the whole handle list. Use organization:null unless the user asks for one.

Useful query patterns:
cyclone NDMA preparedness emergency kit coastal evacuation shelter
heatwave NDMA heat illness vulnerable_people
weather_alerts IMD warning category
floodwater electrical contamination wounds
route_safety flood turn around don't drown
food_safety power_outage refrigerator 4 hours 40 degrees freezer 48 hours
water_safety boil_water disinfection bleach 30 minutes emergency drinking water
wash sanitation shelter_hygiene children emergencies
diarrhoea ors dehydration symptoms
carbon_monoxide generators outdoors 20 feet windows doors vents
drug_safety medicine_disruption insulin diabetes wet medicine disaster
structural reenter flooded home mold building damage landslide
fire smoke evacuation wildfire
risk_communication misinformation rumor official source

Use tools for:
- Questions asking what offline/official documents say.
- Exact official facts: numbers, thresholds, durations, temperatures, quantities, named guidance, warning-category definitions, or source-sensitive rules.
- Stable disaster/public-health guidance where exact wording matters.

Do not use tools for:
- Emotional support, translation, summarizing user text, or broad common-sense safety steps.
- Live/current verification: road open now, bridge safe now, shelter beds now, rescue ETA, active warning today, route status, or forwarded local claims.
- Pill identification, dose advice, diagnosis, photo/building safety certification, or overriding medicine labels.

Protocol:
1. If tools are needed, first call search_official_docs.
2. If search returns a relevant doc_id, call read_official_doc with that exact doc_id.
3. If search returns no relevant document, stop tool use and say the offline documents do not support the specific claim.
4. After reading, answer only from returned evidence. If evidence is weak or missing, say so and give safe generic guidance.

Answering:
Be short, concrete, calm, and crisis-appropriate. Copy exact constants only from returned evidence. Do not fabricate official facts, live statuses, routes, shelter capacity, rescue timing, medicine identity, doses, or safety guarantees. Match the user's language style when practical, including Roman Hinglish.

Examples:
User: Is doc me cyclone ke bare me kya useful official baat hai?
<tool_call>{"name":"search_official_docs","arguments":{"query":"cyclone NDMA preparedness emergency kit", "hazard":"cyclone", "organization":null, "top_k":4}}</tool_call>

User: Generator balcony me chala sakte hain? Door open rahega.
<tool_call>{"name":"search_official_docs","arguments":{"query":"carbon_monoxide generators outdoors 20 feet windows doors vents", "hazard":"carbon_monoxide", "organization":null, "top_k":4}}</tool_call>

User: Can Beacon confirm rescue boats will arrive in one hour?
Do not use tools. Say offline documents cannot verify current rescue ETA, then give safer next steps."""


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_rows = [row for row in rows if bool(row.get("tool_required", False))]
    no_tool_rows = [row for row in rows if not bool(row.get("tool_required", False))]
    if len(tool_rows) < TOOL_REQUIRED_EVAL_ROWS or len(no_tool_rows) < TOOL_NOT_REQUIRED_EVAL_ROWS:
        raise RuntimeError(
            "Not enough rows for mixed eval: "
            f"tool_required={len(tool_rows)}/{TOOL_REQUIRED_EVAL_ROWS}, "
            f"tool_not_required={len(no_tool_rows)}/{TOOL_NOT_REQUIRED_EVAL_ROWS}"
        )
    eval_rows = tool_rows[:TOOL_REQUIRED_EVAL_ROWS] + no_tool_rows[:TOOL_NOT_REQUIRED_EVAL_ROWS]
    if len(eval_rows) != MAX_EVAL_ROWS:
        raise RuntimeError(f"Mixed eval row count mismatch: expected={MAX_EVAL_ROWS} got={len(eval_rows)}")
    return eval_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_deps() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *PINNED])


def resolve(candidates: list[Path], child: str, label: str) -> Path:
    for candidate in candidates:
        if (candidate / child).exists():
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for found in sorted(input_root.rglob(child)):
            lower = str(found.parent).lower()
            if child == "final_eval.jsonl" and "beacon-tool-plus-no-tool-final-eval" in lower:
                return found.parent
            if child in {"official_doc_index.jsonl", "official_doc_chunk_index.jsonl"} and "beacon-official-doc-tool" in lower:
                return found.parent
            if child == "config.json" and "gemma-4" in lower and "e2b-it" in lower:
                return found.parent
    visible = [str(path) for path in sorted(input_root.iterdir())] if input_root.exists() else []
    raise RuntimeError(f"Could not resolve {label}; visible_inputs={visible}")


def find_child_dir(root: Path, child: str) -> Path | None:
    if (root / child).exists():
        return root
    for found in sorted(root.rglob(child)):
        return found.parent
    return None


def resolve_adapter() -> Path:
    for candidate in ADAPTER_CANDIDATES:
        if candidate.exists():
            found = find_child_dir(candidate, "adapter_config.json")
            if found is not None:
                return found
    input_root = Path("/kaggle/input")
    adapter_dirs = [
        found.parent for found in sorted(input_root.rglob("adapter_config.json"))
        if "beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter" in str(found).lower()
    ] if input_root.exists() else []
    if adapter_dirs:
        return adapter_dirs[0]
    zip_candidates = [
        path for path in sorted(input_root.rglob("*.zip"))
        if "beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter" in str(path).lower()
    ] if input_root.exists() else []
    for zip_path in zip_candidates:
        extract_dir = OUT_DIR / "extracted_adapter" / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        found = find_child_dir(extract_dir, "adapter_config.json")
        if found is not None:
            return found
    visible = [str(path) for path in sorted(input_root.iterdir())] if input_root.exists() else []
    raise RuntimeError(f"Could not resolve adapter; visible_inputs={visible}")


def assert_t4(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    names = [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())]
    if torch_module.cuda.device_count() != 1 or "t4" not in names[0].lower():
        raise RuntimeError(f"Expected single T4; visible={names}")
    return {"gpu_names": names, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def normalize_text(text: str) -> str:
    text = text.replace("\u00b0", " degrees ")
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text).lower())


def bm25_search(query: str, docs: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
    query_terms = tokenize(query)
    if not query_terms or not docs:
        return []
    query_counts = Counter(query_terms)
    doc_terms = [tokenize(text) for _, text in docs]
    avg_len = sum(len(terms) for terms in doc_terms) / max(len(doc_terms), 1)
    doc_freq: Counter[str] = Counter()
    for terms in doc_terms:
        doc_freq.update(set(terms))
    scored: list[tuple[str, float]] = []
    for (row_id, _), terms in zip(docs, doc_terms, strict=True):
        tf = Counter(terms)
        score = 0.0
        for term, q_count in query_counts.items():
            if term not in tf:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + 1.5 * (1 - 0.75 + 0.75 * (len(terms) / max(avg_len, 1)))
            score += q_count * idf * ((tf[term] * 2.5) / denom)
        if score > 0:
            scored.append((row_id, round(score, 4)))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def search_official_docs(query: str, doc_index: list[dict[str, Any]], hazard: str | None = None, organization: str | None = None, top_k: int = 5) -> dict[str, Any]:
    candidates = doc_index
    if hazard:
        hazard_lower = hazard.lower()
        candidates = [row for row in candidates if hazard_lower in " ".join(row.get("hazards", [])).lower()]
    if organization:
        org_lower = organization.lower()
        candidates = [row for row in candidates if org_lower in str(row.get("organization", "")).lower()]
    docs = [
        (
            row["doc_id"],
            " ".join([row.get("doc_id", ""), row.get("title", ""), row.get("organization", ""), row.get("abstract", ""), " ".join(row.get("hazards", []))]),
        )
        for row in candidates
    ]
    by_id = {row["doc_id"]: row for row in candidates}
    hits = []
    for doc_id, score in bm25_search(query, docs, max(1, min(int(top_k), 8))):
        row = by_id[doc_id]
        hits.append({"score": score, "doc_id": doc_id, "title": row.get("title", ""), "organization": row.get("organization", ""), "hazards": list(row.get("hazards", []))})
    return {"documents": hits}


def read_official_doc(doc_id: str, section_query: str, section_index: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
    candidates = [row for row in section_index if row["doc_id"] == doc_id]
    docs = [
        (
            row["section_id"],
            " ".join([row.get("section_id", ""), row.get("title", ""), " ".join(row.get("hazards", [])), row.get("text", "")]),
        )
        for row in candidates
    ]
    by_id = {row["section_id"]: row for row in candidates}
    hits = []
    for section_id, score in bm25_search(section_query, docs, max(1, min(int(top_k), 6))):
        row = by_id[section_id]
        hits.append({"score": score, "doc_id": doc_id, "section_id": section_id, "title": row.get("title", ""), "snippet": row.get("snippet", "")[:900], "key_facts": list(row.get("key_facts", []))})
    return {"sections": hits}


def parse_tool_call(text: str) -> dict[str, Any] | None:
    match = TOOL_CALL_RE.search(text)
    raw = match.group(1).strip() if match else ""
    if not raw and "<tool_call>" in text:
        after = text.split("<tool_call>", 1)[1]
        start = after.find("{")
        if start >= 0:
            depth = 0
            end = -1
            for offset, char in enumerate(after[start:], start=start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = offset + 1
                        break
            if end > start:
                raw = after[start:end]
    if not raw:
        raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = parse_function_style_tool_call(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("name"), str) and isinstance(parsed.get("arguments"), dict):
        return parsed
    return None


def parse_function_style_tool_call(text: str) -> dict[str, Any] | None:
    match = re.search(r"tool_call\s*:\s*(search_official_docs|read_official_doc)\s*\((.*?)\)\s*$", text.strip(), re.S)
    if not match:
        return None
    name = match.group(1)
    expr = f"{name}({match.group(2)})"
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    args: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            value = ast.literal_eval(keyword.value)
        except Exception:
            if isinstance(keyword.value, ast.Name) and keyword.value.id in {"None", "null"}:
                value = None
            else:
                return None
        args[keyword.arg] = value
    return {"name": name, "arguments": args}


def is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"", "...", "specific search query", "document_id_from_search_official_docs", "document_id", "null", "none", "string"}:
        return True
    return any(token in lowered for token in ["doc_id_from", "document_id_from", "from_search", "from_read", "doc_id_", "official_guidance"])


def execute_tool(
    call: dict[str, Any],
    doc_index: list[dict[str, Any]],
    section_index: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    name = str(call.get("name", ""))
    args = call.get("arguments", {})
    if name == "search_official_docs":
        query = str(args.get("query", "")).strip()
        if is_placeholder(query):
            return name, {"error": "invalid tool call: search_official_docs requires a concrete query from the user request."}, True
        payload = search_official_docs(query, doc_index, args.get("hazard"), args.get("organization"), int(args.get("top_k", 5)))
        state["last_search_doc_ids"] = [str(item.get("doc_id", "")) for item in payload.get("documents", []) if item.get("doc_id")]
        state["read_requests"] = set()
        if not state["last_search_doc_ids"]:
            state["search_returned_no_documents"] = True
        return name, payload, False
    if name == "read_official_doc":
        doc_id = str(args.get("doc_id", "")).strip()
        section_query = str(args.get("section_or_page_query") or args.get("query") or "").strip()
        allowed_doc_ids = set(state.get("last_search_doc_ids", []))
        if is_placeholder(doc_id):
            return name, {"error": "invalid tool call: read_official_doc requires a concrete doc_id returned by search_official_docs."}, True
        if not allowed_doc_ids:
            return name, {"error": "invalid tool call: read_official_doc cannot be called because the last search returned no documents."}, True
        if doc_id not in allowed_doc_ids:
            return name, {"error": f"invalid tool call: doc_id {doc_id!r} was not returned by the immediately preceding search."}, True
        if is_placeholder(section_query):
            return name, {"error": "invalid tool call: read_official_doc requires a concrete section_or_page_query."}, True
        request_key = (doc_id, section_query.lower(), int(args.get("top_k", 3)))
        read_requests = state.setdefault("read_requests", set())
        if request_key in read_requests:
            return name, {"error": "invalid tool call: repeated identical read_official_doc request. Stop tool use and answer from available evidence."}, True
        read_requests.add(request_key)
        payload = read_official_doc(doc_id, section_query, section_index, int(args.get("top_k", 3)))
        return name, payload, False
    return name or "unknown_tool", {"error": f"unknown tool: {name}"}, True


def initial_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    source_messages = row.get("messages", [])
    system = source_messages[0].get("content", "") if source_messages and source_messages[0].get("role") == "system" else ""
    system = f"{system}\n\n{TOOL_SYSTEM_CONTRACT}".strip()
    user_prompt = str(row.get("user_prompt") or "")
    if not user_prompt:
        for message in source_messages:
            if message.get("role") == "user":
                user_prompt = str(message.get("content", ""))
                break
    if not user_prompt:
        for message in row.get("prompt_messages", []):
            if message.get("role") == "user":
                user_prompt = str(message.get("content", ""))
                break
    if not user_prompt:
        user_prompt = str(row.get("prompt") or "")
    return [
        {"role": "system", "content": str(system)},
        {"role": "user", "content": user_prompt},
    ]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = "user" if message.get("role") == "tool_result" else message.get("role", "user")
        chunks.append(f"<|turn>{role}\n{message.get('content', '')}<turn|>")
    chunks.append("<|turn>assistant\n")
    return "\n".join(chunks).removeprefix("<bos>")


def generate_text(model: Any, tokenizer: Any, messages: list[dict[str, str]], torch_module: Any) -> str:
    prompt = render_prompt(tokenizer, messages)
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    encoded = inner(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
    with torch_module.inference_mode():
        output = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None, pad_token_id=inner.eos_token_id)
    return inner.decode(output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def run_tool_loop(model: Any, tokenizer: Any, row: dict[str, Any], doc_index: list[dict[str, Any]], section_index: list[dict[str, Any]], torch_module: Any) -> tuple[str, list[dict[str, Any]], str]:
    messages = initial_messages(row)
    trace: list[dict[str, Any]] = []
    finish_reason = "final_answer"
    state: dict[str, Any] = {"last_search_doc_ids": [], "read_requests": set()}
    for step in range(MAX_TOOL_CALLS + 1):
        assistant_text = generate_text(model, tokenizer, messages, torch_module)
        call = parse_tool_call(assistant_text)
        trace.append({"step": step, "role": "assistant", "content": assistant_text, "parsed_tool_call": call})
        if call is None:
            return assistant_text, trace, finish_reason
        if not bool(row.get("tool_required", False)):
            finish_reason = "unexpected_tool_call"
            return assistant_text, trace, finish_reason
        if step >= MAX_TOOL_CALLS:
            finish_reason = "max_tool_calls_reached"
            return assistant_text, trace, finish_reason
        tool_name, payload, terminal_error = execute_tool(call, doc_index, section_index, state)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "tool_result", "name": tool_name, "content": f"<tool_result name=\"{tool_name}\">{json.dumps(payload, ensure_ascii=False)}</tool_result>"})
        trace.append({"step": step, "role": "tool", "name": tool_name, "content": payload})
        if terminal_error:
            finish_reason = "invalid_tool_call"
            return assistant_text, trace, finish_reason
        if tool_name == "search_official_docs" and not payload.get("documents"):
            finish_reason = "search_no_documents"
            return assistant_text, trace, finish_reason
    finish_reason = "loop_exhausted"
    return "", trace, finish_reason


def main() -> None:
    result: dict[str, Any] = {"status": "failure", "stage": "start"}
    write_json(RESULT_PATH, result)
    try:
        result["stage"] = "install"
        install_deps()
        result["package_versions"] = {name: metadata.version(name) for name in ["unsloth", "transformers", "peft", "bitsandbytes"]}
        write_json(RESULT_PATH, result)

        result["stage"] = "imports"
        import unsloth  # noqa: F401
        import torch
        from unsloth import FastLanguageModel
        from peft import PeftModel

        result.update(assert_t4(torch))
        data_dir = resolve(DATA_CANDIDATES, "final_eval.jsonl", "final eval data")
        tool_index_dir = resolve(TOOL_INDEX_CANDIDATES, "official_doc_index.jsonl", "official doc tool index")
        model_path = resolve(MODEL_CANDIDATES, "config.json", "Gemma base model")
        adapter_path = resolve_adapter()
        eval_path = data_dir / "final_eval.jsonl"
        eval_hash = sha256_file(eval_path)
        if eval_hash != EXPECTED_FINAL_EVAL_HASH:
            raise RuntimeError(f"Eval hash mismatch: expected={EXPECTED_FINAL_EVAL_HASH} observed={eval_hash}")
        rows = read_jsonl(eval_path)
        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError(f"Expected {EXPECTED_ROWS} rows; got {len(rows)}")
        eval_rows = select_eval_rows(rows)
        eval_mix = Counter("tool_required" if bool(row.get("tool_required", False)) else "tool_not_required" for row in eval_rows)
        doc_index = read_jsonl(tool_index_dir / "official_doc_index.jsonl")
        section_index = read_jsonl(tool_index_dir / "official_doc_chunk_index.jsonl")
        result.update({
            "data_dir": str(data_dir),
            "tool_index_dir": str(tool_index_dir),
            "model_path": str(model_path),
            "adapter_path": str(adapter_path),
            "eval_hash": eval_hash,
            "row_count": len(rows),
            "eval_row_limit": MAX_EVAL_ROWS,
            "eval_mix": dict(eval_mix),
            "generated_row_target": len(eval_rows),
            "doc_count": len(doc_index),
            "section_count": len(section_index),
        })
        write_json(RESULT_PATH, result)

        result["stage"] = "load_model"
        model, tokenizer = FastLanguageModel.from_pretrained(str(model_path), max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=True)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        inner = getattr(tokenizer, "tokenizer", tokenizer)
        if inner.pad_token is None:
            inner.pad_token = inner.eos_token
        inner.padding_side = "left"
        FastLanguageModel.for_inference(model)
        write_json(RESULT_PATH, result)

        answers_path = OUT_DIR / "answers.jsonl"
        traces_path = OUT_DIR / "tool_traces.jsonl"
        for path in [answers_path, traces_path]:
            if path.exists():
                path.unlink()
        counts: Counter[str] = Counter()
        result["stage"] = "generate_e2e"
        for idx, row in enumerate(eval_rows, start=1):
            answer, trace, finish_reason = run_tool_loop(model, tokenizer, row, doc_index, section_index, torch)
            tool_calls = [item for item in trace if item.get("parsed_tool_call")]
            tool_results = [item for item in trace if item.get("role") == "tool"]
            counts[finish_reason] += 1
            counts[f"tool_calls_{len(tool_calls)}"] += 1
            if tool_calls:
                counts["used_tool"] += 1
            else:
                counts["no_tool_used"] += 1
            first_generation = trace[0].get("content", "") if trace else ""
            answer_row = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "example_id": row["row_id"],
                "candidate_id": CANDIDATE_ID,
                "row_family": row.get("row_family"),
                "tool_required": bool(row.get("tool_required", False)),
                "hazard": row.get("hazard"),
                "risk_level": row.get("risk_level"),
                "prompt": row.get("user_prompt") or row.get("prompt"),
                "model_answer": answer,
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "finish_reason": finish_reason,
                "tool_call_count": len(tool_calls),
                "tool_result_count": len(tool_results),
                "tool_names": [item["parsed_tool_call"]["name"] for item in tool_calls],
                "first_generation_sha256": hashlib.sha256(first_generation.encode("utf-8")).hexdigest(),
                "generation_config": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "max_tool_calls": MAX_TOOL_CALLS, "prompt_mode": "keyword_handle_prompt_35_tool_15_no_tool_rows", "assistant_role": "assistant", "parser": "xml_or_function_style_tool_call"},
            }
            append_jsonl(answers_path, answer_row)
            append_jsonl(traces_path, {"example_id": row["row_id"], "trace": trace})
            result.update({"generated_rows": idx, "tool_loop_counts": dict(counts)})
            write_json(RESULT_PATH, result)
            print(f"[beacon-tool-e2e] {idx}/{len(eval_rows)} calls={len(tool_calls)} finish={finish_reason}", flush=True)

        result["status"] = "success"
        result["stage"] = "complete"
        result["answer_hash"] = sha256_file(answers_path)
        result["trace_hash"] = sha256_file(traces_path)
        result["outputs"] = {"answers": str(answers_path), "tool_traces": str(traces_path)}
        write_json(RESULT_PATH, result)
    except Exception as exc:
        result["status"] = "failure"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-30:]
        write_json(RESULT_PATH, result)
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
