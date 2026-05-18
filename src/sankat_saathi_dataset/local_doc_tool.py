from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .local_grounding_research import (
    DEFAULT_OUT_DIR as SOURCE_RESEARCH_DIR,
    DEFAULT_SOURCE_CORPUS,
    REQUIRED_HAZARD_FAMILIES,
    ROOT,
    read_jsonl,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "beacon-official-doc-tool-v1"
DEFAULT_OUT_DIR = ROOT / "data" / "local_grounding" / "official_doc_tool_v1"
DEFAULT_SFT_OUT_DIR = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1"
DEFAULT_TOOL_SFT_ROWS = 1200
TOKEN_RE = re.compile(r"[a-z0-9]+")
EXACT_CLAIM_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:hours?|minutes?|days?|feet|degrees?|grams?|g|percent)|"
    r"40\s*degrees|20\s*feet|15\s*(?:grams?|g)|1\s*minute|3\s*minutes|30\s*minutes)\b",
    re.I,
)
UNSUPPORTED_MYTH_CLAIMS = {"72 hours", "3 days", "60 degrees", "60f"}
NEEDS_TOOL_RE = re.compile(
    r"\b(how long|how many|minutes?|hours?|days?|temperature|degrees?|"
    r"fridge|freezer|boil|bleach|disinfect|insulin|diabetes|quick carbs?|"
    r"official|guidance|document|threshold|warning|route|shelter|rescue|open now|safe now)\b",
    re.I,
)
LIVE_STATUS_RE = re.compile(r"\b(open now|safe now|current|right now|tonight|available now|rescue.*minutes?|which road)\b", re.I)


SELECTED_DOC_IDS = [
    "ndma_cyclone_guidelines_pdf",
    "ndma_cyclone",
    "ndma_heat_wave",
    "imd_weather_warnings",
    "cdc_floodwater_safety",
    "cdc_reenter_flooded_home",
    "ready_floods",
    "nws_turn_around",
    "nws_flood_safety",
    "fda_food_water_floods",
    "cdc_food_after_emergency",
    "cdc_emergency_water",
    "epa_emergency_disinfection",
    "who_wash_emergencies",
    "unicef_wash_emergencies",
    "who_diarrhoea",
    "cdc_power_outage",
    "ready_power_outages",
    "cdc_co_clinical_disasters",
    "cdc_diabetes_emergencies",
    "cdc_insulin_emergency",
    "fda_drugs_disaster",
    "epa_flood_cleanup_iaq",
    "ready_landslides",
    "ready_heat",
    "nws_lightning",
    "ready_winter",
    "nws_winter",
    "ready_wildfires",
    "who_risk_comm",
]

INDIA_FALLBACK_WAIVERS = {
    "flood_route": "No clean India-official flood route document is available in the current local corpus; use global/US stable floodwater and turn-around route guidance only for offline safety boundaries, not local road status.",
    "water_wash": "No clean India-official WASH document is available in the current local corpus; use WHO/UNICEF/CDC/EPA stable public-health guidance for generic offline water, sanitation, and disinfection decisions.",
    "food_safety": "No clean FSSAI emergency food document is available in the current local corpus; use CDC/FDA public-domain stable food-safety constants until an India official source is downloaded and reviewed.",
    "power_co_electrical": "No clean India-official power outage or carbon monoxide document is available in the current local corpus; use CDC/Ready stable CO/electrical guidance for generic offline safety, not local utility status.",
    "medicine_diabetes": "No clean India-official diabetes disruption document is available in the current local corpus; use CDC/FDA stable emergency medicine handling guidance while blocking diagnosis, identification, and dose changes.",
    "wounds_cleanup": "No clean India-official flood wound/cleanup document is available in the current local corpus; use CDC/EPA stable cleanup and wound-boundary guidance.",
    "shelter_vulnerable": "No clean India-official shelter hygiene document is available in the current local corpus; use UNICEF/WHO/Ready stable vulnerable-group and hygiene guidance without claiming shelter availability.",
    "landslide_structural": "No clean India-official landslide public guidance is available in the current local corpus; use Ready landslide/structural warning guidance as a temporary offline fallback.",
}


@dataclass(frozen=True)
class SearchHit:
    score: float
    doc_id: str
    title: str
    organization: str
    hazards: list[str]


@dataclass(frozen=True)
class SectionHit:
    score: float
    doc_id: str
    section_id: str
    title: str
    snippet: str
    key_facts: list[str]


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    manifest: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_text(text: str) -> str:
    text = (
        text.replace("â€”", "-")
        .replace("â€“", "-")
        .replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€\u009d", '"')
        .replace("â€", '"')
        .replace("â€˜", "'")
        .replace("Âº", " degrees ")
        .replace("Â°", " degrees ")
        .replace("Â", "")
    )
    replacements = {
        "â€”": "-",
        "â€“": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€\u009d": '"',
        "â€": '"',
        "Â": "",
        "\u00b0": " degrees ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
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


def source_rows(research_dir: Path = SOURCE_RESEARCH_DIR) -> dict[str, dict[str, Any]]:
    return {str(row["document_id"]): row for row in read_jsonl(research_dir / "downloaded_document_cards.jsonl")}


def chunk_rows(source_corpus_dir: Path = DEFAULT_SOURCE_CORPUS) -> list[dict[str, Any]]:
    return read_jsonl(source_corpus_dir / "retrieval_chunks" / "retrieval_chunks.jsonl")


def build_doc_index(research_dir: Path = SOURCE_RESEARCH_DIR) -> list[dict[str, Any]]:
    rows_by_id = source_rows(research_dir)
    selected = []
    for rank, doc_id in enumerate(SELECTED_DOC_IDS):
        row = rows_by_id.get(doc_id)
        if not row:
            continue
        selected.append(
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": doc_id,
                "document_id": doc_id,
                "rank": rank,
                "title": row.get("title", ""),
                "organization": row.get("organization", ""),
                "jurisdiction": row.get("jurisdiction", ""),
                "source_id": row.get("source_id", ""),
                "url": row.get("url", ""),
                "canonical_url": row.get("url", ""),
                "terms_url": row.get("terms_url", ""),
                "published_date": row.get("published_date", ""),
                "modified_date": row.get("modified_date", ""),
                "retrieved_at": row.get("retrieved_at", ""),
                "hazards": row.get("hazards", []) or [],
                "hazard_families": hazard_families_for(row.get("hazards", []) or []),
                "license": row.get("license", ""),
                "copyright_status": row.get("copyright_status", ""),
                "can_store_raw": bool(row.get("can_store_raw", False)),
                "can_retrieve": bool(row.get("can_retrieve", False)),
                "can_train": bool(row.get("can_train", False)),
                "review_status": row.get("review_status", ""),
                "selection_role": selection_role(str(row.get("review_status", ""))),
                "selection_rationale": selection_rationale(row),
                "official_tier": official_tier(row),
                "accepted_uses": ["offline_doc_lookup", "tool_sft_trace_source"],
                "quality_score": row.get("overall_score", 0),
                "abstract": make_abstract(row),
                "offline_use": "official_doc_lookup",
                "live_status_policy": "never_claim_current_status" if "live_fact_uncertainty" in row.get("hazards", []) else "stable_guidance_only",
                "training_export_allowed": False,
            }
        )
    return selected


def selection_role(review_status: str) -> str:
    return {
        "accepted_core": "core",
        "accepted_supporting": "supporting",
        "accepted_retrieval_only": "tool_only",
    }.get(review_status, "review_only")


def official_tier(row: dict[str, Any]) -> str:
    jurisdiction = str(row.get("jurisdiction", "")).lower()
    organization = str(row.get("organization", "")).lower()
    if jurisdiction == "india" and any(term in organization for term in ["national disaster", "meteorological", "ministry"]):
        return "india_official"
    if any(term in organization for term in ["world health", "unicef"]):
        return "un_or_global_public_health"
    if any(term in organization for term in ["centers for disease", "food and drug", "environmental protection", "weather service", "ready.gov"]):
        return "public_domain_global_applicable"
    return "supporting_authority"


def selection_rationale(row: dict[str, Any]) -> str:
    role = selection_role(str(row.get("review_status", "")))
    hazards = ", ".join(row.get("hazards", [])[:4])
    return f"Selected as {role} because it is an accepted authoritative source with clean extraction for {hazards}."


def hazard_families_for(hazards: list[str]) -> list[str]:
    hazard_set = set(hazards)
    return [family for family, required in REQUIRED_HAZARD_FAMILIES.items() if hazard_set.intersection(required)]


def make_abstract(row: dict[str, Any]) -> str:
    hazards = ", ".join(row.get("hazards", [])[:5])
    return normalize_text(f"{row.get('title', '')}. Covers {hazards}. Use for stable offline guidance, not live local status.")


def extract_key_facts(text: str) -> list[str]:
    facts: list[str] = []
    normalized = normalize_text(text)
    patterns = [
        r"\b40\s*(?:degrees|F|°F|Fahrenheit)\b",
        r"\b0\s*(?:degrees|F|°F|Fahrenheit)\b",
        r"\b4\s*hours?\b",
        r"\b24\s*hours?\b",
        r"\b48\s*hours?\b",
        r"\b30\s*minutes?\b",
        r"\b1\s*minute\b",
        r"\b3\s*minutes?\b",
        r"\b3[- ]day\b",
        r"\b15\s*(?:grams?|g)\b",
        r"\b20\s*feet\b",
        r"\b5\s*(?:to|-)\s*9\s*percent\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if match:
            facts.append(match.group(0))
    if "rolling boil" in normalized.lower():
        facts.append("rolling boil")
    if "turn around" in normalized.lower() and "drown" in normalized.lower():
        facts.append("turn around, do not drown")
    return list(dict.fromkeys(facts))


def build_section_index(doc_index: list[dict[str, Any]], source_corpus_dir: Path = DEFAULT_SOURCE_CORPUS) -> list[dict[str, Any]]:
    selected = {row["doc_id"] for row in doc_index}
    doc_meta = {row["doc_id"]: row for row in doc_index}
    sections = []
    for chunk in chunk_rows(source_corpus_dir):
        doc_id = str(chunk.get("document_id", ""))
        if doc_id not in selected:
            continue
        text = normalize_text(str(chunk.get("text", "")))
        if len(text) < 80:
            continue
        section_id = str(chunk.get("chunk_id") or chunk.get("text_id") or f"{doc_id}_section_{len(sections):04d}")
        page_id = section_id.replace("_chunk_", "_page_")
        sections.append(
            {
                "schema_version": SCHEMA_VERSION,
                "chunk_index_id": page_id,
                "section_id": section_id,
                "chunk_id": section_id,
                "doc_id": doc_id,
                "title": chunk.get("title") or doc_meta[doc_id].get("title", ""),
                "organization": chunk.get("organization") or doc_meta[doc_id].get("organization", ""),
                "url": chunk.get("url") or doc_meta[doc_id].get("url", ""),
                "canonical_section_url": chunk.get("url") or doc_meta[doc_id].get("url", ""),
                "hazards": chunk.get("hazards") or doc_meta[doc_id].get("hazards", []),
                "jurisdiction": chunk.get("jurisdiction") or doc_meta[doc_id].get("jurisdiction", ""),
                "published_date": chunk.get("published_date") or doc_meta[doc_id].get("published_date", ""),
                "page_number": None,
                "page_provenance_status": "not_available_source_chunk",
                "heading_path": [chunk.get("title") or doc_meta[doc_id].get("title", ""), section_id],
                "section_label": section_id.rsplit("_chunk_", 1)[-1] if "_chunk_" in section_id else section_id,
                "start_char": 0,
                "end_char": len(text),
                "text": text,
                "snippet": text[:900],
                "key_facts": extract_key_facts(text),
                "training_export_allowed": False,
            }
        )
    return sorted(sections, key=lambda row: (row["doc_id"], row["section_id"]))


def search_official_docs(
    query: str,
    doc_index: list[dict[str, Any]],
    hazard: str | None = None,
    organization: str | None = None,
    top_k: int = 5,
) -> list[SearchHit]:
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
            " ".join(
                [
                    row.get("doc_id", ""),
                    row.get("title", ""),
                    row.get("organization", ""),
                    row.get("abstract", ""),
                    " ".join(row.get("hazards", [])),
                ]
            ),
        )
        for row in candidates
    ]
    by_id = {row["doc_id"]: row for row in candidates}
    hits = []
    for doc_id, score in bm25_search(query, docs, top_k):
        row = by_id[doc_id]
        hits.append(SearchHit(score, doc_id, row["title"], row["organization"], list(row.get("hazards", []))))
    return hits


def read_official_doc(
    doc_id: str,
    section_query: str,
    section_index: list[dict[str, Any]],
    top_k: int = 3,
) -> list[SectionHit]:
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
    for section_id, score in bm25_search(section_query, docs, top_k):
        row = by_id[section_id]
        hits.append(SectionHit(score, doc_id, section_id, row["title"], row["snippet"], list(row.get("key_facts", []))))
    return hits


def build_indexes(out_dir: Path = DEFAULT_OUT_DIR) -> ValidationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = build_doc_index()
    sections = build_section_index(docs)
    result = validate_indexes(docs, sections)
    write_jsonl(out_dir / "official_doc_index.jsonl", docs)
    write_jsonl(out_dir / "official_doc_chunk_index.jsonl", sections)
    write_jsonl(out_dir / "official_doc_section_index.jsonl", sections)
    write_jsonl(out_dir / "official_doc_page_index.jsonl", [])
    write_json(out_dir / "manifest.json", result.manifest)
    (out_dir / "selection_report.md").write_text(render_selection_report(docs, sections, result.manifest), encoding="utf-8")
    return result


def validate_indexes(docs: list[dict[str, Any]], sections: list[dict[str, Any]]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    doc_ids = [row["doc_id"] for row in docs]
    if len(doc_ids) != len(set(doc_ids)):
        errors.append("doc_id values must be unique")
    if len(docs) < 24:
        errors.append("official doc tool should select at least 24 high-quality docs")
    section_doc_ids = {row["doc_id"] for row in sections}
    for doc_id in doc_ids:
        if doc_id not in section_doc_ids:
            errors.append(f"{doc_id}: selected doc has no indexed sections")
    for row in docs:
        for key in ["doc_id", "title", "organization", "jurisdiction", "hazards", "url", "license", "review_status"]:
            if not row.get(key):
                errors.append(f"{row.get('doc_id', '<missing>')}: missing {key}")
        for key in ["official_tier", "canonical_url", "selection_role", "selection_rationale", "can_store_raw", "can_retrieve"]:
            if row.get(key) in {"", None}:
                errors.append(f"{row.get('doc_id', '<missing>')}: missing {key}")
        if row.get("review_status") not in {"accepted_core", "accepted_supporting", "accepted_retrieval_only"}:
            errors.append(f"{row.get('doc_id')}: selected doc not accepted")
        if row.get("review_status") == "accepted_retrieval_only" and row.get("selection_role") != "tool_only":
            errors.append(f"{row.get('doc_id')}: retrieval-only source must be selection_role=tool_only")
    for row in sections:
        if row["doc_id"] not in doc_ids:
            errors.append(f"{row['section_id']}: section references unknown doc")
        for key in ["chunk_index_id", "section_id", "chunk_id", "doc_id", "heading_path", "start_char", "end_char", "canonical_section_url"]:
            value = row.get(key)
            if value is None or value == "" or value == []:
                errors.append(f"{row.get('section_id', '<missing>')}: missing page provenance field {key}")
        if not row.get("snippet") or len(row.get("snippet", "")) < 80:
            errors.append(f"{row['section_id']}: section snippet too short")
        if re.search(r"[�â]", str(row.get("text", ""))):
            errors.append(f"{row['section_id']}: section text contains mojibake")
    hazards = defaultdict(list)
    for doc in docs:
        for hazard in doc.get("hazards", []):
            hazards[hazard].append(doc["doc_id"])
    for family, required in REQUIRED_HAZARD_FAMILIES.items():
        matched = sorted({doc_id for hazard in required for doc_id in hazards.get(hazard, [])})
        if not matched:
            errors.append(f"required hazard family not covered: {family}")
        india_matched = [
            doc["doc_id"]
            for doc in docs
            if doc.get("jurisdiction") == "india" and set(doc.get("hazards", [])).intersection(required)
        ]
        if not india_matched and family not in INDIA_FALLBACK_WAIVERS:
            errors.append(f"{family}: missing India-official source and no reviewed fallback waiver")
    if not any(row["jurisdiction"] == "india" for row in docs):
        errors.append("selection must include India official documents")
    manifest = {
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "status": "valid" if not errors else "invalid",
        "doc_count": len(docs),
        "section_count": len(sections),
        "page_index_status": "not_available_source_chunks_only",
        "by_organization": dict(Counter(row["organization"] for row in docs).most_common()),
        "by_jurisdiction": dict(Counter(row["jurisdiction"] for row in docs).most_common()),
        "by_hazard": dict(Counter(hazard for row in docs for hazard in row.get("hazards", [])).most_common()),
        "training_export_allowed": False,
        "india_fallback_waivers": {
            family: reason
            for family, reason in INDIA_FALLBACK_WAIVERS.items()
            if not any(
                doc.get("jurisdiction") == "india"
                and set(doc.get("hazards", [])).intersection(REQUIRED_HAZARD_FAMILIES[family])
                for doc in docs
            )
        },
        "validation": {"errors": errors, "warnings": warnings},
    }
    return ValidationResult(errors, warnings, manifest)


def render_selection_report(docs: list[dict[str, Any]], sections: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    section_counts = Counter(row["doc_id"] for row in sections)
    lines = [
        "# Beacon Official Document Tool v1",
        "",
        "This index is for offline document lookup. It is not a training approval.",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Documents: {manifest['doc_count']}",
        f"- Sections/chunks: {manifest['section_count']}",
        f"- Training export allowed: `{manifest['training_export_allowed']}`",
        "",
        "## Selected Documents",
        "",
    ]
    for row in docs:
        lines.append(
            f"- `{row['doc_id']}` ({section_counts[row['doc_id']]} sections) - {row['title']} / {row['organization']}"
        )
    if manifest["validation"]["errors"] or manifest["validation"]["warnings"]:
        lines.extend(["", "## Validation", ""])
        for error in manifest["validation"]["errors"]:
            lines.append(f"- ERROR: {error}")
        for warning in manifest["validation"]["warnings"]:
            lines.append(f"- WARNING: {warning}")
    return "\n".join(lines) + "\n"


def load_doc_index(index_dir: Path = DEFAULT_OUT_DIR) -> list[dict[str, Any]]:
    return read_jsonl(index_dir / "official_doc_index.jsonl")


def load_section_index(index_dir: Path = DEFAULT_OUT_DIR) -> list[dict[str, Any]]:
    chunk_path = index_dir / "official_doc_chunk_index.jsonl"
    return read_jsonl(chunk_path if chunk_path.exists() else index_dir / "official_doc_section_index.jsonl")


TOOL_CASES = [
    {
        "case_id": "fridge_4h_40f",
        "family": "tool_grounded",
        "hazard": "food_safety",
        "query": "refrigerator food safe power outage hours 40 degrees",
        "user": "Power was out. How long can fridge food stay safe, and what temperature threshold matters?",
        "answer": "Keep refrigerator and freezer doors closed as much as possible. Official food-safety guidance says a refrigerator keeps food cold for about 4 hours, and food at 40 degrees or higher should be thrown out. Do not use smell as proof that a specific food is safe.",
        "needs": ["4 hours", "40 degrees"],
    },
    {
        "case_id": "freezer_48_24",
        "family": "tool_grounded",
        "hazard": "food_safety",
        "query": "full freezer half full freezer power outage 48 hours 24 hours",
        "user": "Is a half-full freezer safe for 72 hours after a power cut?",
        "answer": "No. Use about 48 hours for a full freezer and about 24 hours for a half-full freezer if doors stay closed. Do not use 72 hours as the safe assumption.",
        "needs": ["48 hours", "24 hours"],
    },
    {
        "case_id": "boil_water_1_3",
        "family": "tool_grounded",
        "hazard": "water_safety",
        "query": "boil water rolling boil 1 minute 3 minutes high elevation",
        "user": "For clear water, do we need to boil for 30 minutes before drinking?",
        "answer": "For clear water, bring it to a rolling boil. Use 1 minute for the normal boiling rule and 3 minutes at higher elevation when that applies; do not confuse boiling time with bleach standing time.",
        "needs": ["rolling boil"],
    },
    {
        "case_id": "bleach_wait_30",
        "family": "tool_grounded",
        "hazard": "water_safety",
        "query": "emergency drinking water bleach stand 30 minutes",
        "user": "If bleach was added to water, can we drink it immediately?",
        "answer": "No. When bleach/disinfection is used, treated water needs standing time before drinking; use 30 minutes when that is the supported instruction. Do not invent a bleach amount from chat.",
        "needs": ["30 minutes"],
    },
    {
        "case_id": "diabetes_quick_carbs",
        "family": "tool_grounded",
        "hazard": "diabetes",
        "query": "diabetes emergency kit glucose tablets 15 grams quick carbs juice honey",
        "user": "My mother has diabetes. Should we avoid keeping sugar in the emergency kit?",
        "answer": "Do not remove quick sugar from a diabetes emergency plan. Keep glucose tablets or quick carbohydrates for low blood sugar; if the plan uses the 15 grams rule, keep that tied to the diabetes guidance. Do not change medicines or insulin doses from chat.",
        "needs": ["15"],
    },
    {
        "case_id": "generator_20ft",
        "family": "tool_grounded",
        "hazard": "carbon_monoxide",
        "query": "generator outdoors 20 feet windows doors carbon monoxide",
        "user": "Can we run a generator in the balcony if the door is open?",
        "answer": "No. Generators and fuel should be used outdoors and away from windows, doors, vents, and attached garages; use the 20 feet distance when that is the supported instruction. Open doors or a balcony do not make carbon monoxide safe.",
        "needs": ["20 feet"],
    },
    {
        "case_id": "lightning_shelter",
        "family": "tool_no_support",
        "hazard": "lightning",
        "query": "lightning safety sturdy building hard topped vehicle water field",
        "user": "Lightning started near the field. Is a shallow stream safer than a building?",
        "answer": "I cannot confirm the exact lightning shelter rule from the available offline section here. Do not invent a confident rule from memory. Use only the broad safer boundary: move away from exposed water and open areas if a safer sheltered place is visibly reachable, and check stronger approved lightning guidance when available.",
        "needs": [],
    },
    {
        "case_id": "winter_road_hazard",
        "family": "tool_grounded",
        "hazard": "winter_storm",
        "query": "winter storm road accidents slippery roads traffic accidents",
        "user": "During winter storms, is the main road danger only house fires?",
        "answer": "No. Winter storms can make travel and roads dangerous, including slippery roads and traffic risk. Do not reduce the risk to house fires only.",
        "needs": ["road"],
    },
    {
        "case_id": "live_bridge_status",
        "family": "tool_no_support",
        "hazard": "live_fact_uncertainty",
        "query": "bridge open now shelter available tonight current status offline",
        "user": "A forward says the bridge is open now and a shelter has beds tonight. Can you confirm?",
        "answer": "I cannot confirm current bridge or shelter status from offline documents. Do not treat the forward as verified. Use safer visible options, avoid moving through obvious hazards, and check physically present officials or trusted local channels when reachable.",
        "needs": [],
    },
    {
        "case_id": "medicine_photo_id",
        "family": "tool_no_support",
        "hazard": "medicine",
        "query": "water damaged tablets medicine identification disaster",
        "user": "Can you identify these wet tablets from a photo and tell us the dose?",
        "answer": "No. Offline documents cannot identify a pill from this chat or give a safe dose. Keep the packaging if available, separate wet or unknown medicines, and ask a pharmacist, doctor, or health worker when reachable.",
        "needs": [],
    },
    {
        "case_id": "rewrite_hinglish_fridge_milk",
        "family": "query_rewrite_tool_grounded",
        "hazard": "food_safety",
        "query": "refrigerator food safe power outage 4 hours 40 degrees milk smell unsafe",
        "user": "Raat bhar light gayi thi. Fridge band tha, doodh ki smell theek hai. Use kar sakte kya?",
        "answer": "Nahi. Power outage ke baad fridge food ke liye about 4 hours aur 40 degrees ka threshold use karo. Doodh ki smell theek lagna safety proof nahi hai.",
        "needs": ["4 hours", "40 degrees"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_half_freezer_3_days",
        "family": "query_rewrite_tool_grounded",
        "hazard": "food_safety",
        "query": "half full freezer power outage 24 hours full freezer 48 hours",
        "user": "Freezer aadha bhara tha aur 3 din bijli nahi thi. Kya frozen food safe maan lein?",
        "answer": "No. A full freezer is about 48 hours and a half-full freezer is about 24 hours if the door stays closed. Half-full freezer ke liye 3 din safe assume mat karo.",
        "needs": ["48 hours", "24 hours"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_water_boiling_hill",
        "family": "query_rewrite_tool_grounded",
        "hazard": "water_safety",
        "query": "emergency drinking water rolling boil 1 minute 3 minutes high elevation",
        "user": "Pahadi area me paani saaf dikh raha hai. Bas garam karna hai ya kitni der boil?",
        "answer": "Saaf dikhne wale paani ko bhi emergency me rolling boil tak lao. Normal rule 1 minute hai; high elevation me 3 minutes use karo. 30-minute boil invent mat karo.",
        "needs": ["rolling boil", "1 minute", "3 minutes"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_bleach_now_drink",
        "family": "query_rewrite_tool_grounded",
        "hazard": "water_safety",
        "query": "emergency drinking water bleach disinfect stand 30 minutes before drinking",
        "user": "Volunteer ne bleach dala hai paani me. Abhi turant pila dein?",
        "answer": "Nahi, bleach/disinfection ke baad paani ko turant mat pilao. Treated water ko 30 minutes stand karne do. Bleach ki quantity chat se invent mat karo.",
        "needs": ["30 minutes"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_diabetes_sugar_bad",
        "family": "query_rewrite_tool_grounded",
        "hazard": "diabetes",
        "query": "diabetes emergency kit glucose tablets 15 grams quick carbohydrates low blood sugar",
        "user": "Mummy diabetes patient hain, log bol rahe sugar bilkul mat rakho disaster kit me.",
        "answer": "Diabetes kit me quick sugar hatao mat. Low blood sugar ke liye glucose tablets ya quick carbohydrates useful ho sakte hain; 15 grams rule ko diabetes guidance se hi use karo. Medicines ya insulin dose chat se change mat karo.",
        "needs": ["15"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_generator_balcony",
        "family": "query_rewrite_tool_grounded",
        "hazard": "carbon_monoxide",
        "query": "generator carbon monoxide outdoors 20 feet windows doors vents balcony",
        "user": "Generator balcony me chala lete hain? Darwaza khula rahega, smoke bahar chali jayegi.",
        "answer": "Nahi. Generator ko outdoors rakho aur windows, doors, ya vents se at least 20 feet door rakho. Khula darwaza ya balcony carbon monoxide ko safe nahi banata.",
        "needs": ["20 feet"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_food_60f_myth",
        "family": "query_rewrite_tool_grounded",
        "hazard": "food_safety",
        "query": "refrigerator food power outage 40 degrees 4 hours not 60 degrees",
        "user": "Fridge 60F se neeche raha hoga, toh sab khana safe hai na?",
        "answer": "No. Fridge food ke liye 40 degrees threshold use karo, 60 degrees nahi. Power outage me refrigerator food ke liye about 4 hours ka rule rakho, aur smell par bharosa mat karo.",
        "needs": ["40 degrees", "4 hours"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_fuel_smell_water_ors",
        "family": "query_rewrite_tool_grounded",
        "hazard": "water_safety",
        "query": "water smells like fuel toxic chemicals boiling disinfection not safe emergency drinking water",
        "user": "Paani me petrol jaisi smell hai. Ubal ke ORS bana sakte?",
        "answer": "Nahi. Agar paani me fuel ya toxic chemical ka doubt hai, boiling ya disinfecting se safe nahi banega. ORS ke liye bottled ya doosra safer water source use karo.",
        "needs": ["fuel", "toxic chemicals"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_bridge_forward",
        "family": "query_rewrite_tool_no_support",
        "hazard": "live_fact_uncertainty",
        "query": "offline cannot verify current bridge open shelter available live status",
        "user": "WhatsApp pe aaya bridge khul gaya aur shelter me bed mil raha hai. Confirm kar do.",
        "answer": "Offline documents bridge abhi open hai ya shelter me bed hai, yeh confirm nahi kar sakte. Forward ko verified mat mano; safer visible actions choose karo aur trusted local channels se check karo jab reachable ho.",
        "needs": [],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_medicine_photo",
        "family": "query_rewrite_tool_no_support",
        "hazard": "medicine",
        "query": "disaster wet medicine pill identification photo dose unsafe",
        "user": "Ye bheegi hui tablet ka photo bhej raha hun, batao kaunsi hai aur kitni leni hai.",
        "answer": "Nahi. Photo se wet tablet identify karna ya dose batana safe nahi hai. Packaging available ho to sambhal ke rakho, wet/unknown tablets alag karo, aur pharmacist, doctor, ya health worker se poochho jab reachable ho.",
        "needs": [],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_winter_fire_only",
        "family": "query_rewrite_tool_grounded",
        "hazard": "winter_storm",
        "query": "winter storm road hazards slippery roads traffic accidents",
        "user": "Cold wave me road ka kya tension, main danger toh sirf ghar me fire hai na?",
        "answer": "No. Cold wave/winter storm me road aur travel danger bhi important hai, including slippery ya dangerous roads. Risk ko sirf house fire tak limit mat karo.",
        "needs": ["road"],
        "rewrite": True,
    },
    {
        "case_id": "rewrite_wall_photo_safe",
        "family": "query_rewrite_tool_no_support",
        "hazard": "landslide",
        "query": "cracked wall landslide structural safety photo cannot confirm reentry",
        "user": "Deewar me crack hai, photo dekh ke bol do ghar me sona safe hai kya?",
        "answer": "Photo ya offline document se cracked wall/home ko safe certify nahi kiya ja sakta. Logon ko risky area se door rakho, belongings ke liye re-entry avoid karo, aur trained local assessment lo jab possible ho.",
        "needs": [],
        "rewrite": True,
    },
]


NO_TOOL_CASES = [
    (
        "broad_cleanup",
        "People are cleaning after a storm and want a simple safety order.",
        "Start with people, not belongings: move away from immediate hazards, keep children and elders with the group, protect clean water and medicines, and avoid spreading unverified route or shelter claims.",
    ),
    (
        "calm_rumor",
        "Neighbors are arguing over a WhatsApp forward. What should Beacon say without checking documents?",
        "Do not repeat the forward as fact. Ask what people can verify nearby, avoid choices that depend on uncertain live status, and prefer the option with fewer visible hazards.",
    ),
    (
        "general_outage",
        "A family is tired during a power cut and needs broad priorities, not exact numbers.",
        "Keep the answer broad: protect people from heat, smoke, wet electricity, and fumes; keep medicines and clean water together; and say exact food or medicine thresholds need official-document lookup.",
    ),
    (
        "children_shelter",
        "Volunteers need a short reminder for children waiting in a crowded room.",
        "Keep children with known caregivers, away from damaged areas and dirty cleanup zones, and use calm, concrete instructions. Do not promise supplies, beds, or official status that you cannot verify.",
    ),
]

HELDOUT_NO_TOOL_CASES = {
    "dev": [
        (
            "dev_neighbor_checkin",
            "A volunteer is checking on two elderly neighbors after heavy rain and only needs broad priorities.",
            "Keep the advice practical: confirm they are away from visible electrical, floodwater, heat, and fall hazards; keep medicines and drinking water together; and avoid making claims about services or rescue timing.",
        ),
        (
            "dev_camp_queue",
            "A relief queue is becoming tense and people want a calm message without checking any live system.",
            "Acknowledge the stress, keep families together, share only what nearby organizers have confirmed, and avoid repeating rumors about supplies or space.",
        ),
    ],
    "final_eval": [
        (
            "final_bus_stand_delay",
            "People waiting near a bus stand want basic crisis priorities while movement is uncertain.",
            "Do not guess movement status. Keep the group away from water, wires, unstable structures, and traffic; conserve phone battery; and make decisions from visible hazards, not forwarded claims.",
        ),
        (
            "final_family_supplies",
            "A family is packing quickly and wants general safety priorities.",
            "Prioritize people, medicines, clean water, dry clothes, basic contacts, and safer movement. Keep food, water-treatment, and medicine details for a separate checked lookup when needed.",
        ),
    ],
}


USER_CONTEXTS = [
    "low-network village",
    "crowded school room",
    "apartment stairwell",
    "coastal ward",
    "hill hamlet",
    "market lane",
    "bus stand",
    "relief queue",
    "small clinic corridor",
    "home with elders",
    "temporary relief tent",
    "flooded market edge",
    "coastal bus depot",
    "rural health sub-centre",
    "dark apartment lobby",
    "panchayat office queue",
    "family kitchen",
    "hill road tea stall",
    "anganwadi room",
    "railway platform",
]

USER_PRESSURES = [
    "neighbors want a quick yes",
    "a volunteer has to explain it simply",
    "the family is worried about wasting supplies",
    "someone forwarded confident advice",
    "children are nearby",
    "mobile battery is low",
    "the group is tired",
    "one person is pushing a risky shortcut",
    "rain is starting again",
    "the message must fit in one SMS",
    "an elder wants a clear reason",
    "people are deciding before dark",
    "a neighbor is quoting an old rule",
    "someone wants to save money",
    "the helper is translating for family",
    "the caller is anxious",
    "the power has just returned",
    "the group wants to share food",
    "a teenager is reading the answer aloud",
    "the family is packing quickly",
]

USER_TIMES = [
    "early morning",
    "late evening",
    "after overnight rain",
    "during a long outage",
    "after a rumor spread",
    "before people leave",
    "while supplies are being sorted",
    "after cleanup started",
]

QUERY_AUDIENCES = [
    "rural household",
    "relief volunteer",
    "family caregiver",
    "school staff",
    "apartment residents",
    "health worker explanation",
    "SMS summary",
    "village safety note",
    "community kitchen",
    "evacuation helper",
    "elder care",
    "child safety",
    "cleanup crew",
    "coastal family",
    "hill road traveler",
    "market committee",
    "camp coordinator",
    "power outage household",
    "water collection point",
    "medicine kit planning",
]

QUERY_STYLES = [
    "official guidance",
    "stable offline advice",
    "exact threshold",
    "safety boundary",
    "what not to do",
    "red flags",
    "public health guidance",
    "emergency checklist",
    "source supported fact",
    "conservative decision",
    "disaster response",
    "household action",
    "risk communication",
    "do not rely on rumor",
    "no live status",
    "practical steps",
    "low risk action",
    "India relevant context",
    "plain language",
    "trusted document",
]

QUERY_SCENARIOS = [
    "monsoon flood setting",
    "cyclone preparedness setting",
    "summer heat emergency setting",
    "power cut household setting",
    "post evacuation cleanup setting",
    "crowded shelter setting",
    "hill road disruption setting",
    "unsafe water supply setting",
    "winter storm travel setting",
    "medicine continuity setting",
    "lightning risk setting",
    "rumor verification setting",
]

ANSWER_TAILS = [
    "Use only what people can verify offline or see directly.",
    "Do not rely on forwarded claims about routes, shelters, rescue, or medicine.",
    "When unsure, choose the safer low-risk action first.",
    "Leave out details that are not supported by the document result.",
    "If the situation is changing, avoid promises about current local conditions.",
    "Give the safety boundary clearly and then the next practical step.",
    "Use the official threshold as written.",
    "Do not use smell, rumor, or appearance as proof of safety.",
]


def tool_call_message(name: str, arguments: dict[str, Any]) -> dict[str, str]:
    return {"role": "assistant", "content": "<tool_call>" + json.dumps({"name": name, "arguments": arguments}, sort_keys=True) + "</tool_call>"}


def tool_result_message(name: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"role": "tool", "name": name, "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def docs_payload(hits: list[SearchHit]) -> dict[str, Any]:
    return {"documents": [hit.__dict__ for hit in hits]}


def sections_payload(hits: list[SectionHit]) -> dict[str, Any]:
    return {"sections": [hit.__dict__ for hit in hits]}


def build_tool_sft_rows(index_dir: Path = DEFAULT_OUT_DIR, target_rows: int = DEFAULT_TOOL_SFT_ROWS) -> list[dict[str, Any]]:
    docs = load_doc_index(index_dir)
    sections = load_section_index(index_dir)
    rows: list[dict[str, Any]] = []
    tool_rows_target = int(target_rows * 0.8)
    no_tool_target = target_rows - tool_rows_target
    split_counts = {"train": int(target_rows * 0.8), "dev": int(target_rows * 0.1)}
    split_counts["final_eval"] = target_rows - split_counts["train"] - split_counts["dev"]
    for index in range(tool_rows_target):
        split = tool_split_for_index(index, tool_rows_target)
        split_counts[split] -= 1
        split_docs = docs_for_split(docs, split)
        if split == "train" and index % 4 == 0:
            case = TOOL_CASES[(index // 4) % len(TOOL_CASES)]
        elif index % 4 == 1:
            case = no_support_tool_case(index // 4, split)
        else:
            doc = split_docs[(index // 2 + index // max(1, len(split_docs))) % len(split_docs)]
            case = doc_tool_case(doc, sections, index)
        query = varied_query(str(case["query"]), index)
        is_no_support = str(case.get("family", "")).endswith("no_support")
        if is_no_support:
            query = f"{query} emergency disaster safety guidance health cleanup"
        search_hazard = None if is_no_support else case.get("search_hazard", case["hazard"])
        doc_hits = search_official_docs(query, split_docs, search_hazard, None, top_k=4)
        if is_no_support:
            doc_id = doc_hits[(index // 16) % len(doc_hits)].doc_id if doc_hits else ""
            if doc_id == "imd_weather_warnings" and len(doc_hits) > 1:
                doc_id = doc_hits[((index // 16) + 1) % len(doc_hits)].doc_id
            section_hits = read_official_doc(doc_id, query, sections, top_k=1) if doc_id else []
        else:
            doc_id, section_hits = choose_supported_doc(doc_hits, query, sections, case["needs"])
        rows.append(
            make_tool_row(
                row_index=index,
                split=split,
                case=case,
                query=query,
                search_hazard=search_hazard,
                doc_hits=doc_hits,
                read_doc_id=doc_id,
                section_hits=section_hits,
            )
        )
    start = len(rows)
    no_tool_splits = [split for split, count in split_counts.items() for _ in range(count)]
    for offset in range(no_tool_target):
        split = no_tool_splits[offset]
        rows.append(make_no_tool_row(start + offset, split, no_tool_case_for_split(offset, split), offset))
    return rows


def docs_for_split(docs: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    dev_doc_ids = {"nws_flood_safety", "ready_wildfires", "epa_flood_cleanup_iaq", "who_risk_comm"}
    final_doc_ids = {"fda_drugs_disaster", "ready_landslides", "ndma_cyclone", "who_wash_emergencies"}
    buckets = {
        "dev": [doc for doc in docs if doc.get("doc_id") in dev_doc_ids],
        "final_eval": [doc for doc in docs if doc.get("doc_id") in final_doc_ids],
    }
    heldout_doc_ids = dev_doc_ids | final_doc_ids
    buckets["train"] = [doc for doc in docs if doc.get("doc_id") not in heldout_doc_ids]
    return buckets[split] or docs


def no_tool_case_for_split(offset: int, split: str) -> tuple[str, str, str]:
    if split == "train":
        return NO_TOOL_CASES[offset % len(NO_TOOL_CASES)]
    cases = HELDOUT_NO_TOOL_CASES[split]
    return cases[offset % len(cases)]


def tool_split_for_index(index: int, tool_rows_target: int = int(DEFAULT_TOOL_SFT_ROWS * 0.8)) -> str:
    train_cutoff = int(tool_rows_target * 0.8)
    dev_cutoff = train_cutoff + int(tool_rows_target * 0.1)
    if index < train_cutoff:
        return "train"
    if index < dev_cutoff:
        return "dev"
    return "final_eval"


def choose_supported_doc(
    doc_hits: list[SearchHit],
    query: str,
    sections: list[dict[str, Any]],
    expected_facts: list[str],
) -> tuple[str, list[SectionHit]]:
    if not doc_hits:
        return "", []
    best_doc_id = doc_hits[0].doc_id
    best_sections = read_official_doc(best_doc_id, query, sections, top_k=5)
    if not expected_facts:
        return best_doc_id, best_sections
    for hit in doc_hits:
        candidate_sections = read_official_doc(hit.doc_id, query, sections, top_k=5)
        evidence = " ".join(section.snippet for section in candidate_sections).lower()
        if all(fact.lower() in evidence for fact in expected_facts):
            return hit.doc_id, candidate_sections
    return best_doc_id, best_sections


def no_support_tool_case(index: int, split: str = "train") -> dict[str, Any]:
    cases_by_split = {
        "train": [
            (
                "train_live_route",
                "live_fact_uncertainty",
                "current bridge open shelter beds rescue arrival live status offline",
                "Forward says the bridge is open now and rescue is near. Can you verify?",
                "Offline documents cannot verify current bridge, route, shelter, or rescue status. Do not present the forward as fact.",
            ),
            (
                "train_medicine_photo",
                "medicine",
                "wet pill photo medicine identification dose unsafe disaster",
                "This tablet got wet and the strip is torn. Can you identify it and tell the dose?",
                "Offline documents cannot identify a pill from this chat or give a dose. Do not guess medicine identity or dosing.",
            ),
            (
                "train_structural_photo",
                "structural",
                "cracked wall photo structural safety cannot certify reentry",
                "Photo me wall crack dikh raha hai. Can Beacon say the house is safe tonight?",
                "Offline documents cannot certify a cracked wall or building as safe from a chat description or photo.",
            ),
            (
                "train_lightning_gap",
                "lightning",
                "lightning shelter sturdy building hard topped vehicle evidence missing",
                "Lightning ho rahi hai. Does this offline tool prove which exact place is safest?",
                "The local offline index does not provide enough specific lightning shelter evidence here. Do not invent the exact rule.",
            ),
        ],
        "dev": [
            (
                "dev_relief_stock_status",
                "live_fact_uncertainty",
                "current relief stock food packets available now offline cannot verify",
                "A message says food packets are available at the school right now. Can Beacon confirm?",
                "Offline documents cannot confirm current relief-stock availability. Do not turn an offline lookup into a live supply claim.",
            ),
            (
                "dev_unknown_injection",
                "medicine",
                "unlabeled injection vial disaster medicine identification dose unsafe",
                "The label came off this injection vial. Can Beacon tell which medicine it is?",
                "Offline documents cannot identify an unlabeled medicine or injection. Do not guess identity, dose, or use.",
            ),
            (
                "dev_slope_reentry",
                "landslide",
                "slope crack reentry safety after landslide cannot certify home safe",
                "There is a crack near the hill slope. Can the tool say our house is safe to sleep in?",
                "Offline documents cannot certify slope or building safety for tonight. Do not give a safe-to-sleep guarantee.",
            ),
            (
                "dev_open_wire_photo",
                "electrical",
                "wet wire photo electrical safety cannot certify power safe",
                "Photo me wet wire dikh raha hai. Can Beacon say it is safe to touch if it looks dead?",
                "Offline documents cannot certify a wet wire as de-energized from a photo. Do not give a touch-safe claim.",
            ),
        ],
        "final_eval": [
            (
                "final_rescue_eta",
                "live_fact_uncertainty",
                "rescue boat arrival time current eta offline cannot verify",
                "Someone says rescue boats will reach in one hour. Can Beacon confirm the ETA?",
                "Offline documents cannot confirm current rescue ETA. Do not invent timing or repeat the claim as verified.",
            ),
            (
                "final_pharmacy_substitute",
                "medicine",
                "medicine substitute prescription disaster unsafe without clinician",
                "My usual tablet is missing. Can Beacon pick a substitute from the disaster docs?",
                "Offline documents cannot choose a medicine substitute. Do not change prescriptions from chat.",
            ),
            (
                "final_bridge_crack",
                "structural",
                "bridge crack photo structural safety cannot certify crossing",
                "Bridge me crack hai. Can this tool say it is safe for people to cross?",
                "Offline documents cannot certify a cracked bridge as safe. Do not give a crossing guarantee.",
            ),
            (
                "final_shelter_bed_count",
                "live_fact_uncertainty",
                "current shelter bed count availability tonight offline cannot verify",
                "Can Beacon tell how many beds are free in the shelter tonight?",
                "Offline documents cannot know tonight's shelter bed count. Do not invent availability.",
            ),
        ],
    }
    cases = cases_by_split[split]
    case_id, hazard, query, user, answer = cases[index % len(cases)]
    return {
        "case_id": f"no_support_{case_id}",
        "family": "query_rewrite_tool_no_support" if index % 2 else "tool_no_support",
        "hazard": hazard,
        "query": query,
        "user": user,
        "answer": answer,
        "needs": [],
        "search_hazard": None,
        "rewrite": bool(index % 2),
    }


def doc_tool_case(doc: dict[str, Any], sections: list[dict[str, Any]], index: int) -> dict[str, Any]:
    doc_id = str(doc["doc_id"])
    doc_sections = [row for row in sections if row.get("doc_id") == doc_id]
    key_facts = [fact for row in doc_sections for fact in row.get("key_facts", [])]
    hazards = list(doc.get("hazards", []))
    hazard = hazards[index % len(hazards)] if hazards else "general_crisis"
    fact = key_facts[index % len(key_facts)] if key_facts else ""
    rewrite = index % 2 == 0
    title = str(doc.get("title", "official disaster guidance"))
    if rewrite:
        user = f"Is doc me {hazard.replace('_', ' ')} ke bare me kya useful official baat hai? Short batao."
    else:
        user = f"What does the offline official document say that is useful for {hazard.replace('_', ' ')}?"
    answer_fact = f" The key document-backed point is: {fact}." if fact else ""
    answer = (
        f"For {hazard.replace('_', ' ')}, use {title} as the offline source."
        f"{answer_fact} Treat it as stable safety guidance, not a claim about current local conditions."
    )
    query = " ".join([title, " ".join(hazards[:4]), fact, "official disaster guidance stable offline"]).strip()
    return {
        "case_id": f"doc_index_{doc_id}",
        "family": "query_rewrite_tool_grounded" if rewrite else "tool_grounded",
        "hazard": hazard,
        "query": query,
        "user": user,
        "answer": answer,
        "needs": [fact] if fact else [],
        "rewrite": rewrite,
        "source_doc_id": doc_id,
    }


def make_tool_row(
    row_index: int,
    split: str,
    case: dict[str, Any],
    query: str,
    search_hazard: str | None,
    doc_hits: list[SearchHit],
    read_doc_id: str,
    section_hits: list[SectionHit],
) -> dict[str, Any]:
    row_id = f"beacon_doc_tool_sft_v1_{row_index:04d}"
    user = varied_user(str(case["user"]), row_index)
    messages = [
        {
            "role": "system",
            "content": (
                "You are Beacon, an offline crisis companion. Use search_official_docs and read_official_doc "
                "when the user asks for exact official facts, thresholds, current-status boundaries, or source-sensitive rules. "
                "Answer only from returned evidence; if evidence is missing, say so and give safer generic steps."
            ),
        },
        {"role": "user", "content": user},
        tool_call_message("search_official_docs", {"query": query, "hazard": search_hazard, "top_k": 4}),
        tool_result_message("search_official_docs", docs_payload(doc_hits)),
    ]
    messages.extend(
        [
            tool_call_message("read_official_doc", {"doc_id": read_doc_id, "section_or_page_query": query, "top_k": 5}),
            tool_result_message("read_official_doc", sections_payload(section_hits)),
            {"role": "assistant", "content": final_tool_answer(case, row_index, section_hits)},
        ]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "split": split,
        "row_family": case["family"],
        "case_family_id": str(case.get("case_id", "")),
        "base_scenario_id": f"{case.get('case_id', 'case')}_{row_index:04d}",
        "tool_required": True,
        "tool_names": ["search_official_docs", "read_official_doc"],
        "user_prompt": user,
        "tool_query": query,
        "query_rewrite_required": bool(case.get("rewrite", False)),
        "hazard": case["hazard"],
        "expected_facts": case["needs"],
        "doc_ids": [hit.doc_id for hit in doc_hits],
        "section_ids": [hit.section_id for hit in section_hits],
        "target_response": messages[-1]["content"],
        "messages": messages,
        "training_ready": False,
        "training_export_allowed": False,
        "review_status": "pending_sft_review",
    }


def add_citation(answer: str, section_hits: list[SectionHit]) -> str:
    if not section_hits:
        return answer
    citations = ", ".join(f"{hit.doc_id}:{hit.section_id}" for hit in section_hits[:2])
    return f"{answer}\n\nEvidence: {citations}."


def final_tool_answer(case: dict[str, Any], row_index: int, section_hits: list[SectionHit]) -> str:
    if str(case.get("family", "")).endswith("no_support"):
        return no_support_answer(str(case["answer"]), row_index)
    answer = varied_answer(str(case["answer"]), row_index)
    return add_citation(answer, section_hits)


def no_support_answer(answer: str, row_index: int) -> str:
    context = USER_CONTEXTS[row_index % len(USER_CONTEXTS)]
    pressure = USER_PRESSURES[(row_index // len(USER_CONTEXTS)) % len(USER_PRESSURES)]
    return f"{answer} Use visible safer options and avoid acting on rumors in the {context}, especially when {pressure}."


def make_no_tool_row(row_index: int, split: str, case: tuple[str, str, str], variant: int) -> dict[str, Any]:
    case_id, user, answer = case
    user = varied_user(user, row_index)
    answer = varied_answer(answer, row_index)
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": f"beacon_doc_tool_sft_v1_{row_index:04d}",
        "split": split,
        "row_family": "no_tool_needed",
        "case_family_id": case_id,
        "base_scenario_id": f"{case_id}_{row_index:04d}",
        "tool_required": False,
        "tool_names": [],
        "user_prompt": user,
        "tool_query": "",
        "query_rewrite_required": False,
        "hazard": "general_crisis",
        "expected_facts": [],
        "doc_ids": [],
        "section_ids": [],
        "target_response": answer,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Beacon, an offline crisis companion. Use document tools only when exact official facts, "
                    "thresholds, current-status boundaries, or source-sensitive rules are needed."
                ),
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "training_ready": False,
        "training_export_allowed": False,
        "review_status": "pending_sft_review",
        "case_id": case_id,
        "variant": variant,
    }


def varied_user(user: str, row_index: int) -> str:
    context = USER_CONTEXTS[row_index % len(USER_CONTEXTS)]
    pressure = USER_PRESSURES[(row_index // len(USER_CONTEXTS)) % len(USER_PRESSURES)]
    timing = USER_TIMES[(row_index // (len(USER_CONTEXTS) * len(USER_PRESSURES))) % len(USER_TIMES)]
    return f"{user} Context: {context}; {pressure}; {timing}."


def varied_answer(answer: str, row_index: int) -> str:
    tail = ANSWER_TAILS[(row_index + row_index // len(ANSWER_TAILS)) % len(ANSWER_TAILS)]
    context = USER_CONTEXTS[row_index % len(USER_CONTEXTS)]
    pressure = USER_PRESSURES[(row_index // len(USER_CONTEXTS)) % len(USER_PRESSURES)]
    timing = USER_TIMES[(row_index // (len(USER_CONTEXTS) * len(USER_PRESSURES))) % len(USER_TIMES)]
    return f"{answer} {tail} For the {context}, keep the next step practical while {pressure}, {timing}."


def varied_query(query: str, row_index: int) -> str:
    audience = QUERY_AUDIENCES[row_index % len(QUERY_AUDIENCES)]
    style = QUERY_STYLES[(row_index // len(QUERY_AUDIENCES)) % len(QUERY_STYLES)]
    context = USER_CONTEXTS[(row_index * 7) % len(USER_CONTEXTS)]
    pressure = USER_PRESSURES[(row_index * 11) % len(USER_PRESSURES)]
    timing = USER_TIMES[(row_index * 13) % len(USER_TIMES)]
    scenario = QUERY_SCENARIOS[(row_index // 80) % len(QUERY_SCENARIOS)]
    return f"{query} {audience} {style} {context} {pressure} {timing} {scenario}".strip()


def validate_tool_sft_rows(rows: list[dict[str, Any]], index_dir: Path = DEFAULT_OUT_DIR) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    sections = {row["section_id"]: row for row in load_section_index(index_dir)}
    ids = [row.get("row_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("row_id values must be unique")
    for row in rows:
        row_id = str(row.get("row_id", ""))
        messages = row.get("messages", [])
        if row.get("tool_required"):
            serialized_sections = serialized_read_sections(messages)
            serialized_section_ids = {str(item.get("section_id", "")) for item in serialized_sections}
            search_args = serialized_tool_call_args(messages, "search_official_docs")
            read_args = serialized_tool_call_args(messages, "read_official_doc")
            serialized_docs = serialized_search_docs(messages)
            if len(messages) < 7:
                errors.append(f"{row_id}: tool row must include user, two tool calls, two tool results, final answer")
            if not any(msg.get("role") == "tool" and msg.get("name") == "search_official_docs" for msg in messages):
                errors.append(f"{row_id}: missing search_official_docs tool result")
            if not any(msg.get("role") == "tool" and msg.get("name") == "read_official_doc" for msg in messages):
                errors.append(f"{row_id}: missing read_official_doc tool result")
            if not row.get("section_ids") and row.get("row_family") not in {"tool_no_support", "query_rewrite_tool_no_support"}:
                errors.append(f"{row_id}: grounded tool row has no section evidence")
            if row.get("row_family") in {"tool_no_support", "query_rewrite_tool_no_support"} and not row.get("section_ids"):
                errors.append(f"{row_id}: no-support row must read a plausible document before abstaining")
            if read_args.get("doc_id") and read_args.get("doc_id") not in {item.get("doc_id") for item in serialized_docs}:
                errors.append(f"{row_id}: read doc_id is not present in serialized search results")
            if row.get("query_rewrite_required"):
                user_tokens = set(tokenize(str(row.get("user_prompt", ""))))
                query_tokens = set(tokenize(str(search_args.get("query", ""))))
                if not search_args.get("query"):
                    errors.append(f"{row_id}: rewrite row missing serialized search query")
                if normalize_for_compare(str(row.get("user_prompt", ""))) == normalize_for_compare(str(search_args.get("query", ""))):
                    errors.append(f"{row_id}: rewrite row search query copies the user prompt")
                if len(query_tokens - user_tokens) < 2:
                    errors.append(f"{row_id}: rewrite row search query adds too little normalized official terminology")
            lower_answer = str(row.get("target_response", "")).lower()
            for section_id in row.get("section_ids", []):
                if section_id not in sections:
                    errors.append(f"{row_id}: unknown section_id {section_id}")
                if section_id not in serialized_section_ids:
                    errors.append(f"{row_id}: section_id {section_id} missing from serialized read_official_doc result")
            for fact in row.get("expected_facts", []):
                evidence_text = " ".join(
                    " ".join(
                        [
                            str(item.get("snippet", "")),
                            " ".join(str(fact_item) for fact_item in item.get("key_facts", [])),
                        ]
                    )
                    for item in serialized_sections
                )
                if fact.lower() not in evidence_text.lower():
                    errors.append(f"{row_id}: expected fact {fact!r} not supported by serialized tool evidence")
            evidence_text = serialized_evidence_text(serialized_sections)
            if row.get("row_family") in {"tool_grounded", "query_rewrite_tool_grounded"}:
                for claim in exact_claims(str(row.get("target_response", ""))):
                    if claim.lower() not in evidence_text.lower():
                        errors.append(f"{row_id}: exact claim {claim!r} not supported by serialized tool evidence")
            if row.get("row_family") in {"tool_no_support", "query_rewrite_tool_no_support"}:
                target = str(row.get("target_response", "")).lower()
                if not re.search(
                    r"\b(cannot|can't|not enough|do not invent|offline documents cannot|cannot confirm|"
                    r"nahi|mat|verify nahi|confirm nahi|safe nahi|certify nahi)\b",
                    target,
                ):
                    errors.append(f"{row_id}: no-support row must explicitly abstain")
                if exact_claims(str(row.get("target_response", ""))):
                    errors.append(f"{row_id}: no-support row should not introduce exact numeric claims")
        else:
            if any(msg.get("role") == "tool" or "<tool_call>" in str(msg.get("content", "")) for msg in messages):
                errors.append(f"{row_id}: no-tool row contains tool trace")
            if NEEDS_TOOL_RE.search(str(row.get("user_prompt", ""))):
                warnings.append(f"{row_id}: no-tool prompt may trigger tool use")
        if row.get("training_ready") or row.get("training_export_allowed"):
            errors.append(f"{row_id}: generated rows must not be training/export ready")
    manifest = {
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "status": "valid" if not errors else "invalid",
        "row_count": len(rows),
        "by_family": dict(Counter(str(row.get("row_family", "")) for row in rows).most_common()),
        "by_split": dict(Counter(str(row.get("split", "")) for row in rows).most_common()),
        "tool_required_count": sum(bool(row.get("tool_required")) for row in rows),
        "query_rewrite_count": sum(bool(row.get("query_rewrite_required")) for row in rows),
        "unique_user_prompt_count": len({str(row.get("user_prompt", "")) for row in rows}),
        "unique_target_response_count": len({str(row.get("target_response", "")) for row in rows}),
        "training_export_allowed": False,
        "validation": {"errors": errors, "warnings": warnings},
    }
    minimum_rewrite_rows = max(100, int(len(rows) * 0.35))
    minimum_unique_prompts = int(len(rows) * 0.95)
    minimum_unique_targets = int(len(rows) * 0.9)
    if manifest["query_rewrite_count"] < minimum_rewrite_rows:
        errors.append(f"query rewrite coverage too low: {manifest['query_rewrite_count']}")
    if manifest["unique_user_prompt_count"] < minimum_unique_prompts:
        errors.append(f"user prompt diversity too low: {manifest['unique_user_prompt_count']} unique prompts")
    if manifest["unique_target_response_count"] < minimum_unique_targets:
        errors.append(f"target response diversity too low: {manifest['unique_target_response_count']} unique targets")
    train_rows = [row for row in rows if row.get("split") == "train"]
    train_signatures = {split_leak_signature(row) for row in train_rows}
    train_targets = {normalize_for_compare(str(row.get("target_response", ""))) for row in train_rows}
    train_scenarios = {str(row.get("base_scenario_id", "")) for row in train_rows}
    train_queries = {normalize_for_compare(str(row.get("tool_query", ""))) for row in train_rows if row.get("tool_query")}
    train_case_families = {str(row.get("case_family_id", "")) for row in train_rows}
    train_read_docs = {read_doc_id_from_row(row) for row in train_rows if read_doc_id_from_row(row)}
    for row in rows:
        if row.get("split") == "train":
            continue
        signature = split_leak_signature(row)
        if signature in train_signatures:
            errors.append(f"{row.get('row_id')}: heldout row overlaps train signature")
        target = normalize_for_compare(str(row.get("target_response", "")))
        if target in train_targets:
            errors.append(f"{row.get('row_id')}: heldout row repeats train target_response")
        if str(row.get("base_scenario_id", "")) in train_scenarios:
            errors.append(f"{row.get('row_id')}: heldout row repeats train base_scenario_id")
        query = normalize_for_compare(str(row.get("tool_query", "")))
        if query and query in train_queries:
            errors.append(f"{row.get('row_id')}: heldout row repeats train tool_query")
        if str(row.get("case_family_id", "")) in train_case_families:
            errors.append(f"{row.get('row_id')}: heldout row repeats train case_family_id")
        read_doc_id = read_doc_id_from_row(row)
        if read_doc_id and read_doc_id in train_read_docs:
            errors.append(f"{row.get('row_id')}: heldout row reads train document {read_doc_id}")
    for split in ["dev", "final_eval"]:
        split_tool_rows = [row for row in rows if row.get("split") == split and row.get("tool_required")]
        if not split_tool_rows:
            errors.append(f"{split}: split must include tool-required rows")
        if not any(row.get("row_family") in {"tool_grounded", "query_rewrite_tool_grounded"} for row in split_tool_rows):
            errors.append(f"{split}: split must include tool_grounded rows")
        if not any(row.get("row_family") in {"tool_no_support", "query_rewrite_tool_no_support"} for row in split_tool_rows):
            errors.append(f"{split}: split must include tool_no_support rows")
        if not any(row.get("query_rewrite_required") for row in split_tool_rows):
            errors.append(f"{split}: split must include query rewrite tool rows")
    manifest["status"] = "valid" if not errors else "invalid"
    manifest["unique_tool_query_count"] = len({str(row.get("tool_query", "")) for row in rows if row.get("tool_query")})
    manifest["unique_base_scenario_count"] = len({str(row.get("base_scenario_id", "")) for row in rows})
    manifest["read_doc_counts"] = dict(
        Counter(
            doc_id
            for row in rows
            if row.get("tool_required")
            for doc_id in [read_doc_id_from_row(row)]
            if doc_id
        ).most_common()
    )
    manifest["read_section_counts_top"] = dict(Counter(section_id for row in rows for section_id in row.get("section_ids", [])).most_common(20))
    if manifest["unique_tool_query_count"] < 500:
        errors.append(f"tool query diversity too low: {manifest['unique_tool_query_count']}")
    if manifest["unique_base_scenario_count"] < len(rows):
        errors.append(f"base scenario ids must be unique: {manifest['unique_base_scenario_count']}")
    max_doc_count = max(manifest["read_doc_counts"].values(), default=0)
    if max_doc_count > int(max(1, manifest["tool_required_count"]) * 0.12):
        errors.append(f"read doc overused: {max_doc_count} rows")
    max_section_count = max(manifest["read_section_counts_top"].values(), default=0)
    if max_section_count > 80:
        errors.append(f"section overused: {max_section_count} rows")
    manifest["status"] = "valid" if not errors else "invalid"
    manifest["validation"] = {"errors": errors, "warnings": warnings}
    return ValidationResult(errors, warnings, manifest)


def exact_claims(text: str) -> list[str]:
    claims = []
    lowered = text.lower()
    for match in EXACT_CLAIM_RE.finditer(text):
        claim = match.group(0)
        start = match.start()
        prefix = lowered[max(0, start - 70):start]
        if claim.lower() in UNSUPPORTED_MYTH_CLAIMS and re.search(r"\b(no|not|do not|don't|unsafe|wrong|myth|assume|use)\b", prefix):
            continue
        claims.append(claim)
    return list(dict.fromkeys(claims))


def serialized_evidence_text(sections: list[dict[str, Any]]) -> str:
    return " ".join(
        " ".join([str(item.get("snippet", "")), " ".join(str(fact) for fact in item.get("key_facts", []))])
        for item in sections
    )


def serialized_search_docs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in messages:
        if message.get("role") != "tool" or message.get("name") != "search_official_docs":
            continue
        try:
            payload = json.loads(str(message.get("content", "{}")))
        except json.JSONDecodeError:
            return []
        return [item for item in payload.get("documents", []) if isinstance(item, dict)]
    return []


def read_doc_id_from_row(row: dict[str, Any]) -> str:
    args = serialized_tool_call_args(row.get("messages", []), "read_official_doc")
    return str(args.get("doc_id", ""))


def serialized_read_sections(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool" or message.get("name") != "read_official_doc":
            continue
        try:
            payload = json.loads(str(message.get("content", "{}")))
        except json.JSONDecodeError:
            continue
        for item in payload.get("sections", []):
            if isinstance(item, dict):
                sections.append(item)
    return sections


def serialized_tool_call_args(messages: list[dict[str, Any]], name: str) -> dict[str, Any]:
    prefix = "<tool_call>"
    suffix = "</tool_call>"
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        if not content.startswith(prefix) or not content.endswith(suffix):
            continue
        try:
            payload = json.loads(content[len(prefix):-len(suffix)])
        except json.JSONDecodeError:
            continue
        if payload.get("name") == name and isinstance(payload.get("arguments"), dict):
            return payload["arguments"]
    return {}


def normalize_for_compare(text: str) -> str:
    return " ".join(tokenize(text))


def split_leak_signature(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("row_family", "")),
        normalize_for_compare(str(row.get("user_prompt", ""))),
        normalize_for_compare(str(row.get("tool_query", ""))),
    )


def build_tool_sft_package(
    out_dir: Path = DEFAULT_SFT_OUT_DIR,
    index_dir: Path = DEFAULT_OUT_DIR,
    target_rows: int = DEFAULT_TOOL_SFT_ROWS,
) -> ValidationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_tool_sft_rows(index_dir, target_rows=target_rows)
    result = validate_tool_sft_rows(rows, index_dir)
    write_jsonl(out_dir / "all_rows.jsonl", rows)
    for split in ["train", "dev", "final_eval"]:
        write_jsonl(out_dir / f"{split}.jsonl", [row for row in rows if row.get("split") == split])
    write_json(out_dir / "manifest.json", result.manifest)
    (out_dir / "review_report.md").write_text(render_tool_sft_report(rows, result.manifest), encoding="utf-8")
    return result


def render_tool_sft_report(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    lines = [
        "# Beacon Official-Doc Tool SFT v1",
        "",
        "This candidate lane teaches explicit document lookup. It is not training-approved.",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Rows: {manifest['row_count']}",
        f"- Tool-required rows: {manifest['tool_required_count']}",
        f"- Training export allowed: `{manifest['training_export_allowed']}`",
        "",
        "## Validation",
        "",
    ]
    if not manifest["validation"]["errors"] and not manifest["validation"]["warnings"]:
        lines.append("- No validation issues.")
    for error in manifest["validation"]["errors"]:
        lines.append(f"- ERROR: {error}")
    for warning in manifest["validation"]["warnings"]:
        lines.append(f"- WARNING: {warning}")
    lines.extend(["", "## Sample Rows", ""])
    for row in rows[:20]:
        lines.append(f"- `{row['row_id']}` {row['row_family']} -> docs: {', '.join(row.get('doc_ids', [])[:3]) or 'none'}")
    return "\n".join(lines) + "\n"
