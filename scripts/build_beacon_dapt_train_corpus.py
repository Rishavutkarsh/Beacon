from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
import zipfile
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BUILDER = ROOT / "scripts" / "build_beacon_dapt_source_corpus.py"
OUT_DIR = ROOT / "data" / "dapt_corpus" / "beacon_crisis_v1"
RAW_DIR = OUT_DIR / "raw"
EXTRACTED_DIR = OUT_DIR / "extracted"
REJECTED_DIR = OUT_DIR / "rejected"

USER_AGENT = "BeaconDaptCorpusBuilder/1.0 (+https://github.com/Rishavutkarsh/Beacon)"
TIMEOUT_SECONDS = 35
PDF_MAX_PAGES = 650
ZIP_MAX_PDFS = 12
MIN_EXTRACTED_CHARS = 900
MIN_TRAIN_TOKENS = 2_000_000
PREFERRED_TRAIN_TOKENS = 5_000_000
DEV_DOC_RATIO = 0.08
MAX_DISCOVERED_DOCS = 520
RELIEFWEB_QUERY_LIMIT = 60
DIRECT_TEXTS: dict[str, str] = {}


def load_source_builder() -> Any:
    spec = importlib.util.spec_from_file_location("beacon_source_builder", SOURCE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source_builder = load_source_builder()


@dataclass
class SourceCard:
    source_id: str
    organization: str
    domain: str
    tier: int
    jurisdiction: str
    source_type: str
    notes: str = ""


@dataclass
class DocumentCandidate:
    document_id: str
    source_id: str
    url: str
    title: str
    organization: str
    jurisdiction: str
    language: str
    hazards: list[str]
    source_type: str
    document_type: str = "html"
    license: str = "recorded_metadata_not_blocking_hackathon_dapt"
    discovered_from: str = "seed"


@dataclass
class DocumentCard:
    document_id: str
    source_id: str
    url: str
    title: str
    organization: str
    jurisdiction: str
    language: str
    hazards: list[str]
    source_type: str
    document_type: str
    license: str
    discovered_from: str
    retrieved_at: str
    accepted: bool
    reject_reason: str = ""
    raw_sha256: str = ""
    extracted_sha256: str = ""
    raw_path: str = ""
    extracted_path: str = ""
    extraction_status: str = "not_attempted"
    text_chars: int = 0
    estimated_tokens: int = 0


def utc_now() -> str:
    return datetime.now(UTC).date().isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def safe_remove_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except PermissionError:
        for child in sorted(path.rglob("*"), reverse=True):
            try:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            except (OSError, PermissionError):
                continue


def write_bytes_allow_locked_existing(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(content)
    except PermissionError:
        if not path.exists():
            raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_cards() -> list[SourceCard]:
    return [
        SourceCard("ndma", "National Disaster Management Authority, India", "ndma.gov.in", 1, "india", "india_official"),
        SourceCard("imd", "India Meteorological Department", "imd.gov.in", 1, "india", "india_official"),
        SourceCard("mohfw", "Ministry of Health and Family Welfare, India", "mohfw.gov.in", 1, "india", "india_official"),
        SourceCard("fssai", "Food Safety and Standards Authority of India", "fssai.gov.in", 1, "india", "india_official"),
        SourceCard("cdc", "Centers for Disease Control and Prevention", "cdc.gov", 2, "us_official", "global_official"),
        SourceCard("ready", "Ready.gov / FEMA", "ready.gov", 2, "us_official", "global_official"),
        SourceCard("fema", "Federal Emergency Management Agency", "fema.gov", 2, "us_official", "global_official"),
        SourceCard("nws", "National Weather Service", "weather.gov", 2, "us_official", "global_official"),
        SourceCard("epa", "US Environmental Protection Agency", "epa.gov", 2, "us_official", "global_official"),
        SourceCard("fda", "US Food and Drug Administration", "fda.gov", 2, "us_official", "global_official"),
        SourceCard("usfa", "US Fire Administration", "usfa.fema.gov", 2, "us_official", "global_official"),
        SourceCard("who", "World Health Organization", "who.int", 2, "global", "global_official"),
        SourceCard("unicef", "UNICEF", "unicef.org", 2, "global", "global_official"),
        SourceCard("ifrc", "International Federation of Red Cross and Red Crescent Societies", "ifrc.org", 2, "global", "ngo_reputable"),
        SourceCard("sphere", "Sphere Association", "spherestandards.org", 2, "global", "ngo_reputable"),
        SourceCard("redcross", "American Red Cross", "redcross.org", 3, "us/global-applicable", "ngo_reputable"),
        SourceCard("reliefweb", "ReliefWeb / OCHA indexed humanitarian guidance", "reliefweb.int", 3, "global", "humanitarian_reputable"),
    ]


SOURCE_BY_ID = {card.source_id: card for card in source_cards()}


HAZARD_KEYWORDS = {
    "wash": ["wash", "sanitation", "hygiene", "water", "diarrhoea", "diarrhea", "cholera"],
    "food_safety": ["food", "formula", "refrigerator", "freezer", "fda"],
    "power_outage": ["power", "outage", "blackout", "generator"],
    "carbon_monoxide": ["carbon monoxide", "co poisoning", "generator", "charcoal"],
    "electrical": ["electrical", "downed line", "power line", "electrocution"],
    "medicine": ["medicine", "medication", "insulin", "diabetes", "drug"],
    "wounds": ["wound", "injury", "tetanus", "first aid"],
    "route_uncertainty": ["evacuation", "route", "warning", "alert", "shelter"],
    "shelter": ["shelter", "crowding", "camp", "settlement"],
    "heat_cold": ["heat", "heatwave", "winter", "cold", "hypothermia"],
    "lightning": ["lightning", "thunderstorm"],
    "flood_cyclone_landslide": ["flood", "cyclone", "hurricane", "landslide", "storm", "tsunami"],
    "fire_lpg_chemical": ["fire", "wildfire", "chemical", "hazardous", "gas", "lpg"],
    "accessibility": ["disability", "older adults", "children", "pregnant", "accessibility"],
    "misinformation": ["rumor", "rumour", "misinformation", "risk communication", "community engagement"],
    "response_management": ["incident command", "nims", "national response framework", "emergency management", "cert"],
}


RELEVANT_TERMS = sorted({term for terms in HAZARD_KEYWORDS.values() for term in terms} | {
    "disaster", "emergency", "preparedness", "response", "recovery", "public health", "natural hazards",
    "coursematerials", "courseoverview", "student manual", "participant manual", "incident command", "nims", "cert",
})
BAD_URL_TERMS = [
    "/news/", "/press-release/", "/events/", "/careers", "/donate", "/privacy", "/contact", "/about-us",
    "facebook", "twitter", "linkedin", "youtube", "instagram", "login", "subscribe", "photo", ".jpg", ".png", ".mp4",
]
LIVE_TERMS = ["current alerts", "incident status", "road closure", "shelter availability", "today's update"]


def candidate(
    source_id: str,
    url: str,
    title: str,
    hazards: list[str] | None = None,
    language: str = "en",
    document_type: str = "html",
    discovered_from: str = "seed",
) -> DocumentCandidate:
    source = SOURCE_BY_ID[source_id]
    return DocumentCandidate(
        document_id=make_document_id(source_id, url),
        source_id=source_id,
        url=url,
        title=title,
        organization=source.organization,
        jurisdiction=source.jurisdiction,
        language=language,
        hazards=hazards or infer_hazards(url + " " + title),
        source_type=source.source_type,
        document_type=document_type,
        discovered_from=discovered_from,
    )


def explicit_candidates() -> list[DocumentCandidate]:
    urls = [
        ("ready", "https://www.ready.gov/sites/default/files/2021-11/are-you-ready-guide.pdf", "Are You Ready? An In-depth Guide to Citizen Preparedness", ["preparedness", "multi_hazard"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/documents/fema_cpg-101-v3-developing-maintaining-eops.pdf", "Developing and Maintaining Emergency Operations Plans CPG 101", ["preparedness", "shelter", "route_uncertainty"], "pdf"),
        ("fema", "https://www.fema.gov/pdf/areyouready/basic_preparedness.pdf", "Are You Ready? Basic Preparedness", ["preparedness", "multi_hazard"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/2020-07/fema-cert_basic-training-participant-manual_01-01-2011.pdf", "CERT Basic Training Participant Manual", ["preparedness", "fire_lpg_chemical", "wounds", "route_uncertainty", "shelter"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/2020-07/fema_cert-basic-training-instructor-guide-compiled_102815.pdf", "CERT Basic Training Instructor Guide", ["preparedness", "fire_lpg_chemical", "wounds", "route_uncertainty", "shelter"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/documents/fema_national-disaster-recovery-framework-third-edition_2024.pdf", "National Disaster Recovery Framework Third Edition", ["preparedness", "shelter", "accessibility", "misinformation"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/2020-07/pre-disaster-recovery-planning-guide-local-governments.pdf", "Pre-Disaster Recovery Planning Guide for Local Governments", ["preparedness", "shelter", "accessibility"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/2020-07/predisaster-recovery-planning-guide-for-state-governments.pdf", "Pre-Disaster Recovery Planning Guide for State Governments", ["preparedness", "shelter", "accessibility"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/2020-07/fema_effective-coordination-recovery-resources-guide_020515.pdf", "Effective Coordination of Recovery Resources", ["preparedness", "shelter", "route_uncertainty"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/documents/NRF_FINALApproved_2011028.pdf", "National Response Framework Fourth Edition", ["preparedness", "response", "accessibility"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/2020-04/NRF_FINALApproved_2011028.pdf", "National Response Framework", ["preparedness", "response", "accessibility"], "pdf"),
        ("fema", "https://www.fema.gov/sites/default/files/2020-05/CPG_101_V2_30NOV2010_FINAL_508.pdf", "Comprehensive Preparedness Guide 101 Version 2", ["preparedness", "shelter", "route_uncertainty"], "pdf"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-100.c", "IS-100.c Introduction to the Incident Command System Course Materials", ["preparedness", "response"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-200.c", "IS-200.c Basic Incident Command System for Initial Response Course Materials", ["preparedness", "response"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-700.b", "IS-700.b National Incident Management System Course Materials", ["preparedness", "response"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-800.d", "IS-800.d National Response Framework Course Materials", ["preparedness", "response"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-230.e", "IS-230.e Fundamentals of Emergency Management Course Materials", ["preparedness", "response", "recovery"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-235.c", "IS-235.c Emergency Planning Course Materials", ["preparedness", "shelter", "route_uncertainty"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-317.a", "IS-317.a Introduction to CERT Course Materials", ["preparedness", "fire_lpg_chemical", "wounds"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-368", "IS-368 Including People with Disabilities and Others with Access and Functional Needs", ["preparedness", "accessibility", "shelter"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-5.a", "IS-5.a Introduction to Hazardous Materials Course Materials", ["fire_lpg_chemical", "preparedness"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-10.a", "IS-10.a Animals in Disasters: Awareness and Preparedness", ["preparedness", "shelter"], "html"),
        ("fema", "https://training.fema.gov/is/coursematerials.aspx?code=IS-11.a", "IS-11.a Animals in Disasters: Community Planning", ["preparedness", "shelter"], "html"),
        ("who", "https://iris.who.int/bitstream/handle/10665/375390/WHO-EURO-2014-8519-48291-71705-eng.pdf", "Floods and health: fact sheets for health professionals", ["flood_cyclone_landslide", "public_health", "wash"], "pdf"),
        ("who", "https://www.who.int/docs/default-source/wash-documents/wash-in-emergencies/technical-notes-on-wash-in-emergencies/who-tn-09-how-much-water-is-needed.pdf?sfvrsn=1e876b2a_6", "How much water is needed in emergencies", ["wash", "water_safety"], "pdf"),
        ("who", "https://www.who.int/docs/default-source/wash-documents/wash-in-emergencies/technical-notes-on-wash-in-emergencies/who-tn-10-hygiene-promotion-in-emergencies.pdf", "Hygiene promotion in emergencies", ["wash", "shelter"], "pdf"),
        ("who", "https://www.who.int/docs/default-source/wash-documents/wash-in-emergencies/technical-notes-on-wash-in-emergencies/who-tn-13-planning-for-excreta-disposal-in-emergencies.pdf", "Planning for excreta disposal in emergencies", ["wash", "shelter"], "pdf"),
        ("sphere", "https://spherestandards.org/wp-content/uploads/Sphere-Handbook-2018-EN.pdf", "Sphere Handbook 2018", ["wash", "shelter", "food_safety", "public_health"], "pdf"),
        ("ifrc", "https://www.ifrc.org/document/public-awareness-and-public-education-disaster-risk-reduction-guide", "Public awareness and public education for disaster risk reduction guide", ["preparedness", "misinformation"], "html"),
        ("unicef", "https://www.unicef.org/wash/emergencies", "WASH in emergencies", ["wash", "children", "shelter"], "html"),
        ("unicef", "https://www.unicef.org/documents/three-star-approach-wash-schools-field-guide", "Three Star Approach for WASH in Schools Field Guide", ["wash", "children"], "html"),
        ("redcross", "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies.html", "Types of Emergencies", ["preparedness", "multi_hazard"], "html"),
        ("redcross", "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/make-a-plan.html", "Make a Disaster Preparedness Plan", ["preparedness", "shelter"], "html"),
        ("usfa", "https://www.usfa.fema.gov/prevention/life-safety-hazards/carbon-monoxide/index.html", "Carbon Monoxide Poisoning Prevention", ["carbon_monoxide", "fire_lpg_chemical"], "html"),
        ("ndma", "https://ndma.gov.in/sites/default/files/PDF/Guidelines/cyclones.pdf", "NDMA Guidelines: Management of Cyclones", ["flood_cyclone_landslide", "shelter"], "pdf"),
        ("ndma", "https://ndma.gov.in/Natural-Hazards/Cyclone", "NDMA Cyclone", ["flood_cyclone_landslide"], "html"),
        ("ndma", "https://ndma.gov.in/Natural-Hazards/Heat-Wave", "NDMA Heat Wave", ["heat_cold"], "html"),
        ("imd", "https://mausam.imd.gov.in/", "IMD Weather Portal", ["route_uncertainty", "flood_cyclone_landslide"], "html"),
        ("mohfw", "https://www.mohfw.gov.in/", "MoHFW health information portal", ["public_health", "heat_cold"], "html"),
        ("fssai", "https://www.fssai.gov.in/", "FSSAI food safety portal", ["food_safety"], "html"),
    ]
    return [candidate(source, url, title, hazards, document_type=doc_type) for source, url, title, hazards, doc_type in urls]


def crawl_seed_urls() -> list[DocumentCandidate]:
    fema_course_codes = [
        "IS-3", "IS-5.a", "IS-7", "IS-8.a", "IS-10.a", "IS-11.a", "IS-15.b", "IS-26", "IS-27", "IS-29.a",
        "IS-100.c", "IS-120.c", "IS-130.a", "IS-200.c", "IS-230.d", "IS-235.b",
        "IS-230.e", "IS-235.c", "IS-240.c", "IS-241.c", "IS-242.c", "IS-244.b", "IS-288.a", "IS-293", "IS-315.a",
        "IS-317.a", "IS-320", "IS-360", "IS-363", "IS-366.a", "IS-368", "IS-393.b",
        "IS-405", "IS-520", "IS-552", "IS-559", "IS-700.b", "IS-703.b", "IS-706",
        "IS-75", "IS-800.d", "IS-906", "IS-907", "IS-909", "IS-916", "IS-951", "IS-2200",
    ]
    seeds = [
        candidate("ready", "https://www.ready.gov/be-informed", "Ready.gov Disasters and Emergencies"),
        candidate("cdc", "https://www.cdc.gov/natural-disasters/", "CDC Natural Disasters and Severe Weather"),
        candidate("cdc", "https://www.cdc.gov/floods/safety/index.html", "CDC Flood Safety"),
        candidate("fda", "https://www.fda.gov/food/buy-store-serve-safe-food/food-and-water-safety-during-power-outages-and-floods", "FDA Food and Water Safety During Power Outages and Floods"),
        candidate("epa", "https://www.epa.gov/natural-disasters", "EPA Natural Disasters"),
        candidate("nws", "https://www.weather.gov/safety/", "NWS Weather Safety"),
        candidate("who", "https://www.who.int/teams/environment-climate-change-and-health/water-sanitation-and-health/environmental-health-in-emergencies", "WHO Environmental health in emergencies"),
        candidate("ifrc", "https://www.ifrc.org/our-work/disasters-climate-and-crises", "IFRC disasters climate and crises"),
        candidate("redcross", "https://www.redcross.org/get-help/how-to-prepare-for-emergencies.html", "Red Cross emergency preparedness"),
        candidate("fema", "https://training.fema.gov/IS/searchISbycurriculum.aspx?keywords=preparedness", "FEMA preparedness course search"),
    ]
    seeds.extend(
        candidate("fema", f"https://training.fema.gov/is/coursematerials.aspx?code={code}", f"{code} FEMA Course Materials")
        for code in fema_course_codes
    )
    out: list[DocumentCandidate] = []
    queue: deque[tuple[DocumentCandidate, int]] = deque((seed, 0) for seed in seeds)
    seen: set[str] = set()
    while queue and len(out) < MAX_DISCOVERED_DOCS:
        current, depth = queue.popleft()
        normalized_url = normalize_url(current.url)
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        if not is_relevant_url(normalized_url):
            continue
        out.append(current)
        depth_limit = 2 if "training.fema.gov" in normalized_url.lower() else 1
        if depth >= depth_limit or current.document_type in {"pdf", "zip"}:
            continue
        try:
            content, content_type = fetch(normalized_url)
        except Exception:
            continue
        if "html" not in content_type.lower():
            continue
        for link, label in html_links(content, normalized_url):
            if len(out) + len(queue) >= MAX_DISCOVERED_DOCS * 2:
                break
            if not allowed_domain(link) or not is_relevant_url(link):
                continue
            source_id = source_id_for_url(link)
            if not source_id:
                continue
            lowered = link.lower().split("?")[0]
            doc_type = "pdf" if lowered.endswith(".pdf") else "zip" if lowered.endswith(".zip") else "html"
            queue.append((candidate(source_id, link, label or link, document_type=doc_type, discovered_from=normalized_url), depth + 1))
    return out


def reliefweb_candidates() -> list[DocumentCandidate]:
    terms = [
        "disaster preparedness guide",
        "emergency shelter handbook",
        "WASH emergencies guideline",
        "flood health guideline",
        "heatwave action plan guide",
        "cyclone preparedness guide",
        "community based disaster risk reduction manual",
        "emergency food safety guidance",
        "humanitarian standards handbook",
        "risk communication community engagement emergency guide",
    ]
    out: list[DocumentCandidate] = []
    seen_urls: set[str] = set()
    guide_terms = ["guide", "guideline", "manual", "handbook", "toolkit", "standard", "training", "preparedness", "framework"]
    for term in terms:
        try:
            response = requests.get(
                "https://api.reliefweb.int/v1/reports",
                params=[
                    ("appname", "beacon-dapt-corpus"),
                    ("limit", str(RELIEFWEB_QUERY_LIMIT)),
                    ("profile", "full"),
                    ("query[value]", term),
                    ("fields[include][]", "title"),
                    ("fields[include][]", "body"),
                    ("fields[include][]", "url"),
                    ("fields[include][]", "file"),
                    ("fields[include][]", "source"),
                    ("fields[include][]", "country"),
                    ("fields[include][]", "disaster_type"),
                    ("fields[include][]", "date"),
                ],
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        for item in payload.get("data", []):
            fields = item.get("fields", {})
            title = str(fields.get("title") or "")
            title_low = title.lower()
            if not any(key in title_low for key in guide_terms):
                continue
            page_url = str(fields.get("url") or item.get("href") or "")
            body = clean_text(str(fields.get("body") or ""))
            source_names = ", ".join(source.get("name", "") for source in fields.get("source", [])[:3] if source.get("name"))
            if body and len(body) >= MIN_EXTRACTED_CHARS and page_url and page_url not in seen_urls:
                seen_urls.add(page_url)
                document_id = make_document_id("reliefweb", page_url)
                DIRECT_TEXTS[page_url] = body
                out.append(
                    DocumentCandidate(
                        document_id=document_id,
                        source_id="reliefweb",
                        url=page_url,
                        title=title,
                        organization=source_names or "ReliefWeb / OCHA indexed source",
                        jurisdiction="global",
                        language="en",
                        hazards=infer_hazards(title + " " + body[:3000]),
                        source_type="humanitarian_reputable",
                        document_type="text",
                        discovered_from=f"reliefweb_api:{term}",
                    )
                )
            for file_info in fields.get("file", [])[:3]:
                file_url = str(file_info.get("url") or "")
                if not file_url or file_url in seen_urls:
                    continue
                if not file_url.lower().split("?")[0].endswith(".pdf"):
                    continue
                seen_urls.add(file_url)
                out.append(
                    DocumentCandidate(
                        document_id=make_document_id("reliefweb", file_url),
                        source_id="reliefweb",
                        url=file_url,
                        title=f"{title} PDF",
                        organization=source_names or "ReliefWeb / OCHA indexed source",
                        jurisdiction="global",
                        language="en",
                        hazards=infer_hazards(title),
                        source_type="humanitarian_reputable",
                        document_type="pdf",
                        discovered_from=f"reliefweb_api:{term}",
                    )
                )
    return out


def existing_source_corpus_candidates() -> list[DocumentCandidate]:
    docs = read_jsonl(ROOT / "data" / "source_corpus" / "document_cards.jsonl")
    out = []
    for row in docs:
        source_id = row.get("source_id")
        if source_id not in SOURCE_BY_ID:
            continue
        out.append(
            DocumentCandidate(
                document_id=make_document_id(source_id, row["url"]),
                source_id=source_id,
                url=row["url"],
                title=row.get("title", row["url"]),
                organization=row.get("organization", SOURCE_BY_ID[source_id].organization),
                jurisdiction=row.get("jurisdiction", SOURCE_BY_ID[source_id].jurisdiction),
                language=row.get("language", "en"),
                hazards=row.get("hazards") or infer_hazards(row["url"] + " " + row.get("title", "")),
                source_type=SOURCE_BY_ID[source_id].source_type,
                document_type=row.get("document_type", "html"),
                license=row.get("license", "recorded_metadata_not_blocking_hackathon_dapt"),
                discovered_from="source_corpus_v0",
            )
        )
    return out


def fetch(url: str) -> tuple[bytes, str]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "")


def html_links(content: bytes, base_url: str) -> list[tuple[str, str]]:
    text = content.decode("utf-8", errors="replace")
    links = []
    for match in re.finditer(r'(?is)<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', text):
        href = match.group(1).strip()
        label = re.sub(r"<[^>]+>", " ", match.group(2))
        label = re.sub(r"\s+", " ", source_builder.repair_mojibake(label)).strip()
        links.append((normalize_url(urljoin(base_url, href)), label))
    return links


def normalize_url(url: str) -> str:
    return urldefrag(url)[0].strip()


def source_id_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for source_id, card in SOURCE_BY_ID.items():
        if host == card.domain or host.endswith("." + card.domain):
            return source_id
    if host.endswith("fema.gov"):
        return "fema"
    return ""


def allowed_domain(url: str) -> bool:
    return bool(source_id_for_url(url))


def is_relevant_url(url: str) -> bool:
    low = url.lower()
    if any(term in low for term in BAD_URL_TERMS):
        return False
    if any(term in low for term in LIVE_TERMS):
        return False
    if "training.fema.gov" in low and low.endswith(".zip"):
        return True
    if low.endswith((".xlsx", ".ppt", ".pptx", ".doc", ".docx")):
        return False
    if low.endswith(".zip") and "training.fema.gov" not in low:
        return False
    return any(term.replace(" ", "-") in low or term.replace(" ", "_") in low or term in low for term in RELEVANT_TERMS)


def make_document_id(source_id: str, url: str) -> str:
    parsed = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "_", (parsed.netloc + parsed.path).lower()).strip("_")[:86]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{source_id}_{slug}_{digest}"


def infer_hazards(text: str) -> list[str]:
    low = text.lower()
    hazards = [hazard for hazard, terms in HAZARD_KEYWORDS.items() if any(term in low for term in terms)]
    return hazards or ["preparedness"]


def detect_language(text: str) -> str:
    devanagari = sum(1 for char in text if "\u0900" <= char <= "\u097f")
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    if devanagari > 200 and ascii_letters > 200:
        return "hi-en"
    if devanagari > 200:
        return "hi"
    hinglish_terms = ["pani", "bijli", "dawai", "bachcha", "garam", "barish", "safai", "khana"]
    if any(re.search(rf"\b{term}\b", text.lower()) for term in hinglish_terms):
        return "hinglish"
    return "en"


def is_pdf(candidate: DocumentCandidate, content_type: str) -> bool:
    return candidate.document_type == "pdf" or "pdf" in content_type.lower() or candidate.url.lower().split("?")[0].endswith(".pdf")


def is_zip(candidate: DocumentCandidate, content_type: str) -> bool:
    lowered = candidate.url.lower().split("?")[0]
    return candidate.document_type == "zip" or lowered.endswith(".zip") or "zip" in content_type.lower()


def extract_pdf(content: bytes) -> tuple[str, str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return "", "pdf_extraction_unavailable"
    import io

    reader = PdfReader(io.BytesIO(content), strict=False)
    page_count = len(reader.pages)
    pages = []
    for page in reader.pages[:PDF_MAX_PAGES]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = clean_text("\n".join(pages))
    if not text:
        return "", "empty_pdf_text"
    if page_count > PDF_MAX_PAGES:
        return text, f"ok_truncated_pdf_pages_{PDF_MAX_PAGES}_of_{page_count}"
    return text, f"ok_pdf_pages_{page_count}"


def extract_zip_pdfs(content: bytes) -> tuple[str, str]:
    import io

    texts: list[str] = []
    pdf_count = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if not lower.endswith(".pdf"):
                continue
            if any(skip in lower for skip in ["visual", "slide", "poi", "brochure", "flyer", "fact_sheet"]):
                continue
            if pdf_count >= ZIP_MAX_PDFS:
                break
            try:
                text, _status = extract_pdf(archive.read(name))
            except Exception:
                continue
            if text:
                pdf_count += 1
                texts.append(f"{Path(name).name}\n{text}")
    text = clean_text("\n".join(texts))
    if not text:
        return "", "empty_zip_pdf_text"
    return text, f"ok_zip_pdfs_{pdf_count}"


def extract_html(content: bytes) -> str:
    return clean_text(source_builder.strip_html(content))


def clean_text(text: str) -> str:
    text = source_builder.repair_mojibake(text)
    text = re.sub(r"\r", "\n", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 35:
            continue
        low = line.lower()
        if source_builder.should_skip_extracted_line(line):
            continue
        if any(skip in low for skip in ["cookie", "subscribe", "sign up for email", "follow us on", "accessibility statement"]):
            continue
        lines.append(line)
    return "\n".join(source_builder.dedupe_preserve_order(lines))


def quality_reject_reason(text: str, candidate: DocumentCandidate) -> str:
    if not text:
        return "empty_extracted_text"
    if len(text) < MIN_EXTRACTED_CHARS:
        return f"extracted_text_too_short chars={len(text)}"
    low = text.lower()
    if not any(term in low for term in RELEVANT_TERMS):
        return "not_crisis_relevant_after_extraction"
    if sum(low.count(term) for term in ["cookie", "privacy", "subscribe", "login", "share this"]) > 15:
        return "boilerplate_heavy"
    if any(term in low[:2000] for term in LIVE_TERMS) and "guidance" not in low[:2000]:
        return "live_status_or_alert_page"
    return ""


def word_chunks(text: str, min_words: int = 450, max_words: int = 1100) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        if count and count + len(words) > max_words:
            if count >= min_words:
                chunks.append("\n".join(current))
                current, count = [], 0
            else:
                chunks.append("\n".join(current))
                current, count = [], 0
        current.append(paragraph)
        count += len(words)
    if current:
        chunks.append("\n".join(current))
    return [chunk for chunk in chunks if len(chunk) >= 800]


def estimated_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def dev_split(document_id: str) -> str:
    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "dev" if bucket < int(DEV_DOC_RATIO * 100) else "train"


def build() -> int:
    safe_remove_tree(OUT_DIR)
    for directory in [RAW_DIR, EXTRACTED_DIR, REJECTED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    retrieved_at = utc_now()
    candidates_by_url: dict[str, DocumentCandidate] = {}
    for item in [*existing_source_corpus_candidates(), *explicit_candidates(), *crawl_seed_urls(), *reliefweb_candidates()]:
        candidates_by_url.setdefault(normalize_url(item.url), item)

    accepted_cards: list[DocumentCard] = []
    rejected_cards: list[DocumentCard] = []
    all_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    raw_hashes: set[str] = set()
    extracted_hashes: set[str] = set()
    normalized_fingerprints: set[str] = set()

    for candidate_item in sorted(candidates_by_url.values(), key=lambda row: (row.source_id, row.url)):
        card = DocumentCard(
            document_id=candidate_item.document_id,
            source_id=candidate_item.source_id,
            url=candidate_item.url,
            title=candidate_item.title,
            organization=candidate_item.organization,
            jurisdiction=candidate_item.jurisdiction,
            language=candidate_item.language,
            hazards=candidate_item.hazards,
            source_type=candidate_item.source_type,
            document_type=candidate_item.document_type,
            license=candidate_item.license,
            discovered_from=candidate_item.discovered_from,
            retrieved_at=retrieved_at,
            accepted=False,
        )
        try:
            if candidate_item.document_type == "text":
                text = DIRECT_TEXTS.get(candidate_item.url, "")
                status = "ok_direct_text"
                raw_content = json.dumps({"url": candidate_item.url, "title": candidate_item.title, "text": text}, ensure_ascii=False).encode("utf-8")
                raw_hash = sha256_bytes(raw_content)
                if raw_hash in raw_hashes:
                    raise ValueError("duplicate_raw_hash")
                raw_hashes.add(raw_hash)
                raw_path = RAW_DIR / f"{candidate_item.document_id}.json"
                write_bytes_allow_locked_existing(raw_path, raw_content)
                card.raw_path = str(raw_path.relative_to(ROOT))
                card.raw_sha256 = raw_hash
            else:
                content, content_type = fetch(candidate_item.url)
                raw_hash = sha256_bytes(content)
                if raw_hash in raw_hashes:
                    raise ValueError("duplicate_raw_hash")
                raw_hashes.add(raw_hash)
                pdf = is_pdf(candidate_item, content_type)
                zip_doc = is_zip(candidate_item, content_type)
                extension = ".zip" if zip_doc else ".pdf" if pdf else ".html"
                raw_path = RAW_DIR / f"{candidate_item.document_id}{extension}"
                write_bytes_allow_locked_existing(raw_path, content)
                card.raw_path = str(raw_path.relative_to(ROOT))
                card.raw_sha256 = raw_hash
                if zip_doc:
                    text, status = extract_zip_pdfs(content)
                elif pdf:
                    text, status = extract_pdf(content)
                else:
                    text, status = extract_html(content), "ok_html"
            card.extraction_status = status
            reject_reason = quality_reject_reason(text, candidate_item)
            if reject_reason:
                raise ValueError(reject_reason)
            extracted_hash = sha256_text(text)
            if extracted_hash in extracted_hashes:
                raise ValueError("duplicate_extracted_hash")
            extracted_hashes.add(extracted_hash)
            fingerprint = re.sub(r"\W+", "", text.lower())[:12000]
            if fingerprint in normalized_fingerprints:
                raise ValueError("near_duplicate_text_prefix")
            normalized_fingerprints.add(fingerprint)
            language = detect_language(text)
            card.language = language
            card.hazards = sorted(set(candidate_item.hazards + infer_hazards(text[:5000])))
            card.extracted_sha256 = extracted_hash
            card.text_chars = len(text)
            card.estimated_tokens = estimated_tokens(text)
            extracted_path = EXTRACTED_DIR / f"{candidate_item.document_id}.txt"
            extracted_path.write_text(text + "\n", encoding="utf-8")
            card.extracted_path = str(extracted_path.relative_to(ROOT))
            card.accepted = True
            accepted_cards.append(card)
            for index, chunk in enumerate(word_chunks(text)):
                row = {
                    "text_id": f"{candidate_item.document_id}_{index:04d}",
                    "document_id": candidate_item.document_id,
                    "source_id": candidate_item.source_id,
                    "source_type": candidate_item.source_type,
                    "organization": candidate_item.organization,
                    "jurisdiction": candidate_item.jurisdiction,
                    "language": language,
                    "hazards": card.hazards,
                    "title": candidate_item.title,
                    "url": candidate_item.url,
                    "retrieved_at": retrieved_at,
                    "license": candidate_item.license,
                    "text_sha256": sha256_text(chunk),
                    "estimated_tokens": estimated_tokens(chunk),
                    "text": chunk,
                }
                all_rows.append(row)
                retrieval_rows.append({**row, "chunk_id": row["text_id"]})
        except Exception as exc:
            card.accepted = False
            card.reject_reason = f"{type(exc).__name__}: {exc}"
            rejected_cards.append(card)

    train_rows = [row for row in all_rows if dev_split(row["document_id"]) == "train"]
    dev_rows = [row for row in all_rows if dev_split(row["document_id"]) == "dev"]
    write_jsonl(OUT_DIR / "source_cards.jsonl", [asdict(card) for card in source_cards()])
    write_jsonl(OUT_DIR / "document_cards.jsonl", [asdict(card) for card in accepted_cards])
    write_jsonl(OUT_DIR / "rejected_document_cards.jsonl", [asdict(card) for card in rejected_cards])
    write_jsonl(OUT_DIR / "dapt_all.jsonl", all_rows)
    write_jsonl(OUT_DIR / "dapt_train.jsonl", train_rows)
    write_jsonl(OUT_DIR / "dapt_dev.jsonl", dev_rows)
    write_jsonl(OUT_DIR / "retrieval_chunks.jsonl", retrieval_rows)
    manifest = build_manifest(accepted_cards, rejected_cards, all_rows, train_rows, dev_rows, retrieval_rows, retrieved_at)
    (OUT_DIR / "dapt_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def build_manifest(
    accepted_cards: list[DocumentCard],
    rejected_cards: list[DocumentCard],
    all_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    retrieved_at: str,
) -> dict[str, Any]:
    total_tokens = sum(row["estimated_tokens"] for row in all_rows)
    train_docs = {row["document_id"] for row in train_rows}
    dev_docs = {row["document_id"] for row in dev_rows}
    file_hashes = {}
    for name in ["dapt_all.jsonl", "dapt_train.jsonl", "dapt_dev.jsonl", "retrieval_chunks.jsonl"]:
        path = OUT_DIR / name
        file_hashes[name] = sha256_bytes(path.read_bytes()) if path.exists() else ""
    return {
        "created_at": retrieved_at,
        "dapt_ready": total_tokens >= MIN_TRAIN_TOKENS and bool(train_rows) and bool(dev_rows) and not (train_docs & dev_docs),
        "minimum_token_target": MIN_TRAIN_TOKENS,
        "preferred_token_target": PREFERRED_TRAIN_TOKENS,
        "estimated_tokens": total_tokens,
        "accepted_document_count": len(accepted_cards),
        "rejected_document_count": len(rejected_cards),
        "dapt_row_count": len(all_rows),
        "train_row_count": len(train_rows),
        "dev_row_count": len(dev_rows),
        "retrieval_chunk_count": len(retrieval_rows),
        "train_document_count": len(train_docs),
        "dev_document_count": len(dev_docs),
        "train_dev_document_overlap": sorted(train_docs & dev_docs),
        "by_source": dict(Counter(row.source_id for row in accepted_cards).most_common()),
        "by_source_type": dict(Counter(row.source_type for row in accepted_cards).most_common()),
        "by_language": dict(Counter(row.language for row in accepted_cards).most_common()),
        "by_hazard": dict(Counter(hazard for row in accepted_cards for hazard in row.hazards).most_common()),
        "top_sources_by_tokens": token_breakdown(all_rows, "source_id"),
        "rejection_reasons": dict(Counter(card.reject_reason.split(":")[0] for card in rejected_cards).most_common()),
        "known_weaknesses": known_weaknesses(total_tokens, accepted_cards),
        "recommended_training_variables_to_discuss": {
            "objective": "causal_lm_continuation",
            "max_seq_length": 1024,
            "epochs": "1 short epoch first",
            "learning_rate": "start conservative, e.g. 5e-6 to 1e-5 for QLoRA",
            "adapter_scope": "compare attention-only vs attention+MLP only if compute allows",
        },
        "file_hashes": file_hashes,
    }


def token_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(key, "unknown"))] += int(row.get("estimated_tokens", 0))
    return dict(counts.most_common())


def known_weaknesses(total_tokens: int, accepted_cards: list[DocumentCard]) -> list[str]:
    weaknesses = []
    if total_tokens < MIN_TRAIN_TOKENS:
        weaknesses.append("below_2m_estimated_tokens_do_not_train_without_user_accepting_small_cpt_run")
    languages = Counter(card.language for card in accepted_cards)
    if languages.get("hi", 0) + languages.get("hi-en", 0) + languages.get("hinglish", 0) < 3:
        weaknesses.append("limited_hindi_hinglish_source_text")
    if Counter(card.source_type for card in accepted_cards).get("india_official", 0) < 5:
        weaknesses.append("india_official_coverage_still_light")
    return weaknesses


def validate() -> int:
    errors: list[str] = []
    manifest_path = OUT_DIR / "dapt_manifest.json"
    train = read_jsonl(OUT_DIR / "dapt_train.jsonl")
    dev = read_jsonl(OUT_DIR / "dapt_dev.jsonl")
    all_rows = read_jsonl(OUT_DIR / "dapt_all.jsonl")
    docs = read_jsonl(OUT_DIR / "document_cards.jsonl")
    rejected = read_jsonl(OUT_DIR / "rejected_document_cards.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not train:
        errors.append("dapt_train.jsonl is empty")
    if not dev:
        errors.append("dapt_dev.jsonl is empty")
    train_docs = {row.get("document_id") for row in train}
    dev_docs = {row.get("document_id") for row in dev}
    overlap = sorted(train_docs & dev_docs)
    if overlap:
        errors.append(f"train/dev document overlap: {overlap[:10]}")
    for row in all_rows:
        for key in ["text_id", "document_id", "source_id", "url", "hazards", "language", "text_sha256", "text"]:
            if not row.get(key):
                errors.append(f"{row.get('text_id', '<missing>')}: missing {key}")
        if len(row.get("text", "")) < 800:
            errors.append(f"{row.get('text_id')}: short text block")
    for doc in docs:
        if not doc.get("extracted_path") or not doc.get("extracted_sha256") or not doc.get("text_chars"):
            errors.append(f"{doc.get('document_id')}: accepted doc missing extraction metadata")
    for doc in rejected:
        if not doc.get("reject_reason"):
            errors.append(f"{doc.get('document_id')}: rejected doc missing reason")
    if manifest.get("estimated_tokens", 0) < MIN_TRAIN_TOKENS:
        errors.append(f"token target not met: {manifest.get('estimated_tokens', 0)} < {MIN_TRAIN_TOKENS}")
    if errors:
        print("\n".join(errors[:100]))
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more")
        return 1
    print(
        "validation_passed "
        f"tokens={manifest.get('estimated_tokens')} train_rows={len(train)} dev_rows={len(dev)} docs={len(docs)}"
    )
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    if command == "build":
        raise SystemExit(build())
    if command == "validate":
        raise SystemExit(validate())
    raise SystemExit(f"unknown command: {command}")
