from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "source_corpus"
RAW_DIR = OUT_DIR / "raw"
EXTRACTED_DIR = OUT_DIR / "extracted"
DAPT_DIR = OUT_DIR / "dapt_clean"
RETRIEVAL_DIR = OUT_DIR / "retrieval_chunks"
NEWS_DIR = OUT_DIR / "news_event_cards"
REJECTED_DIR = OUT_DIR / "rejected"

USER_AGENT = "BeaconSourceCorpusBuilder/0.1 (+https://github.com/Rishavutkarsh/Beacon)"
TIMEOUT_SECONDS = 45
PDF_MAX_PAGES = 80
MIN_EXTRACTED_CHARS = 500


@dataclass
class SourceCard:
    source_id: str
    organization: str
    domain: str
    source_tier: int
    jurisdiction: str
    trust_policy: str
    notes: str = ""


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
    document_type: str
    published_date: str
    retrieved_at: str
    staleness_class: str
    license: str
    terms_url: str
    copyright_status: str
    can_store_raw: bool
    can_train: bool
    can_retrieve: bool
    review_status: str
    reject_reason: str = ""
    raw_sha256: str = ""
    extracted_sha256: str = ""
    raw_path: str = ""
    extracted_path: str = ""
    extraction_status: str = "not_attempted"
    text_chars: int = 0
    notes: str = ""


@dataclass
class NewsEventCard:
    event_id: str
    title: str
    url: str
    outlet: str
    published_date: str
    jurisdiction: str
    hazards: list[str]
    event_summary: str
    relevance_notes: str
    copyright_status: str = "copyrighted_news_metadata_only"
    can_store_raw: bool = False
    can_train: bool = False
    can_retrieve: bool = False


def utc_now() -> str:
    return datetime.now(UTC).date().isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_file(
        path,
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
    )


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        if path.exists():
            path.unlink(missing_ok=True)
        path.write_text(text, encoding="utf-8")


def slugify(value: str) -> str:
    value = re.sub(r"https?://", "", value.lower())
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")[:96]


def source_cards() -> list[SourceCard]:
    return [
        SourceCard("ndma", "National Disaster Management Authority, India", "ndma.gov.in", 1, "india", "India official disaster-management source; document-level license review required."),
        SourceCard("sachet", "SACHET / NDMA National Disaster Alert Portal", "sachet.ndma.gov.in", 1, "india", "India official alert/dos-donts source; live alerts excluded from DAPT."),
        SourceCard("imd", "India Meteorological Department", "imd.gov.in", 1, "india", "India official weather source; warnings are retrieval-only unless evergreen guidance."),
        SourceCard("cdc", "Centers for Disease Control and Prevention", "cdc.gov", 2, "us/global-applicable", "US government public-domain public-health guidance; stable pages can enter DAPT."),
        SourceCard("fda", "US Food and Drug Administration", "fda.gov", 2, "us/global-applicable", "US government public-domain food/drug safety guidance; stable pages can enter DAPT."),
        SourceCard("epa", "US Environmental Protection Agency", "epa.gov", 2, "us/global-applicable", "US government public-domain water/environment guidance; stable pages can enter DAPT."),
        SourceCard("ready", "Ready.gov / FEMA", "ready.gov", 2, "us/global-applicable", "US government public-domain preparedness guidance; stable pages can enter DAPT."),
        SourceCard("nws", "US National Weather Service", "weather.gov", 2, "us/global-applicable", "US government public-domain weather safety guidance; stable pages can enter DAPT."),
        SourceCard("usgs", "US Geological Survey", "usgs.gov", 2, "us/global-applicable", "US government public-domain geological hazard guidance; stable pages can enter DAPT."),
        SourceCard("who", "World Health Organization", "who.int", 2, "global", "Authoritative global health guidance; copyright varies by document, conservative retrieval-first eligibility."),
        SourceCard("ifrc", "International Federation of Red Cross and Red Crescent Societies", "ifrc.org", 2, "global", "Authoritative humanitarian guidance; license review required per document."),
        SourceCard("sphere", "Sphere Association", "spherestandards.org", 2, "global", "Humanitarian standards; license review required per document."),
        SourceCard("unicef", "UNICEF", "unicef.org", 2, "global", "Authoritative child/WASH guidance; license review required per document."),
    ]


def seed_documents(retrieved_at: str) -> list[DocumentCard]:
    us_public = {
        "license": "US government public domain / public information",
        "terms_url": "https://www.usa.gov/government-copyright",
        "copyright_status": "public_domain_us_government",
        "can_store_raw": True,
        "can_train": True,
        "can_retrieve": True,
        "review_status": "accepted_dapt_retrieval",
    }
    india_retrieval = {
        "license": "official public website; training rights not assumed",
        "terms_url": "site terms/copyright notice pending document-level review",
        "copyright_status": "official_copyright_unclear",
        "can_store_raw": True,
        "can_train": False,
        "can_retrieve": True,
        "review_status": "accepted_retrieval_only",
    }
    global_retrieval = {
        "license": "official/public guidance; training rights not assumed",
        "terms_url": "document terms pending review",
        "copyright_status": "copyright_unclear_or_restricted",
        "can_store_raw": True,
        "can_train": False,
        "can_retrieve": True,
        "review_status": "accepted_retrieval_only",
    }

    def doc(
        document_id: str,
        source_id: str,
        url: str,
        title: str,
        organization: str,
        jurisdiction: str,
        hazards: list[str],
        document_type: str = "html",
        published_date: str = "current",
        staleness_class: str = "evergreen",
        policy: dict[str, Any] | None = None,
        notes: str = "",
        reject_reason: str = "",
    ) -> DocumentCard:
        policy = policy or us_public
        return DocumentCard(
            document_id=document_id,
            source_id=source_id,
            url=url,
            title=title,
            organization=organization,
            jurisdiction=jurisdiction,
            language="en",
            hazards=hazards,
            document_type=document_type,
            published_date=published_date,
            retrieved_at=retrieved_at,
            staleness_class=staleness_class,
            notes=notes,
            reject_reason=reject_reason,
            **policy,
        )

    return [
        doc("cdc_floodwater_safety", "cdc", "https://www.cdc.gov/floods/safety/floodwater-after-a-disaster-or-emergency-safety.html", "Safety Guidelines: Floodwater", "Centers for Disease Control and Prevention", "us/global-applicable", ["floodwater", "wounds", "electrical", "contamination"], published_date="2024-02-06"),
        doc("cdc_reenter_flooded_home", "cdc", "https://www.cdc.gov/floods/safety/reentering-your-flooded-home-safety.html", "Reentering Your Flooded Home Safely", "Centers for Disease Control and Prevention", "us/global-applicable", ["floodwater", "structural", "electrical", "mold"], published_date="2024-02-06"),
        doc("cdc_power_outage", "cdc", "https://www.cdc.gov/natural-disasters/response/what-to-do-protect-yourself-during-a-power-outage.html", "What to Do to Protect Yourself During a Power Outage", "Centers for Disease Control and Prevention", "us/global-applicable", ["power_outage", "carbon_monoxide", "food", "medicine", "heat_cold"], published_date="2024-02-14"),
        doc("cdc_food_after_emergency", "cdc", "https://www.cdc.gov/food-safety/foods/keep-food-safe-after-emergency.html", "Keep Food Safe After a Disaster or Emergency", "Centers for Disease Control and Prevention", "us/global-applicable", ["food_safety", "floodwater", "power_outage"], published_date="current"),
        doc("cdc_emergency_water", "cdc", "https://www.cdc.gov/water-emergency/about/index.html", "About Drinking Water Emergencies", "Centers for Disease Control and Prevention", "us/global-applicable", ["water_safety", "disinfection", "boil_water"], published_date="current"),
        doc("cdc_co_clinical_disasters", "cdc", "https://www.cdc.gov/carbon-monoxide/hcp/clinical-guidance/index.html", "Clinical Guidance for Carbon Monoxide Poisoning Following Disasters and Severe Weather", "Centers for Disease Control and Prevention", "us/global-applicable", ["carbon_monoxide", "generators", "symptoms"], published_date="2024-07-08"),
        doc("cdc_diabetes_emergencies", "cdc", "https://www.cdc.gov/diabetes/articles/diabetes-care-emergencies.html", "Diabetes Care During Emergencies", "Centers for Disease Control and Prevention", "us/global-applicable", ["diabetes", "medicine_disruption", "emergency_kit"], published_date="2024-05-15"),
        doc("cdc_insulin_emergency", "cdc", "https://www.cdc.gov/diabetes/articles/managing-insulin-in-emergency.html", "Managing Insulin in an Emergency", "Centers for Disease Control and Prevention", "us/global-applicable", ["diabetes", "insulin", "medicine_disruption"], published_date="2024-05-15"),
        doc("fda_food_water_floods", "fda", "https://www.fda.gov/food/buy-store-serve-safe-food/food-and-water-safety-during-power-outages-and-floods", "Food and Water Safety During Power Outages and Floods", "US Food and Drug Administration", "us/global-applicable", ["food_safety", "water_safety", "floodwater", "power_outage"], policy=us_public),
        doc("fda_drugs_disaster", "fda", "https://www.fda.gov/drugs/emergency-preparedness-drugs/safe-drug-use-after-natural-disaster", "Safe Drug Use After a Natural Disaster", "US Food and Drug Administration", "us/global-applicable", ["medicine_disruption", "floodwater", "drug_safety"], policy=us_public),
        doc("epa_emergency_disinfection", "epa", "https://www.epa.gov/your-drinking-water/emergency-disinfection-drinking-water", "Emergency Disinfection of Drinking Water", "US Environmental Protection Agency", "us/global-applicable", ["water_safety", "disinfection", "chemical_contamination"], policy=us_public),
        doc("epa_flood_cleanup_iaq", "epa", "https://www.epa.gov/indoor-air-quality-iaq/resources-flood-cleanup-and-indoor-air-quality", "Resources for Flood Cleanup and Indoor Air Quality", "US Environmental Protection Agency", "us/global-applicable", ["flood_cleanup", "mold", "indoor_air", "shelter_hygiene"], policy=us_public),
        doc("ready_floods", "ready", "https://www.ready.gov/floods", "Floods", "Ready.gov / FEMA", "us/global-applicable", ["flood", "route_safety", "preparedness"], policy=us_public),
        doc("ready_power_outages", "ready", "https://www.ready.gov/power-outages", "Power Outages", "Ready.gov / FEMA", "us/global-applicable", ["power_outage", "preparedness", "carbon_monoxide"], policy=us_public),
        doc("ready_heat", "ready", "https://www.ready.gov/heat", "Extreme Heat", "Ready.gov / FEMA", "us/global-applicable", ["heatwave", "vulnerable_people"], policy=us_public),
        doc("ready_winter", "ready", "https://www.ready.gov/winter-weather", "Winter Weather", "Ready.gov / FEMA", "us/global-applicable", ["cold_wave", "winter_storm", "power_outage"], policy=us_public),
        doc("ready_wildfires", "ready", "https://www.ready.gov/wildfires", "Wildfires", "Ready.gov / FEMA", "us/global-applicable", ["fire", "smoke", "evacuation"], policy=us_public),
        doc("ready_landslides", "ready", "https://www.ready.gov/landslides-debris-flow", "Landslides and Debris Flow", "Ready.gov / FEMA", "us/global-applicable", ["landslide", "structural", "route_safety"], policy=us_public),
        doc("nws_flood_safety", "nws", "https://www.weather.gov/safety/flood", "Flood Safety", "US National Weather Service", "us/global-applicable", ["flood", "route_safety"], policy=us_public),
        doc("nws_turn_around", "nws", "https://www.weather.gov/safety/flood-turn-around-dont-drown", "Turn Around Don't Drown", "US National Weather Service", "us/global-applicable", ["flood", "route_safety"], policy=us_public),
        doc("nws_lightning", "nws", "https://www.weather.gov/safety/lightning", "Lightning Safety", "US National Weather Service", "us/global-applicable", ["lightning", "storm"], policy=us_public),
        doc("nws_heat", "nws", "https://www.weather.gov/safety/heat", "Heat Safety", "US National Weather Service", "us/global-applicable", ["heatwave"], policy=us_public),
        doc("nws_winter", "nws", "https://www.weather.gov/safety/winter", "Winter Weather Safety", "US National Weather Service", "us/global-applicable", ["cold_wave", "winter_storm"], policy=us_public),
        doc("usgs_landslide_signs", "usgs", "https://www.usgs.gov/programs/landslide-hazards/what-are-signs-landslide-development-what-do-i-do-if-a-landslide-occurs", "What are the signs of landslide development?", "US Geological Survey", "us/global-applicable", ["landslide", "structural"], policy=us_public),
        doc("ndma_dos_donts", "sachet", "https://sachet.ndma.gov.in/DosDont", "SACHET Dos and Don'ts", "National Disaster Management Authority, India", "india", ["flood", "cyclone", "landslide", "heatwave", "lightning", "fire"], policy=india_retrieval, staleness_class="seasonal", reject_reason="sachet_static_capture_contains_embedded_browser_api_key"),
        doc("ndma_flood_guidelines_pdf", "ndma", "https://ndma.gov.in/images/guidelines/flood.pdf", "National Disaster Management Guidelines: Management of Floods", "National Disaster Management Authority, India", "india", ["flood", "preparedness", "response"], document_type="pdf", published_date="2008", policy=india_retrieval),
        doc("ndma_cyclone_guidelines_pdf", "ndma", "https://ndma.gov.in/sites/default/files/PDF/Guidelines/cyclones.pdf", "National Disaster Management Guidelines: Management of Cyclones", "National Disaster Management Authority, India", "india", ["cyclone", "coastal_evacuation", "shelter"], document_type="pdf", published_date="2008", policy=india_retrieval),
        doc("ndma_heatwave_guidelines_pdf", "ndma", "https://ndma.gov.in/sites/default/files/PDF/Guidelines/heatwaveguidelines.pdf", "Guidelines for Preparation of Action Plan: Prevention and Management of Heat-Wave", "National Disaster Management Authority, India", "india", ["heatwave", "public_health", "preparedness"], document_type="pdf", published_date="2017", policy=india_retrieval, staleness_class="seasonal", reject_reason="known_bad_pdf_extraction_too_short"),
        doc("nidm_ndma_heatwave_pdf", "ndma", "https://nidm.gov.in/PDF/pubs/NDMA/27.pdf", "National Guidelines for Preparation of Action Plan: Prevention and Management of Heat Wave", "National Institute of Disaster Management / NDMA", "india", ["heatwave", "public_health"], document_type="pdf", published_date="current copy", policy=india_retrieval, staleness_class="seasonal", notes="Mirror/copy from NIDM domain; retrieval-only until license reviewed.", reject_reason="known_bad_pdf_extraction_too_short"),
        doc("ndma_floods", "ndma", "https://ndma.gov.in/Natural-Hazards/Floods", "NDMA Floods", "National Disaster Management Authority, India", "india", ["flood", "preparedness"], policy=india_retrieval),
        doc("ndma_cyclone", "ndma", "https://ndma.gov.in/Natural-Hazards/Cyclone", "NDMA Cyclone", "National Disaster Management Authority, India", "india", ["cyclone", "preparedness"], policy=india_retrieval),
        doc("ndma_landslide", "ndma", "https://ndma.gov.in/Natural-Hazards/Landslide", "NDMA Landslide", "National Disaster Management Authority, India", "india", ["landslide", "structural"], policy=india_retrieval),
        doc("ndma_heat_wave", "ndma", "https://ndma.gov.in/Natural-Hazards/Heat-Wave", "NDMA Heat Wave", "National Disaster Management Authority, India", "india", ["heatwave"], policy=india_retrieval, staleness_class="seasonal"),
        doc("imd_weather_warnings", "imd", "https://mausam.imd.gov.in/", "IMD Weather Portal", "India Meteorological Department", "india", ["weather_alerts", "live_fact_uncertainty"], policy={**india_retrieval, "can_train": False, "review_status": "accepted_retrieval_only"}, staleness_class="live", notes="Live/current warnings are excluded from DAPT."),
        doc("who_wash_emergencies", "who", "https://www.who.int/teams/environment-climate-change-and-health/water-sanitation-and-health/environmental-health-in-emergencies", "Environmental health in emergencies", "World Health Organization", "global", ["wash", "water_safety", "sanitation"], policy=global_retrieval),
        doc("who_diarrhoea", "who", "https://www.who.int/health-topics/diarrhoea", "Diarrhoea", "World Health Organization", "global", ["diarrhoea", "ors", "dehydration"], policy=global_retrieval),
        doc("who_risk_comm", "who", "https://www.who.int/emergencies/risk-communications", "Risk communications", "World Health Organization", "global", ["risk_communication", "misinformation"], policy=global_retrieval),
        doc("ifrc_public_awareness", "ifrc", "https://www.ifrc.org/document/public-awareness-and-public-education-disaster-risk-reduction", "Public awareness and public education for disaster risk reduction", "International Federation of Red Cross and Red Crescent Societies", "global", ["preparedness", "risk_communication"], policy=global_retrieval),
        doc("sphere_handbook", "sphere", "https://spherestandards.org/handbook/", "Sphere Handbook", "Sphere Association", "global", ["wash", "shelter", "food_security", "health"], policy=global_retrieval),
        doc("unicef_wash_emergencies", "unicef", "https://www.unicef.org/wash/emergencies", "WASH in emergencies", "UNICEF", "global", ["wash", "children", "emergencies"], policy=global_retrieval),
    ]


def news_cards() -> list[NewsEventCard]:
    return [
        NewsEventCard(
            event_id="news_2024_wayanad_landslide_reuters_archive",
            title="Landslides in Kerala/Wayanad, India, 2024",
            url="https://www.reutersconnect.com/item/landslides-in-the-hills-in-wayanad/dGFnOnJldXRlcnMuY29tLDIwMjQ6bmV3c21sX1JDMkg1OUE1TlFKUA%3D%3D",
            outlet="Reuters Connect",
            published_date="2024-07-30",
            jurisdiction="india/kerala",
            hazards=["landslide", "flood", "rescue", "structural"],
            event_summary="Metadata card for Wayanad landslide event realism: heavy rain, landslide damage, rescue teams, access disruption.",
            relevance_notes="Use only for scenario archetypes and language realism; do not store copyrighted article/media text.",
        ),
        NewsEventCard(
            event_id="news_2024_tripura_floods_reuters",
            title="Floods and landslides in Tripura displace tens of thousands",
            url="https://www.investing.com/news/world-news/floods-landslides-in-indias-tripura-displace-tens-of-thousands-3584664",
            outlet="Reuters via Investing.com",
            published_date="2024-08-23",
            jurisdiction="india/tripura",
            hazards=["flood", "landslide", "evacuation", "shelter"],
            event_summary="Metadata card for northeastern India flood/landslide displacement and evacuation realism.",
            relevance_notes="Copyrighted wire content; metadata-only.",
        ),
        NewsEventCard(
            event_id="news_2025_cyclone_montha_reuters",
            title="India evacuates coastal residents as cyclone gains strength",
            url="https://www.tradingview.com/news/reuters.com%2C2025%3Anewsml_L1N3W805E%3A0-india-evacuates-tens-of-thousands-as-cyclone-montha-gains-strength/",
            outlet="Reuters via TradingView",
            published_date="2025",
            jurisdiction="india/andhra-pradesh",
            hazards=["cyclone", "coastal_evacuation", "live_fact_uncertainty"],
            event_summary="Metadata card for cyclone evacuation and official-warning realism.",
            relevance_notes="Contains time-bound operational facts; metadata-only.",
        ),
    ]


def fetch(url: str) -> tuple[bytes, str]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    return response.content, content_type


def strip_html(content: bytes) -> str:
    raw = content.decode("utf-8", errors="replace")
    raw = repair_mojibake(raw)
    raw = re.sub(r"(?is)<(script|style|nav|header|footer|noscript|svg).*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?is)</(p|div|li|h1|h2|h3|h4|tr)>", "\n", raw)
    raw = re.sub(r"(?is)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 40:
            continue
        if should_skip_extracted_line(line):
            continue
        lines.append(line)
    return "\n".join(dedupe_preserve_order(lines))


def repair_mojibake(text: str) -> str:
    replacements = {
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\ufffd": '"',
        "\u00e2\u20ac\u201d": "-",
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u00a6": "...",
        "\u00c3\u00b1": "ñ",
        "\u00c3\u00a9": "é",
        "\u00c3\u00a1": "á",
        "\u00c3\u00ad": "í",
        "\u00c3\u00b3": "ó",
        "\u00c3\u00ba": "ú",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    if "â" not in text:
        return text
    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text
    return repaired if len(repaired) > len(text) * 0.8 else text


def should_skip_extracted_line(line: str) -> bool:
    normalized = line.lower()
    skip_phrases = [
        "skip to main content",
        "share this page",
        "cookies",
        "privacy policy",
        "an official website of the united states government",
        "a .gov website belongs to an official government organization",
        "means you've safely connected to the .gov website",
        "share sensitive information only on official, secure websites",
        "please visit fema.gov for up-to-date information on current disaster declarations",
        "if you have questions about your disaster assistance application",
        "chemicals and hazardous materials incidents",
        "download infographic pdf",
    ]
    return any(phrase in normalized for phrase in skip_phrases)


def extract_pdf(content: bytes) -> tuple[str, str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return "", "pdf_extraction_unavailable_install_pypdf"
    import io

    reader = PdfReader(io.BytesIO(content))
    pages = []
    page_count = len(reader.pages)
    for page in reader.pages[:PDF_MAX_PAGES]:
        pages.append(page.extract_text() or "")
    text = "\n".join(page.strip() for page in pages if page.strip())
    text = normalize_text_block(text)
    if not text:
        return text, "empty_pdf_text"
    if page_count > PDF_MAX_PAGES:
        return text, f"ok_truncated_pdf_pages_{PDF_MAX_PAGES}_of_{page_count}"
    return text, f"ok_pdf_pages_{page_count}"


def normalize_text_block(text: str) -> str:
    text = repair_mojibake(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 30]
    lines = [line for line in lines if not should_skip_extracted_line(line)]
    return "\n".join(dedupe_preserve_order(lines))


def dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for line in lines:
        key = re.sub(r"\W+", "", line.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def chunk_text(text: str, max_chars: int = 1400) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current += "\n" + paragraph
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def is_pdf(document: DocumentCard, content_type: str) -> bool:
    return document.url.lower().endswith(".pdf") or "pdf" in content_type.lower() or document.document_type.lower() == "pdf"


def build() -> int:
    retrieved_at = utc_now()
    sources = source_cards()
    documents = seed_documents(retrieved_at)
    news = news_cards()

    if OUT_DIR.exists():
        try:
            shutil.rmtree(OUT_DIR)
        except PermissionError:
            for generated_path in [
                OUT_DIR / "source_cards.jsonl",
                OUT_DIR / "document_cards.jsonl",
                OUT_DIR / "source_corpus_report.json",
                REJECTED_DIR / "document_cards_rejected.jsonl",
                DAPT_DIR / "dapt_clean.jsonl",
                RETRIEVAL_DIR / "retrieval_chunks.jsonl",
                NEWS_DIR / "news_event_cards.jsonl",
            ]:
                if generated_path.exists():
                    try:
                        generated_path.unlink()
                    except PermissionError:
                        pass
            for directory in [EXTRACTED_DIR]:
                if directory.exists():
                    try:
                        shutil.rmtree(directory)
                    except PermissionError:
                        pass
    for directory in [RAW_DIR, EXTRACTED_DIR, DAPT_DIR, RETRIEVAL_DIR, NEWS_DIR, REJECTED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    accepted_cards: list[DocumentCard] = []
    rejected_cards: list[DocumentCard] = []
    dapt_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    for document in documents:
        if document.reject_reason:
            rejected_cards.append(document)
            continue
        try:
            if not document.can_store_raw:
                document.extraction_status = "metadata_only_no_raw_storage"
                accepted_cards.append(document)
                continue
            content, content_type = fetch(document.url)
            extension = ".pdf" if is_pdf(document, content_type) else ".html"
            raw_path = RAW_DIR / f"{document.document_id}{extension}"
            try:
                raw_path.write_bytes(content)
            except PermissionError:
                if not raw_path.exists():
                    raise
            document.raw_path = str(raw_path.relative_to(ROOT))
            document.raw_sha256 = sha256_bytes(content)
            if is_pdf(document, content_type):
                text, status = extract_pdf(content)
                document.extraction_status = status
            else:
                text = strip_html(content)
                document.extraction_status = "ok" if text else "empty_html_text"
            if not text:
                raise ValueError("empty_extracted_text")
            if len(text) < MIN_EXTRACTED_CHARS:
                raise ValueError(f"extracted_text_too_short chars={len(text)}")
            if text:
                extracted_path = EXTRACTED_DIR / f"{document.document_id}.txt"
                write_text_file(extracted_path, text + "\n")
                document.extracted_path = str(extracted_path.relative_to(ROOT))
                document.extracted_sha256 = sha256_text(text)
                document.text_chars = len(text)
                if document.can_train and document.staleness_class in {"evergreen", "seasonal"}:
                    for index, chunk in enumerate(chunk_text(text, max_chars=1800)):
                        dapt_rows.append(
                            {
                                "text_id": f"{document.document_id}_dapt_{index:04d}",
                                "document_id": document.document_id,
                                "source_id": document.source_id,
                                "title": document.title,
                                "url": document.url,
                                "organization": document.organization,
                                "jurisdiction": document.jurisdiction,
                                "hazards": document.hazards,
                                "license": document.license,
                                "can_train": True,
                                "text": chunk,
                            }
                        )
                if document.can_retrieve:
                    for index, chunk in enumerate(chunk_text(text, max_chars=1200)):
                        retrieval_rows.append(
                            {
                                "chunk_id": f"{document.document_id}_chunk_{index:04d}",
                                "document_id": document.document_id,
                                "source_id": document.source_id,
                                "title": document.title,
                                "url": document.url,
                                "organization": document.organization,
                                "jurisdiction": document.jurisdiction,
                                "hazards": document.hazards,
                                "published_date": document.published_date,
                                "retrieved_at": document.retrieved_at,
                                "staleness_class": document.staleness_class,
                                "license": document.license,
                                "text": chunk,
                            }
                        )
            accepted_cards.append(document)
        except Exception as exc:
            document.review_status = "rejected_fetch_or_extract_failed"
            document.reject_reason = f"{type(exc).__name__}: {exc}"
            rejected_cards.append(document)

    write_jsonl(OUT_DIR / "source_cards.jsonl", [asdict(item) for item in sources])
    write_jsonl(OUT_DIR / "document_cards.jsonl", [asdict(item) for item in accepted_cards])
    write_jsonl(REJECTED_DIR / "document_cards_rejected.jsonl", [asdict(item) for item in rejected_cards])
    write_jsonl(DAPT_DIR / "dapt_clean.jsonl", dapt_rows)
    write_jsonl(RETRIEVAL_DIR / "retrieval_chunks.jsonl", retrieval_rows)
    write_jsonl(NEWS_DIR / "news_event_cards.jsonl", [asdict(item) for item in news])

    report = {
        "created_at": retrieved_at,
        "source_count": len(sources),
        "candidate_document_count": len(documents),
        "accepted_document_count": len(accepted_cards),
        "rejected_document_count": len(rejected_cards),
        "dapt_row_count": len(dapt_rows),
        "retrieval_chunk_count": len(retrieval_rows),
        "news_event_card_count": len(news),
        "accepted_by_source": count_by([item.source_id for item in accepted_cards]),
        "accepted_by_jurisdiction": count_by([item.jurisdiction for item in accepted_cards]),
        "dapt_by_source": count_by([row["source_id"] for row in dapt_rows]),
        "retrieval_by_source": count_by([row["source_id"] for row in retrieval_rows]),
        "rejections": [{"document_id": item.document_id, "url": item.url, "reason": item.reject_reason} for item in rejected_cards],
    }
    write_text_file(OUT_DIR / "source_corpus_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def count_by(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def validate() -> int:
    dapt_path = DAPT_DIR / "dapt_clean.jsonl"
    docs_path = OUT_DIR / "document_cards.jsonl"
    news_path = NEWS_DIR / "news_event_cards.jsonl"
    errors: list[str] = []
    dapt = [json.loads(line) for line in dapt_path.read_text(encoding="utf-8").splitlines() if line.strip()] if dapt_path.exists() else []
    docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line.strip()] if docs_path.exists() else []
    news = [json.loads(line) for line in news_path.read_text(encoding="utf-8").splitlines() if line.strip()] if news_path.exists() else []
    doc_by_id = {row["document_id"]: row for row in docs}
    for row in dapt:
        required = ["text_id", "document_id", "source_id", "license", "text", "can_train"]
        for key in required:
            if not row.get(key):
                errors.append(f"{row.get('text_id', '<missing>')}: missing {key}")
        if not row.get("can_train"):
            errors.append(f"{row.get('text_id')}: can_train is not true")
        doc = doc_by_id.get(row.get("document_id"))
        if not doc:
            errors.append(f"{row.get('text_id')}: missing document card")
        elif doc.get("staleness_class") == "live":
            errors.append(f"{row.get('text_id')}: live document entered DAPT")
        elif not doc.get("can_train"):
            errors.append(f"{row.get('text_id')}: source document is not trainable")
        elif doc.get("copyright_status") != "public_domain_us_government":
            errors.append(f"{row.get('text_id')}: non-public-domain source entered DAPT")
        elif doc.get("jurisdiction") != "us/global-applicable":
            errors.append(f"{row.get('text_id')}: non-approved jurisdiction entered DAPT")
    for card in news:
        if card.get("can_train") or card.get("can_store_raw") or card.get("can_retrieve"):
            errors.append(f"{card.get('event_id')}: news card incorrectly enabled for train/raw/retrieve")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"validation_passed dapt_rows={len(dapt)} document_cards={len(docs)} news_cards={len(news)}")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    if command == "build":
        raise SystemExit(build())
    if command == "validate":
        raise SystemExit(validate())
    raise SystemExit(f"unknown command: {command}")
