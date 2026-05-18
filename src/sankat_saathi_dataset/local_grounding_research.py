from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CORPUS = ROOT / "data" / "source_corpus"
DEFAULT_OUT_DIR = ROOT / "data" / "local_grounding" / "source_research_v1"

REQUIRED_HAZARD_FAMILIES: dict[str, set[str]] = {
    "flood_route": {"flood", "floodwater", "route_safety"},
    "water_wash": {"water_safety", "wash", "sanitation", "disinfection", "boil_water"},
    "food_safety": {"food_safety", "food", "power_outage"},
    "power_co_electrical": {"power_outage", "carbon_monoxide", "generators", "electrical"},
    "medicine_diabetes": {"medicine_disruption", "medicine", "drug_safety", "diabetes", "insulin"},
    "wounds_cleanup": {"wounds", "flood_cleanup", "mold", "indoor_air", "contamination"},
    "shelter_vulnerable": {"shelter", "shelter_hygiene", "children", "vulnerable_people", "emergencies"},
    "cyclone_coastal": {"cyclone", "coastal_evacuation"},
    "landslide_structural": {"landslide", "structural"},
    "heat_cold_lightning": {"heatwave", "cold_wave", "winter_storm", "lightning", "storm"},
    "misinformation_live_status": {"risk_communication", "misinformation", "weather_alerts", "live_fact_uncertainty"},
}

IGNORED_PATH_PARTS = {"kaggle_outputs", "kaggle/input", "tmp"}
ACCEPTED_STATUSES = {"accepted_core", "accepted_supporting", "accepted_retrieval_only"}
FINAL_STATUSES = ACCEPTED_STATUSES | {"deferred", "rejected"}
LIVE_STATUS_PATTERNS = [
    "bridge is open",
    "bridge open",
    "shelter is available",
    "shelter available",
    "rescue will arrive",
    "rescue boat is coming",
    "route is safe",
    "road is open",
]
MEDICINE_DOSE_PATTERNS = ["take 1 tablet", "take one tablet", "double dose", "correction dose", "units of insulin"]


SUPPLEMENTAL_CANDIDATES: list[dict[str, Any]] = [
    {
        "document_id": "sachet_dos_donts_metadata",
        "source_id": "sachet",
        "url": "https://sachet.ndma.gov.in/DosDont",
        "title": "SACHET Dos and Don'ts",
        "organization": "SACHET / National Disaster Management Authority, India",
        "jurisdiction": "india",
        "language": "en",
        "hazards": [
            "flood",
            "cyclone",
            "landslide",
            "heatwave",
            "lightning",
            "fire",
            "live_fact_uncertainty",
        ],
        "license": "official public website; metadata/link only until storage terms are reviewed",
        "copyright_status": "official_copyright_unclear",
        "staleness_class": "mixed",
        "extraction_status": "deferred",
        "review_status": "deferred",
        "status_reason": "Useful India official source, but current portal extraction may expose app/runtime artifacts; keep as candidate until clean export is available.",
        "recommendation": "metadata_only_candidate",
    },
    {
        "document_id": "ndma_earthquake_guidance_candidate",
        "source_id": "ndma",
        "url": "https://ndma.gov.in/",
        "title": "NDMA earthquake public safety guidance",
        "organization": "National Disaster Management Authority, India",
        "jurisdiction": "india",
        "language": "en",
        "hazards": ["earthquake", "structural", "shelter"],
        "license": "official public website; document-level review required",
        "copyright_status": "official_copyright_unclear",
        "staleness_class": "evergreen",
        "extraction_status": "not_attempted",
        "review_status": "deferred",
        "status_reason": "Important India gap; add once a stable downloadable page/PDF is identified and extracted cleanly.",
        "recommendation": "research_gap",
    },
    {
        "document_id": "mohfw_heat_public_health_candidate",
        "source_id": "mohfw",
        "url": "https://www.mohfw.gov.in/",
        "title": "MoHFW heat illness and emergency public-health guidance",
        "organization": "Ministry of Health and Family Welfare, India",
        "jurisdiction": "india",
        "language": "en",
        "hazards": ["heatwave", "public_health", "vulnerable_people"],
        "license": "official public website; document-level review required",
        "copyright_status": "official_copyright_unclear",
        "staleness_class": "seasonal",
        "extraction_status": "not_attempted",
        "review_status": "deferred",
        "status_reason": "Potentially better India public-health authority for heat illness; needs specific stable document.",
        "recommendation": "research_gap",
    },
    {
        "document_id": "fssai_food_emergency_candidate",
        "source_id": "fssai",
        "url": "https://www.fssai.gov.in/",
        "title": "FSSAI emergency food-safety guidance",
        "organization": "Food Safety and Standards Authority of India",
        "jurisdiction": "india",
        "language": "en",
        "hazards": ["food_safety", "water_safety", "public_health"],
        "license": "official public website; document-level review required",
        "copyright_status": "official_copyright_unclear",
        "staleness_class": "evergreen",
        "extraction_status": "not_attempted",
        "review_status": "deferred",
        "status_reason": "Potential India-specific food-safety source; needs stable emergency/disaster page.",
        "recommendation": "research_gap",
    },
    {
        "document_id": "ifrc_public_awareness_candidate",
        "source_id": "ifrc",
        "url": "https://www.ifrc.org/document/public-awareness-and-public-education-disaster-risk-reduction",
        "title": "Public awareness and public education for disaster risk reduction",
        "organization": "International Federation of Red Cross and Red Crescent Societies",
        "jurisdiction": "global",
        "language": "en",
        "hazards": ["preparedness", "risk_communication", "vulnerable_people"],
        "license": "NGO/public guidance; storage and reuse terms require document-level review",
        "copyright_status": "copyright_unclear_or_restricted",
        "staleness_class": "evergreen",
        "extraction_status": "deferred",
        "review_status": "deferred",
        "status_reason": "High-quality NGO candidate; prior automated fetch was blocked.",
        "recommendation": "manual_review_candidate",
    },
    {
        "document_id": "sphere_handbook_candidate",
        "source_id": "sphere",
        "url": "https://spherestandards.org/handbook/",
        "title": "Sphere Handbook",
        "organization": "Sphere Association",
        "jurisdiction": "global",
        "language": "en",
        "hazards": ["wash", "shelter", "food_security", "health", "vulnerable_people"],
        "license": "humanitarian standard; storage and reuse terms require document-level review",
        "copyright_status": "copyright_unclear_or_restricted",
        "staleness_class": "evergreen",
        "extraction_status": "deferred",
        "review_status": "deferred",
        "status_reason": "Excellent standard for WASH/shelter, but do not store or distill until licensing and extraction are reviewed.",
        "recommendation": "manual_review_candidate",
    },
]


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    manifest: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def normalize_path(value: str, source_root: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def chunk_ids_by_document(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_doc: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.get("document_id", ""))
        chunk_id = str(chunk.get("chunk_id") or chunk.get("text_id") or "")
        if doc_id and chunk_id:
            by_doc[doc_id].append(chunk_id)
    return {doc_id: sorted(ids) for doc_id, ids in by_doc.items()}


def score_source(row: dict[str, Any], chunk_count: int) -> dict[str, Any]:
    jurisdiction = str(row.get("jurisdiction", "")).lower()
    org = str(row.get("organization", "")).lower()
    status = str(row.get("review_status", "")).lower()
    extraction = str(row.get("extraction_status", "")).lower()
    license_note = str(row.get("license", ""))
    hazards = row.get("hazards", []) or []
    text_chars = int(row.get("text_chars") or row.get("char_count") or 0)
    can_retrieve = bool(row.get("can_retrieve", False))
    reject_reason = str(row.get("reject_reason", ""))

    authority_score = 0
    if any(term in org for term in ["national disaster management", "india meteorological", "ministry", "fssai"]):
        authority_score = 5
    elif any(term in org for term in ["world health", "unicef", "red cross", "sphere"]):
        authority_score = 4
    elif any(term in org for term in ["centers for disease", "food and drug", "environmental protection", "ready.gov", "weather service", "geological survey"]):
        authority_score = 4

    india_relevance_score = 5 if jurisdiction == "india" else 3 if jurisdiction in {"global", "us/global-applicable"} else 1
    extraction_ok = extraction == "ok" or extraction.startswith("ok_")
    extraction_score = 5 if extraction_ok and text_chars >= 1500 and chunk_count else 2 if extraction in {"deferred", "not_attempted"} else 0
    licensing_score = 5 if "public domain" in license_note.lower() else 3 if "official" in license_note.lower() or "public guidance" in license_note.lower() else 1
    live_risk_penalty = 2 if str(row.get("staleness_class", "")).lower() == "live" else 0
    rejected_penalty = 5 if reject_reason or status.startswith("rejected") else 0
    practical_score = min(5, max(1, len(hazards)))
    stability_score = 5 if str(row.get("staleness_class", "")).lower() == "evergreen" else 3
    score = (
        authority_score
        + india_relevance_score
        + practical_score
        + stability_score
        + extraction_score
        + licensing_score
        - live_risk_penalty
        - rejected_penalty
    )

    if reject_reason:
        recommended_status = "rejected"
    elif extraction == "ok" and can_retrieve and score >= 24:
        recommended_status = "accepted_core"
    elif extraction == "ok" and can_retrieve and score >= 19:
        recommended_status = "accepted_supporting"
    elif can_retrieve and score >= 16:
        recommended_status = "accepted_retrieval_only"
    else:
        recommended_status = "deferred"

    return {
        "authority_score": authority_score,
        "india_relevance_score": india_relevance_score,
        "practical_actionability_score": practical_score,
        "stability_score": stability_score,
        "extraction_quality_score": extraction_score,
        "licensing_storage_score": licensing_score,
        "live_fact_risk_penalty": live_risk_penalty,
        "rejection_penalty": rejected_penalty,
        "overall_score": score,
        "recommended_status": recommended_status,
    }


def candidate_from_document(row: dict[str, Any], source_root: Path, chunk_ids: list[str]) -> dict[str, Any]:
    extracted_path = normalize_path(str(row.get("extracted_path", "")), source_root)
    raw_path = normalize_path(str(row.get("raw_path", "")), source_root)
    score = score_source(row, len(chunk_ids))
    reason = str(row.get("reject_reason", ""))
    if not reason:
        reason = {
            "accepted_core": "High-authority, cleanly extracted, and useful for stable offline guidance.",
            "accepted_supporting": "Authoritative and useful, but less central or less India-specific than core sources.",
            "accepted_retrieval_only": "Useful for grounding but training/storage permissions or live-status risk require caution.",
            "deferred": "Potentially useful, but needs better extraction, licensing review, or narrower use.",
            "rejected": "Rejected by source-corpus review.",
        }[score["recommended_status"]]
    return {
        "document_id": row.get("document_id", ""),
        "source_id": row.get("source_id", ""),
        "url": row.get("url", ""),
        "title": row.get("title", ""),
        "organization": row.get("organization", ""),
        "jurisdiction": row.get("jurisdiction", ""),
        "language": row.get("language", ""),
        "hazards": row.get("hazards", []) or [],
        "license": row.get("license", ""),
        "terms_url": row.get("terms_url", ""),
        "copyright_status": row.get("copyright_status", ""),
        "staleness_class": row.get("staleness_class", ""),
        "published_date": row.get("published_date", ""),
        "retrieved_at": row.get("retrieved_at", ""),
        "can_store_raw": bool(row.get("can_store_raw", False)),
        "can_train": bool(row.get("can_train", False)),
        "can_retrieve": bool(row.get("can_retrieve", False)),
        "raw_path": raw_path,
        "extracted_path": extracted_path,
        "extraction_status": row.get("extraction_status", ""),
        "text_chars": int(row.get("text_chars") or row.get("char_count") or 0),
        "source_chunk_ids": chunk_ids,
        "review_status": score["recommended_status"],
        "status_reason": reason,
        "recommendation": "use_for_card_distillation" if score["recommended_status"] in {"accepted_core", "accepted_supporting"} else "keep_for_review",
        **score,
    }


def supplemental_candidate(row: dict[str, Any]) -> dict[str, Any]:
    score = score_source(row, 0)
    status = str(row.get("review_status") or score["recommended_status"])
    return {
        "terms_url": "",
        "published_date": "",
        "retrieved_at": "",
        "can_store_raw": False,
        "can_train": False,
        "can_retrieve": status in ACCEPTED_STATUSES,
        "raw_path": "",
        "extracted_path": "",
        "text_chars": 0,
        "source_chunk_ids": [],
        **row,
        **{key: value for key, value in score.items() if key != "recommended_status"},
        "review_status": status,
    }


def build_candidates(source_root: Path = DEFAULT_SOURCE_CORPUS) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    documents = read_jsonl(source_root / "document_cards.jsonl")
    rejected_documents = read_jsonl(source_root / "rejected" / "document_cards_rejected.jsonl")
    if not rejected_documents:
        rejected_documents = read_jsonl(source_root / "rejected_document_cards.jsonl")
    chunks = read_jsonl(source_root / "retrieval_chunks" / "retrieval_chunks.jsonl")
    chunk_map = chunk_ids_by_document(chunks)

    candidates = [
        candidate_from_document(row, source_root, chunk_map.get(str(row.get("document_id")), []))
        for row in documents
    ]
    candidates.extend(
        candidate_from_document({**row, "extraction_status": row.get("extraction_status", "rejected")}, source_root, [])
        for row in rejected_documents
    )
    known_ids = {str(row["document_id"]) for row in candidates}
    candidates.extend(supplemental_candidate(row) for row in SUPPLEMENTAL_CANDIDATES if row["document_id"] not in known_ids)

    candidates = sorted(candidates, key=lambda row: (-int(row.get("overall_score", 0)), str(row.get("document_id", ""))))
    accepted = [row for row in candidates if row["review_status"] in ACCEPTED_STATUSES]
    rejected = [row for row in candidates if row["review_status"] in {"rejected", "deferred"}]
    return candidates, accepted, rejected


def coverage_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = [row for row in candidates if row["review_status"] in ACCEPTED_STATUSES]
    rows: list[dict[str, Any]] = []
    for family, hazards in REQUIRED_HAZARD_FAMILIES.items():
        matched = [
            row
            for row in accepted
            if hazards.intersection(set(row.get("hazards", []) or []))
        ]
        rows.append(
            {
                "hazard_family": family,
                "required_terms": "|".join(sorted(hazards)),
                "accepted_source_count": len(matched),
                "core_source_count": sum(row["review_status"] == "accepted_core" for row in matched),
                "india_source_count": sum(row.get("jurisdiction") == "india" for row in matched),
                "source_ids": "|".join(sorted({str(row.get("document_id", "")) for row in matched})),
                "status": "covered" if matched else "gap",
            }
        )
    return rows


def write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["hazard_family", "required_terms", "accepted_source_count", "core_source_count", "india_source_count", "source_ids", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_manifest(candidates: list[dict[str, Any]], coverage: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    accepted = [row for row in candidates if row["review_status"] in ACCEPTED_STATUSES]
    by_status = Counter(str(row.get("review_status", "")) for row in candidates)
    by_jurisdiction = Counter(str(row.get("jurisdiction", "")) for row in accepted)
    by_org = Counter(str(row.get("organization", "")) for row in accepted)
    by_hazard = Counter(hazard for row in accepted for hazard in row.get("hazards", []) or [])
    gaps = [row["hazard_family"] for row in coverage if row["status"] != "covered"]
    return {
        "created_at_utc": utc_now(),
        "status": "valid" if not errors else "invalid",
        "phase": "source_research_only_no_grounding_cards",
        "candidate_source_count": len(candidates),
        "accepted_document_count": len(accepted),
        "downloaded_document_count": sum(
            bool(row.get("extracted_path"))
            and (str(row.get("extraction_status")) == "ok" or str(row.get("extraction_status")).startswith("ok_"))
            for row in accepted
        ),
        "rejected_or_deferred_count": len(candidates) - len(accepted),
        "coverage_gap_count": len(gaps),
        "coverage_gaps": gaps,
        "by_status": dict(by_status.most_common()),
        "accepted_by_jurisdiction": dict(by_jurisdiction.most_common()),
        "accepted_by_organization": dict(by_org.most_common()),
        "accepted_by_hazard": dict(by_hazard.most_common()),
        "validation": {"errors": errors, "warnings": warnings},
    }


def render_report(candidates: list[dict[str, Any]], coverage: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    accepted_core = [row for row in candidates if row["review_status"] == "accepted_core"]
    accepted_supporting = [row for row in candidates if row["review_status"] == "accepted_supporting"]
    deferred = [row for row in candidates if row["review_status"] in {"deferred", "rejected"}]
    lines = [
        "# Beacon Source Research v1",
        "",
        "This is a source-selection research pack for a future local grounding tool. It does not create grounding cards and should not be treated as a training approval.",
        "",
        "## Recommendation",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Candidate sources reviewed: {manifest['candidate_source_count']}",
        f"- Accepted for grounding research: {manifest['accepted_document_count']}",
        f"- Clean downloaded/extracted accepted documents: {manifest['downloaded_document_count']}",
        f"- Coverage gaps: {', '.join(manifest['coverage_gaps']) if manifest['coverage_gaps'] else 'none'}",
        "",
        "Proceed to compact card creation only after human review of `candidate_sources.jsonl`, especially the deferred India-specific gaps.",
        "",
        "## Core Sources",
        "",
    ]
    for row in accepted_core[:20]:
        lines.append(f"- `{row['document_id']}` - {row['title']} ({row['organization']}); hazards: {', '.join(row.get('hazards', []))}")
    lines.extend(["", "## Supporting Sources", ""])
    for row in accepted_supporting[:20]:
        lines.append(f"- `{row['document_id']}` - {row['title']} ({row['organization']}); hazards: {', '.join(row.get('hazards', []))}")
    lines.extend(["", "## Coverage Matrix Summary", ""])
    for row in coverage:
        lines.append(f"- `{row['hazard_family']}`: {row['status']} ({row['accepted_source_count']} accepted, {row['india_source_count']} India)")
    lines.extend(["", "## Deferred Or Rejected Sources To Review", ""])
    for row in deferred[:20]:
        lines.append(f"- `{row['document_id']}` - {row.get('status_reason') or row.get('reject_reason') or 'needs review'}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Use accepted core/supporting sources to create 20-30 compact grounding cards. Keep deferred/manual-review sources out of cards until extraction and licensing are resolved.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_candidates(candidates: list[dict[str, Any]], source_root: Path = DEFAULT_SOURCE_CORPUS) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    ids = [str(row.get("document_id", "")) for row in candidates]
    if len(ids) != len(set(ids)):
        errors.append("candidate document_id values must be unique")

    for row in candidates:
        doc_id = str(row.get("document_id", ""))
        status = str(row.get("review_status", ""))
        if status not in FINAL_STATUSES:
            errors.append(f"{doc_id}: invalid review_status {status!r}")
        for key in ["url", "organization", "jurisdiction", "hazards", "license"]:
            if not row.get(key):
                errors.append(f"{doc_id}: missing required field {key}")
        if status in ACCEPTED_STATUSES:
            if not row.get("source_chunk_ids"):
                errors.append(f"{doc_id}: accepted source must have at least one source_chunk_id")
            if not row.get("extracted_path"):
                errors.append(f"{doc_id}: accepted source must have extracted_path")
            extraction_status = str(row.get("extraction_status", ""))
            if extraction_status != "ok" and not extraction_status.startswith("ok_"):
                errors.append(f"{doc_id}: accepted source must have extraction_status=ok or ok_*")
            if "unknown" in str(row.get("license", "")).lower():
                errors.append(f"{doc_id}: accepted source cannot have unknown license")
        else:
            if not row.get("status_reason") and not row.get("reject_reason"):
                errors.append(f"{doc_id}: rejected/deferred source must have status_reason or reject_reason")
        for path_key in ["raw_path", "extracted_path"]:
            path_text = str(row.get(path_key, ""))
            normalized = path_text.replace("\\", "/")
            if any(part in normalized for part in IGNORED_PATH_PARTS):
                errors.append(f"{doc_id}: {path_key} points into ignored/runtime output: {path_text}")
        joined = json.dumps(row, ensure_ascii=False).lower()
        for pattern in LIVE_STATUS_PATTERNS:
            if pattern in joined:
                errors.append(f"{doc_id}: contains banned live-status phrase {pattern!r}")
        for pattern in MEDICINE_DOSE_PATTERNS:
            if pattern in joined:
                errors.append(f"{doc_id}: contains medicine dosing phrase {pattern!r}")

    coverage = coverage_rows(candidates)
    for row in coverage:
        if row["status"] != "covered":
            errors.append(f"required hazard family not covered: {row['hazard_family']}")
        elif row["india_source_count"] == 0 and row["hazard_family"] in {"cyclone_coastal", "heat_cold_lightning", "misinformation_live_status"}:
            warnings.append(f"{row['hazard_family']}: covered, but India-specific coverage should be manually reviewed")

    manifest = make_manifest(candidates, coverage, errors, warnings)
    return ValidationResult(errors=errors, warnings=warnings, manifest=manifest)


def build_research_pack(out_dir: Path = DEFAULT_OUT_DIR, source_root: Path = DEFAULT_SOURCE_CORPUS) -> ValidationResult:
    candidates, accepted, rejected = build_candidates(source_root)
    validation = validate_candidates(candidates, source_root)
    coverage = coverage_rows(candidates)
    manifest = make_manifest(candidates, coverage, validation.errors, validation.warnings)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "candidate_sources.jsonl", candidates)
    write_jsonl(out_dir / "downloaded_document_cards.jsonl", accepted)
    write_jsonl(out_dir / "rejected_sources.jsonl", rejected)
    write_coverage_csv(out_dir / "coverage_matrix.csv", coverage)
    write_json(out_dir / "manifest.json", manifest)
    (out_dir / "research_report.md").write_text(render_report(candidates, coverage, manifest), encoding="utf-8")
    return ValidationResult(errors=validation.errors, warnings=validation.warnings, manifest=manifest)
