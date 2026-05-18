from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1"
VISIBLE_SCAFFOLD_RE = re.compile(
    r"(returned snippet|returned section|use the returned|keep the answer limited|"
    r"tool query|rewrite this|normalize this|source fact|evidence source|"
    r"document-backed point|keep the final wording|keep it suitable|"
    r"leave out details that are not supported|for the [a-z -]+, keep the next step practical|"
    r"as the offline source|offline source|evidence:|document-based|stable reference|"
    r"the helpful answer is a short|that keeps the decision usable)",
    re.I,
)
UNSUPPORTED_LIVE_RE = re.compile(r"\b(open now|available now|rescue.*near|rescue.*one hour|beds tonight|safe tonight)\b", re.I)
ABSTAIN_RE = re.compile(r"\b(cannot|can't|not enough|do not invent|cannot confirm|nahi|mat guess|verify nahi|confirm nahi)\b", re.I)
EXACT_CLAIM_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:hours?|minutes?|days?|feet|degrees?|grams?|g|percent)|"
    r"40\s*degrees|20\s*feet|15\s*(?:grams?|g)|1\s*minute|3\s*minutes|30\s*minutes)\b",
    re.I,
)
NEGATED_MYTHS = {"60 degrees", "72 hours", "3 days", "3 din", "30-minute"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def target_stem(row: dict[str, Any], words: int = 14) -> str:
    return " ".join(str(row.get("target_response", "")).split()[:words]).lower()


def tool_result_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for message in row.get("messages", []):
        if message.get("role") != "tool" or message.get("name") != "read_official_doc":
            continue
        try:
            payload = json.loads(str(message.get("content", "{}")))
        except json.JSONDecodeError:
            continue
        sections.extend(payload.get("sections", []))
    return sections


def evidence_text(row: dict[str, Any]) -> str:
    chunks: list[str] = []
    for section in tool_result_sections(row):
        chunks.append(str(section.get("snippet", "")))
        chunks.extend(str(fact) for fact in section.get("key_facts", []))
    return " ".join(chunks).lower()


def row_review(row: dict[str, Any], stem_counts: Counter[str]) -> dict[str, Any]:
    reasons: list[str] = []
    risk_tags: list[str] = []
    target = str(row.get("target_response", ""))
    target_lower = target.lower()
    family = str(row.get("row_family", ""))
    case_family_id = str(row.get("case_family_id", ""))
    hazard = str(row.get("hazard", ""))

    if row.get("tool_required") and not row.get("section_ids"):
        reasons.append("tool_required row has no read section evidence")
        risk_tags.append("invalid_tool_trace")
    if family.endswith("no_support"):
        if not ABSTAIN_RE.search(target):
            reasons.append("no-support row does not clearly abstain in final answer")
            risk_tags.append("weak_abstention")
        if EXACT_CLAIM_RE.search(target):
            reasons.append("no-support row introduces an exact numeric claim")
            risk_tags.append("unsupported_precision")
        if "evidence:" in target_lower:
            reasons.append("no-support row cites evidence despite abstaining")
            risk_tags.append("confusing_citation")
    if VISIBLE_SCAFFOLD_RE.search(target):
        reasons.append("visible scaffold/template language in assistant answer")
        risk_tags.append("scaffold_language")
    stem = target_stem(row)
    if stem_counts[stem] >= 8:
        reasons.append(f"answer stem repeated {stem_counts[stem]} times")
        risk_tags.append("repetition")
    if family.startswith("query_rewrite") and re.search(r"\brewrite|normalize|tool query\b", target_lower):
        reasons.append("assistant final answer talks about rewrite/tool process")
        risk_tags.append("visible_tool_process")
    if "according to the offline official-doc sections returned" in target_lower:
        reasons.append("assistant exposes internal evidence plumbing")
        risk_tags.append("visible_tool_process")
    if row.get("tool_required"):
        evidence = evidence_text(row)
        for claim in EXACT_CLAIM_RE.findall(target):
            claim_text = claim.lower()
            start = target_lower.find(claim_text)
            window = target_lower[max(0, start - 48): start + len(claim_text) + 48] if start >= 0 else ""
            if claim_text in NEGATED_MYTHS and re.search(r"\b(no|not|nahi|mat|do not|don't|unsafe|assume)\b", window):
                continue
            if claim_text and claim_text not in evidence:
                reasons.append(f"exact claim not found in serialized evidence: {claim_text}")
                risk_tags.append("unsupported_precision")
    if UNSUPPORTED_LIVE_RE.search(target) and not family.endswith("no_support"):
        reasons.append("live/current status phrasing appears outside no-support family")
        risk_tags.append("live_fact_risk")
    if case_family_id.startswith("doc_index_"):
        if semantically_suspect_doc_index(hazard, target_lower):
            reasons.append("doc-index row attaches an exact fact to the wrong hazard/task")
            risk_tags.append("semantic_grounding_mismatch")
    if family.endswith("no_support") and weak_generic_no_support(case_family_id, target_lower):
        reasons.append("no-support row is too generic for the specific unsafe request")
        risk_tags.append("weak_abstention")

    if any(tag in risk_tags for tag in ["invalid_tool_trace", "unsupported_precision", "live_fact_risk", "semantic_grounding_mismatch"]):
        status = "rejected"
    elif reasons:
        status = "rewrite"
    else:
        status = "approved"

    return {
        "row_id": row.get("row_id", ""),
        "split": row.get("split", ""),
        "row_family": family,
        "case_family_id": row.get("case_family_id", ""),
        "hazard": row.get("hazard", ""),
        "review_status": status,
        "review_reasons": reasons,
        "risk_tags": sorted(set(risk_tags)),
        "user_prompt": row.get("user_prompt", ""),
        "target_response": target,
    }


def semantically_suspect_doc_index(hazard: str, target_lower: str) -> bool:
    hazard = hazard.lower()
    fact_hazard_pairs = [
        ("40 degrees", {"food_safety", "heatwave"}),
        ("4 hours", {"food_safety", "power_outage"}),
        ("48 hours", {"food_safety", "power_outage"}),
        ("24 hours", {"food_safety", "power_outage"}),
        ("20 feet", {"carbon_monoxide", "generators", "power_outage"}),
        ("15 grams", {"diabetes", "medicine_disruption"}),
        ("30 minutes", {"water_safety", "disinfection", "chemical_contamination"}),
        ("1 minute", {"water_safety", "boil_water"}),
        ("3 minutes", {"water_safety", "boil_water"}),
        ("rolling boil", {"water_safety", "boil_water"}),
        ("turn around, do not drown", {"flood", "floodwater", "route_safety"}),
    ]
    for fact, allowed in fact_hazard_pairs:
        if fact in target_lower and hazard not in allowed:
            return True
    return False


def weak_generic_no_support(case_family_id: str, target_lower: str) -> bool:
    if "cannot verify that specific claim" in target_lower:
        return True
    if ("wall" in case_family_id or "structural" in case_family_id or "slope" in case_family_id) and not re.search(
        r"\b(certify|safe|sleep|re-entry|risky area|assessment)\b", target_lower
    ):
        return True
    if "lightning" in case_family_id and not re.search(r"\b(lightning|water|open areas|shelter)\b", target_lower):
        return True
    if ("medicine" in case_family_id or "injection" in case_family_id or "pharmacy" in case_family_id) and not re.search(
        r"\b(medicine|tablet|dose|pharmacist|doctor|health worker|injection|prescription)\b", target_lower
    ):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", nargs="?", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    package = args.package
    rows = read_jsonl(package / "all_rows.jsonl")
    stem_counts = Counter(target_stem(row) for row in rows)
    reviews = [row_review(row, stem_counts) for row in rows]
    by_status = Counter(row["review_status"] for row in reviews)
    by_reason = Counter(reason for row in reviews for reason in row["review_reasons"])
    by_family_status = Counter((row["row_family"], row["review_status"]) for row in reviews)
    manifest = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_package": str(package),
        "row_count": len(reviews),
        "by_status": dict(by_status.most_common()),
        "by_family_status": {f"{family}::{status}": count for (family, status), count in by_family_status.most_common()},
        "top_reasons": dict(by_reason.most_common(30)),
        "policy": (
            "Strict prototype review: approve only rows with natural user-facing answers, valid tool traces, "
            "no unsupported exact claims, no visible scaffold language, and no high repetition."
        ),
        "training_recommendation": "Do not train on rows marked rewrite or rejected.",
    }
    write_jsonl(package / "row_quality_review_v1.jsonl", reviews)
    write_json(package / "row_quality_review_manifest_v1.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
