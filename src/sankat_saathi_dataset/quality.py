from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any


UNSAFE_PATTERNS = [
    re.compile(r"\bdefinitely safe\b", re.I),
    re.compile(r"\b100%\s*safe\b", re.I),
    re.compile(r"\bguaranteed\b", re.I),
    re.compile(r"\bboiling removes all chemical", re.I),
    re.compile(r"\bsmell(s)? fine.*safe\b", re.I),
    re.compile(r"\breheat(ing)?.*always.*safe\b", re.I),
    re.compile(r"\bdiagnos(e|is)\b", re.I),
]

REQUIRED_RESPONSE_KEYS = {
    "risk_level",
    "immediate_action",
    "resource_plan",
    "unsafe_items",
    "missing_information",
    "escalation_signs",
    "what_not_to_do",
    "hindi_hinglish",
    "uncertainty_note",
}


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    return str(value)


def unsafe_hits(text: str) -> list[str]:
    return [pattern.pattern for pattern in UNSAFE_PATTERNS if pattern.search(text)]


def validate_structured_response(response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_RESPONSE_KEYS.difference(response)
    if missing:
        errors.append(f"missing response keys: {sorted(missing)}")
    risk_level = response.get("risk_level")
    if risk_level not in {"low", "caution", "high", "critical", "cannot_determine"}:
        errors.append(f"invalid risk_level: {risk_level}")
    for key in REQUIRED_RESPONSE_KEYS - {"risk_level", "uncertainty_note"}:
        if not isinstance(response.get(key), list) or not response.get(key):
            errors.append(f"{key} must be a non-empty list")
    hits = unsafe_hits(flatten_text(response))
    if hits:
        errors.append(f"unsafe phrasing found: {hits}")
    return errors


def validate_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not example.get("example_id") and not example.get("pair_id"):
        errors.append("missing id")
    if not example.get("source_ids"):
        errors.append("missing source_ids")
    if not example.get("guidance_fact_ids"):
        errors.append("missing guidance_fact_ids")
    if "assistant_response" in example:
        errors.extend(validate_structured_response(example["assistant_response"]))
    if "chosen" in example:
        errors.extend(validate_structured_response(example["chosen"]))
        rejected_hits = unsafe_hits(example.get("rejected", ""))
        if not rejected_hits and not example.get("rejection_reasons"):
            errors.append("dpo rejected answer needs unsafe pattern or rejection reasons")
    return errors


def dataclass_to_record(item: Any) -> dict[str, Any]:
    return asdict(item)
