from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RiskLevel = Literal["low", "caution", "high", "critical", "cannot_determine"]
Modality = Literal["text", "vision"]


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    url: str
    organization: str
    domain: str
    date: str
    license_notes: str
    use_notes: str


@dataclass(frozen=True)
class GuidanceFact:
    fact_id: str
    source_ids: list[str]
    hazard_type: str
    confidence_category: RiskLevel
    guidance: str
    allowed_advice: list[str]
    forbidden_claims: list[str]
    escalation_triggers: list[str]
    tags: list[str]
    accessed_at: str = ""
    published_at: str = ""
    source_section: str = ""
    jurisdiction: str = ""
    evidence_notes: str = ""
    source_ready: bool = False


@dataclass(frozen=True)
class ImageMetadata:
    image_id: str
    source_url: str
    license: str
    license_url: str
    author: str
    retrieved_at: str
    modifications: str
    visible_labels: list[str]
    provided_context_labels: list[str]
    not_determinable_labels: list[str]
    local_path: str = ""
    split_group: Literal["train", "eval", "shared"] = "shared"
    event_id: str = ""
    hazard_type: str = ""
    manifest_ready: bool = False


@dataclass(frozen=True)
class StructuredAnswer:
    risk_level: RiskLevel
    immediate_action: list[str]
    resource_plan: list[str]
    unsafe_items: list[str]
    missing_information: list[str]
    escalation_signs: list[str]
    what_not_to_do: list[str]
    hindi_hinglish: list[str]
    uncertainty_note: str


@dataclass(frozen=True)
class SftExample:
    example_id: str
    split: Literal["train", "eval"]
    modality: Modality
    source_ids: list[str]
    guidance_fact_ids: list[str]
    user_prompt: str
    assistant_response: StructuredAnswer
    risk_tags: list[str]
    language_mix: Literal["english", "hinglish", "bilingual"]
    image_path: str | None = None
    image_observations: list[str] = field(default_factory=list)
    image_uncertainty: str | None = None
    image_metadata: ImageMetadata | None = None
    eval_rubric: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DpoPair:
    pair_id: str
    source_ids: list[str]
    guidance_fact_ids: list[str]
    prompt: str
    chosen: StructuredAnswer
    rejected: str
    rejection_reasons: list[str]
    risk_tags: list[str]
    image_path: str | None = None
    target_failure_mode: str = ""


def to_jsonable(item: Any) -> dict[str, Any]:
    return asdict(item)
