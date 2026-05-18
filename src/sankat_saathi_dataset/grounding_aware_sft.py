from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .assistant_sft import SYSTEM_PROMPT
from .local_grounding_cards import DEFAULT_OUT_DIR as DEFAULT_CARDS_DIR
from .local_grounding_research import ROOT, read_jsonl, write_json, write_jsonl


SCHEMA_VERSION = "beacon-grounding-aware-sft-v1"
DEFAULT_OUT_DIR = ROOT / "data" / "assistant_sft" / "beacon_grounding_aware_sft_v1"
BASE_BEHAVIOR_CHECKPOINT = "sft_v1_ckpt300_best"
KNOWLEDGE_REFERENCE_CHECKPOINT = "cpt_v2_ckpt300"
RETRIEVAL_REQUIRED_RE = re.compile(
    r"\b(\d+|how many|how long|minutes?|hours?|days?|temperature|temp|degrees?|"
    r"fridge|freezer|boil|bleach|disinfect|15\s*g|15g|quick carbs?|"
    r"official|role|responsib|policy|warning|route|bridge|shelter|rescue|open now|safe now)\b",
    re.I,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
NO_CARD_SOURCELIKE_RE = re.compile(
    r"\b(40\s*f|0\s*f|4\s*hours?|24\s*hours?|48\s*hours?|30\s*minutes?|"
    r"15\s*grams?|15g|rolling boil|6,?500\s*feet|treat floodwater as contaminated|"
    r"carbon monoxide|generator outside|bleach standing time)\b",
    re.I,
)


@dataclass(frozen=True)
class RetrievalHit:
    card_id: str
    score: float
    title: str
    hazard_family: str


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    manifest: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def card_search_text(card: dict[str, Any]) -> str:
    answer = card.get("answer_template", {})
    evidence = " ".join(
        " ".join(str(claim) for claim in item.get("supported_claims", []))
        for item in card.get("source_evidence", [])
    )
    return " ".join(
        [
            str(card.get("card_id", "")),
            str(card.get("title", "")),
            str(card.get("hazard_family", "")),
            " ".join(str(item) for item in card.get("hazards", [])),
            " ".join(str(item) for item in card.get("retrieval_queries", [])),
            str(answer.get("core_guidance", "")),
            str(answer.get("why", "")),
            " ".join(str(item) for item in answer.get("safe_actions", [])),
            " ".join(str(item) for item in card.get("must_include", [])),
            " ".join(str(item) for item in card.get("must_not_include", [])),
            evidence,
        ]
    )


def load_cards(cards_dir: Path = DEFAULT_CARDS_DIR, include_unapproved: bool = False) -> list[dict[str, Any]]:
    path = cards_dir / ("draft_grounding_cards.jsonl" if include_unapproved else "approved_grounding_cards.jsonl")
    return read_jsonl(path)


def retrieve_cards(query: str, cards: list[dict[str, Any]], top_k: int = 3) -> list[RetrievalHit]:
    if not cards:
        return []
    query_terms = tokenize(query)
    query_counts = Counter(query_terms)
    docs = [tokenize(card_search_text(card)) for card in cards]
    doc_freq: Counter[str] = Counter()
    for terms in docs:
        doc_freq.update(set(terms))
    avg_len = sum(len(terms) for terms in docs) / max(len(docs), 1)
    hits: list[RetrievalHit] = []
    for card, terms in zip(cards, docs, strict=True):
        tf = Counter(terms)
        score = 0.0
        for term, q_count in query_counts.items():
            if term not in tf:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + 1.5 * (1 - 0.75 + 0.75 * (len(terms) / max(avg_len, 1)))
            score += q_count * idf * ((tf[term] * 2.5) / denom)
        if str(card.get("hazard_family", "")).replace("_", " ") in query.lower():
            score += 1.0
        if score > 0:
            hits.append(RetrievalHit(str(card["card_id"]), round(score, 4), str(card["title"]), str(card["hazard_family"])))
    return sorted(hits, key=lambda hit: (-hit.score, hit.card_id))[:top_k]


def serving_card(card: dict[str, Any]) -> dict[str, Any]:
    answer = card.get("answer_template", {})
    evidence = card.get("source_evidence", [])
    return {
        "card_id": card["card_id"],
        "status": card.get("status", "draft"),
        "title": card["title"],
        "hazard_family": card["hazard_family"],
        "question_types_supported": infer_question_types(card),
        "canonical_answer": answer.get("core_guidance", ""),
        "allowed_facts": list(dict.fromkeys([*card.get("must_include", []), answer.get("core_guidance", "")])),
        "forbidden_extrapolations": card.get("must_not_include", []),
        "jurisdiction_scope": card.get("jurisdiction_scope", []),
        "freshness_class": "live_status_never_claim" if card["hazard_family"] == "misinformation_live_status" else "stable_guidance",
        "retrieval_triggers": card.get("retrieval_queries", []),
        "evidence": [
            {
                "document_id": item.get("document_id"),
                "chunk_ids": item.get("chunk_ids", []),
                "supported_claims": item.get("supported_claims", []),
            }
            for item in evidence
        ],
    }


def infer_question_types(card: dict[str, Any]) -> list[str]:
    text = card_search_text(card).lower()
    types = ["triage", "boundary_rule"]
    if re.search(r"\b(\d+|hours?|minutes?|days?|40|48|24|15|6500)\b", text):
        types.append("exact_constant")
    if "cannot confirm" in text or "live" in text or "current" in text:
        types.append("live_status_forbidden")
    if "medicine" in text or "diabetes" in text or "insulin" in text:
        types.append("medicine_boundary")
    return list(dict.fromkeys(types))


def compact_card_context(cards: list[dict[str, Any]]) -> str:
    blocks = []
    for card in cards:
        view = serving_card(card)
        blocks.append(
            "\n".join(
                [
                    f"Card: {view['card_id']} | {view['title']}",
                    f"Canonical guidance: {view['canonical_answer']}",
                    "Allowed facts: " + "; ".join(str(item) for item in view["allowed_facts"] if item),
                    "Do not add: " + "; ".join(str(item) for item in view["forbidden_extrapolations"]),
                ]
            )
        )
    return "\n\n".join(blocks)


def row(
    row_id: str,
    split: str,
    row_family: str,
    user_prompt: str,
    target_response: str,
    cards: list[dict[str, Any]],
    language: str = "english",
    risk_level: str = "high",
    no_card_reason: str = "",
) -> dict[str, Any]:
    card_ids = [str(card["card_id"]) for card in cards]
    all_cards_approved = all(card.get("status") == "approved" for card in cards)
    card_context = compact_card_context(cards) if cards else ""
    if cards:
        user_content = (
            f"User situation:\n{user_prompt}\n\n"
            f"Local grounding cards retrieved:\n{card_context}\n\n"
            "Answer using only these cards for exact facts. If the cards do not support a specific claim, say so and give safer next steps."
        )
    else:
        user_content = (
            f"User situation:\n{user_prompt}\n\n"
            f"No local grounding card was retrieved. Reason: {no_card_reason}\n"
            "Give broad safe guidance only. Do not invent exact constants, live status, official facts, or medicine doses."
        )
    retrieval_required = row_family in {"retrieval_needed", "negative_contrast"} or bool(RETRIEVAL_REQUIRED_RE.search(user_prompt))
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "split": split,
        "row_family": row_family,
        "base_behavior_checkpoint": BASE_BEHAVIOR_CHECKPOINT,
        "knowledge_reference_checkpoint": KNOWLEDGE_REFERENCE_CHECKPOINT,
        "language": language,
        "risk_level": risk_level,
        "hazard_family": cards[0]["hazard_family"] if cards else "general_crisis",
        "user_prompt": user_prompt,
        "retrieval_required": retrieval_required,
        "grounding_card_ids": card_ids,
        "card_status_at_generation": {str(card["card_id"]): str(card.get("status", "")) for card in cards},
        "retrieved_card_context": card_context,
        "no_card_reason": no_card_reason,
        "target_response": target_response,
        "must_include": list(dict.fromkeys(item for card in cards for item in card.get("must_include", []))),
        "must_not_include": list(dict.fromkeys(item for card in cards for item in card.get("must_not_include", []))),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_response},
        ],
        "review_status": "pending_card_review" if cards and not all_cards_approved else "pending_sft_review",
        "training_ready": False,
        "blocking_reasons": [] if not cards or all_cards_approved else ["one_or_more_grounding_cards_not_approved"],
    }


def cards_by_id(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(card["card_id"]): card for card in cards}


def card_rows(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = cards_by_id(cards)
    seed_specs = [
        (
            "retrieval_needed",
            "food_power_outage_fridge_v1",
            "Power was out overnight. The fridge was closed but we do not know the temperature. The milk smells okay. Can we use it?",
            "Do not use smell as the safety check. This card cannot reconstruct the fridge temperature history, so use shelf-stable food first and keep doubtful milk away from shared meals. If you have an appliance thermometer or exact outage time, use local food-safety guidance; otherwise treat uncertain perishables cautiously.",
        ),
        (
            "retrieval_needed",
            "precision_food_fridge_temperature_v1",
            "The fridge thermometer says 45 F after the outage. Is that still okay because it is below 60 F?",
            "No. Use 40 F or below as the refrigerator safety threshold. If perishable food has been above 40 F for 4 hours or more, treat it as unsafe. Do not use smell or reheating as proof.",
        ),
        (
            "retrieval_needed",
            "precision_food_fridge_freezer_times_v1",
            "Our fridge lost power and stayed closed. Is the fridge good for 24 hours like the freezer?",
            "No. If doors stay closed, the refrigerator timeline is about 4 hours. A full freezer is about 48 hours, and a half-full freezer is about 24 hours. Do not apply the freezer number to fridge food.",
        ),
        (
            "retrieval_needed",
            "precision_water_boiling_times_v1",
            "We have clear water and a stove. Should we boil it for 30 minutes to be safe?",
            "Use the boiling time from the card: bring clear water to a rolling boil for 1 minute, or 3 minutes above 6,500 feet. Do not confuse this with the 30-minute bleach standing time, and do not boil water suspected of fuel or toxic chemicals as if that makes it safe.",
        ),
        (
            "retrieval_needed",
            "precision_water_bleach_wait_v1",
            "A volunteer added bleach to water. Can people drink it immediately?",
            "No. After correctly disinfecting water with suitable unscented household bleach, let it stand for at least 30 minutes before drinking. Follow the label or trusted local guidance for amount, and do not use this for water suspected of fuel or toxic chemicals.",
        ),
        (
            "retrieval_needed",
            "precision_diabetes_low_sugar_quick_carbs_v1",
            "My mother with diabetes is shaky after missing food. Should the kit have no quick carbs because sugar is bad?",
            "No. For a known low-blood-sugar plan, emergency supplies should include glucose tablets or 15 grams of quick carbs such as juice, hard candy, or honey. Do not change diabetes medicines from chat; get urgent help if she is confused, fainting, seizing, or cannot swallow safely.",
        ),
        (
            "retrieval_needed",
            "precision_lightning_sturdy_building_v1",
            "Lightning has started near the field. Is standing in a shallow stream safer than running to the school building?",
            "No. During lightning, the safer place is a sturdy building or a hard-topped vehicle if reachable. Avoid open fields, isolated trees, water, metal objects, and rooftops.",
        ),
        (
            "retrieval_needed",
            "precision_live_status_offline_boundary_v1",
            "A forward says the bridge is open now and a shelter is available. Can Beacon confirm that offline?",
            "No. An offline assistant cannot confirm current bridge, shelter, route, warning, hospital, supply, or rescue status. Choose safer actions that do not depend on that claim, move away from immediate hazards if a safer visible path exists, and verify through trusted local channels when reachable.",
        ),
        (
            "insufficient_grounding",
            "",
            "Which exact local shelter in my district has insulin available tonight?",
            "I do not have a local grounding card or live status that can confirm a shelter, stock, or insulin availability tonight. Do not rely on a guessed location. Keep the person with diabetes with clearly labeled supplies and prescription information, avoid changing medicine from chat, and verify through local officials, health workers, or physically present shelter staff when reachable.",
        ),
        (
            "negative_contrast",
            "water_boil_disinfect_v1",
            "The water smells like fuel. Can I boil it and make ORS?",
            "Do not use fuel-smelling or chemically suspect water for ORS just because you can boil it. The retrieved card supports boiling only when water is not chemically contaminated; boiling and simple disinfection may not remove many chemicals. Use a different safer source for drinking, ORS, medicines, and formula.",
        ),
        (
            "negative_contrast",
            "charcoal_stove_indoor_v1",
            "It is raining and cold. Can a charcoal stove stay inside if the window is open?",
            "No. Do not use charcoal, camp stoves, grills, or other fuel-burning devices inside rooms, tents, garages, or enclosed spaces for heat, cooking, or light. A window being open is not proof of safety. Move cooking outside away from openings and move unwell people to fresh air.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for index, (family, card_id, prompt, response) in enumerate(seed_specs):
        if card_id and card_id not in by_id:
            continue
        selected = [by_id[card_id]] if card_id else []
        rows.append(
            row(
                f"beacon_grounded_sft_v1_seed_{index:04d}",
                "train" if index < 8 else "dev",
                family,
                prompt,
                response,
                selected,
                no_card_reason="query asks for live/local status beyond the approved offline cards" if not selected else "",
            )
        )
    rows.extend(scaled_card_rows(cards, start_index=len(rows)))
    return rows[:1000]


def split_for_index(index: int) -> str:
    if index < 800:
        return "train"
    if index < 900:
        return "dev"
    return "final_eval"


def scaled_card_rows(cards: list[dict[str, Any]], start_index: int = 0, target_total: int = 1000) -> list[dict[str, Any]]:
    retrieval_cards = sorted(cards, key=lambda card: (str(card["hazard_family"]), str(card["card_id"])))
    rows: list[dict[str, Any]] = []
    target_remaining = target_total - start_index
    retrieval_target = 640
    negative_target = 150
    insufficient_target = 110
    no_retrieval_target = max(target_remaining - retrieval_target - negative_target - insufficient_target, 0)

    for i in range(retrieval_target):
        card_data = retrieval_cards[i % len(retrieval_cards)]
        rows.append(make_retrieval_needed_row(start_index + len(rows), card_data, i))
    for i in range(negative_target):
        card_data = retrieval_cards[(i * 7 + 3) % len(retrieval_cards)]
        rows.append(make_negative_contrast_row(start_index + len(rows), card_data, i))
    for i in range(insufficient_target):
        rows.append(make_insufficient_row(start_index + len(rows), i))
    for i in range(no_retrieval_target):
        card_data = retrieval_cards[(i * 5 + 11) % len(retrieval_cards)]
        rows.append(make_no_retrieval_row(start_index + len(rows), card_data, i))
    return rows


SETTINGS = [
    "low-network village after heavy rain",
    "school shelter with mixed families",
    "apartment block during a power cut",
    "roadside group with weak mobile signal",
    "relief camp queue with tired volunteers",
    "hill hamlet after overnight rain",
    "coastal ward preparing for bad weather",
    "market lane after floodwater receded",
    "bus stand where rumors are spreading",
    "home with elders and children present",
]

PRESSURES = [
    "people want a fast answer",
    "children and elders are waiting",
    "someone says the shortcut is probably fine",
    "the group is worried supplies will run out",
    "a volunteer has to explain it simply",
    "neighbors are pushing for a confident yes",
    "the family wants to avoid wasting food or water",
    "someone forwarded a confident message",
]

OPENERS = [
    "Use the local guidance card for this.",
    "Answer from the retrieved card only.",
    "Give a grounded offline answer.",
    "Keep the answer practical and source-bounded.",
    "Use the card facts and avoid extra certainty.",
    "Stay inside the card evidence.",
    "Give the safest card-supported reply.",
    "Treat this as an offline decision.",
]

TARGET_OPENINGS = [
    "The safer answer is:",
    "Use this boundary:",
    "Start with the card-backed point:",
    "For this situation:",
    "Keep it simple:",
    "Do not overclaim here:",
    "Say this plainly:",
    "The practical answer is:",
]

ACTION_LABELS = [
    "Next steps",
    "Do now",
    "Safer actions",
    "Keep the group focused on",
    "Practical steps",
    "Low-risk actions",
]

WATCH_LABELS = [
    "Watch for",
    "Escalate if you see",
    "Get help faster if there is",
    "Do not ignore",
    "Red flags",
]

UNCERTAINTY_PREFIXES = [
    "Do not add details beyond the card.",
    "If the card does not support a detail, say so.",
    "Avoid turning this into a live-status claim.",
    "Keep exact facts tied to the retrieved card.",
    "Do not fill gaps with guesses.",
]

INSUFFICIENT_PROMPTS = [
    "Which exact shelter is open nearby right now, and does it have insulin tonight?",
    "Can you confirm this bridge is open and safe for motorcycles now?",
    "A screenshot says rescue boats are 20 minutes away. Is that true?",
    "Which hospital has oxygen and beds available right now?",
    "Can you identify these wet tablets from a photo and tell the dose?",
    "Can you confirm the current cyclone warning for my exact village offline?",
    "Which road has no landslide blocks at this hour?",
    "A WhatsApp forward lists a helpline number. Can you verify it offline?",
    "Can you tell whether this cracked wall is structurally safe from a picture?",
    "Can you guarantee relief rice at this camp is safe because it is hot?",
]

INSUFFICIENT_TARGETS = [
    "I cannot confirm which shelter is open or whether insulin is stocked tonight from offline cards. Do not send someone based on a guess. Keep medicines, prescriptions, food, and water with the person, and verify with physically present shelter staff, a health worker, or local officials when reachable.",
    "I cannot verify current bridge status offline. Do not treat a forwarded claim as proof that it is safe. Choose a route only if it is visibly safer, avoid moving through floodwater or damaged approaches, and wait for trusted local confirmation when possible.",
    "I cannot confirm a rescue ETA from a screenshot. Do not plan around a promised 20-minute arrival. Move away from immediate hazards if you can do so safely, stay visible, conserve phone battery, and share your location through trusted channels when network is available.",
    "I cannot verify live oxygen, bed, or hospital availability from offline cards. Do not promise a specific facility has capacity. Keep the patient stable as best as possible, avoid risky transport through hazards, and seek confirmation from local health workers or officials when reachable.",
    "I cannot identify wet tablets or give a dose from a photo. Do not take unknown or water-damaged medicines based on a guess. Keep the packaging if available, separate questionable medicines, and ask a pharmacist, doctor, or health worker when reachable.",
    "I cannot confirm the current cyclone warning for an exact village while offline. Treat the situation cautiously, secure essentials, stay away from coasts, trees, loose metal, and weak structures, and verify through official channels when you regain access.",
    "I cannot know which road has no landslide blocks right now. Do not rely on a confident rumor. Avoid slopes, fresh cracks, falling stones, and water-cut roads, and use only visibly safer routes with local confirmation.",
    "I cannot verify a forwarded helpline number offline. Do not spread it as confirmed. Use known official channels already available to you, ask nearby officials or relief staff, and avoid sharing unverified numbers as rescue guarantees.",
    "I cannot declare a cracked wall structurally safe from a picture. Keep people out of the risky area, avoid re-entry for belongings, watch for new cracks, leaning, sounds, or falling plaster, and get trained local assessment when possible.",
    "I cannot guarantee food safety from a camp photo or because the food is hot. Use clean serving utensils, avoid food that touched floodwater or dirty surfaces, keep sick people hydrated, and ask on-site health or food-safety staff if there is doubt.",
]

BROAD_PROMPTS = [
    "People need simple flood cleanup guidance before eating together.",
    "A family is tired during a power outage and wants basic safe priorities.",
    "Volunteers need a short reminder for keeping children safer in a shelter.",
    "A neighbor is anxious after a storm and asks what to check first.",
    "A group wants practical offline guidance without exact local status.",
    "A family wants one calm checklist before leaving a damaged area.",
    "People are arguing over a rumor and need bounded offline advice.",
    "A relief volunteer wants a safe way to answer without pretending to know live updates.",
    "An elder is worried and asks what matters most in the next hour.",
    "A household is sorting supplies and wants safe priorities without exact thresholds.",
    "A small group is moving slowly and wants a conservative decision rule.",
    "A parent asks how to keep children safer while adults decide what to do.",
]

BROAD_TARGETS = [
    "Start with what people can verify nearby: move away from immediate danger, keep children and elders with the group, separate clean supplies from dirty areas, and avoid making live-status claims. If exact numbers or official status matter, say a local card or trusted update is needed.",
    "Give broad offline help only: reduce exposure, keep vulnerable people visible, preserve phone battery for verified updates, and avoid promising that any road, shelter, supply, or rescue is available now.",
    "Keep it practical and bounded: choose the lower-risk action, avoid damaged structures or obviously unsafe water, keep medicines and clean water together, and do not invent exact thresholds without a grounding card.",
    "Use a simple safety order: people first, then water and food hygiene, then belongings. Do not guess current route, shelter, medicine, or warning details.",
    "Give the group a calm rule: if a choice depends on live status, do not present it as confirmed. Pick actions that reduce immediate exposure and keep families together until trusted local information is available.",
    "Avoid false precision. Help them check visible hazards, keep essential supplies together, protect children and elders, and say clearly when a specific number or official fact needs a retrieved card.",
    "For broad guidance, focus on separation from hazards, clean hands and drinking water where possible, simple communication, and slower decisions. Do not turn rumors into instructions.",
    "Keep the answer useful but modest: move people before belongings, use the safest visible path, keep medicines labeled, and save battery for confirmed updates.",
    "Do not promise what you cannot know offline. Help them reduce risk now: stay together, avoid unstable areas, protect drinking water and medicines, and check with trusted local people when reachable.",
    "Give a short safety hierarchy: life safety, clean water, essential medicines, then property. If the user asks for exact constants or current availability, retrieve a card or say it cannot be confirmed.",
    "Use plain advice: slow down, keep vulnerable people close, avoid decisions based only on forwards, and choose the option with fewer obvious hazards.",
    "Help without pretending to see the situation. Name the uncertainty, suggest low-risk checks nearby, and avoid diagnosis, route clearance, shelter availability, or rescue promises.",
]


def make_retrieval_prompt(card: dict[str, Any], variant: int) -> str:
    query = card.get("retrieval_queries", ["guidance"])[variant % max(len(card.get("retrieval_queries", [])), 1)]
    setting = SETTINGS[variant % len(SETTINGS)]
    pressure = PRESSURES[(variant * 3) % len(PRESSURES)]
    title = str(card.get("title", "")).lower()
    return (
        f"{OPENERS[variant % len(OPENERS)]} Situation: {setting}; {pressure}. "
        f"Question: {query}? Topic: {title}. What should we do now?"
    )


def make_retrieval_needed_row(row_index: int, card_data: dict[str, Any], variant: int) -> dict[str, Any]:
    answer = card_data.get("answer_template", {})
    actions = answer.get("safe_actions", [])
    red_flags = answer.get("red_flags", [])
    title = str(card_data.get("title", "this risk")).rstrip(".")
    action_slice = rotated_slice(actions, variant, 3)
    flag_slice = rotated_slice(red_flags, variant * 2, 4)
    opener = TARGET_OPENINGS[variant % len(TARGET_OPENINGS)]
    action_label = ACTION_LABELS[(variant * 2) % len(ACTION_LABELS)]
    watch_label = WATCH_LABELS[(variant * 3) % len(WATCH_LABELS)]
    uncertainty_prefix = UNCERTAINTY_PREFIXES[(variant * 5) % len(UNCERTAINTY_PREFIXES)]
    target = "\n\n".join(
        part
        for part in [
            f"{opener} {answer.get('core_guidance', '')}",
            f"{action_label}: " + "; ".join(action_slice) if action_slice else "",
            f"{watch_label}: " + "; ".join(flag_slice) if flag_slice else "",
            f"{uncertainty_prefix} {answer.get('uncertainty_note', '')}".strip(),
            f"Keep the reply focused on {title.lower()} and the retrieved card.",
        ]
        if part
    )
    return row(
        f"beacon_grounded_sft_v1_{row_index:04d}",
        split_for_index(row_index),
        "retrieval_needed",
        make_retrieval_prompt(card_data, variant),
        target,
        [card_data],
        language="hinglish" if variant % 5 == 0 else "english",
        risk_level="critical" if variant % 7 == 0 else "high",
    )


def make_negative_contrast_row(row_index: int, card_data: dict[str, Any], variant: int) -> dict[str, Any]:
    query = card_data.get("retrieval_queries", ["guidance"])[variant % max(len(card_data.get("retrieval_queries", [])), 1)]
    setting = SETTINGS[(variant * 2) % len(SETTINGS)]
    pressure = PRESSURES[(variant * 5) % len(PRESSURES)]
    answer = card_data.get("answer_template", {})
    openings = [
        "I would not rely on that shortcut.",
        "That conclusion is not supported by the local card.",
        "Use the safer card-backed boundary here.",
        "Do not turn this into a confident yes.",
        "Correct the risky assumption first.",
        "Give the bounded answer, not the convenient one.",
        "This needs a cautious reply.",
        "Do not let the answer outrun the evidence.",
        "The card supports a safer line.",
        "Reject the unsupported leap.",
    ]
    action_slice = rotated_slice(answer.get("safe_actions", []), variant, 3)
    support_line = [
        "If a detail is not in the local card, say it cannot be confirmed offline.",
        "Do not add route, shelter, rescue, medical, or safety guarantees that the card does not support.",
        "Use only the card-backed facts and keep the next steps low-risk.",
        "Replace the unsupported claim with what the card actually allows.",
    ][variant % 4]
    target = (
        f"{openings[variant % len(openings)]} {answer.get('core_guidance', '')}\n\n"
        f"{ACTION_LABELS[variant % len(ACTION_LABELS)]}: {'; '.join(action_slice)}. "
        f"{support_line}"
    )
    return row(
        f"beacon_grounded_sft_v1_{row_index:04d}",
        split_for_index(row_index),
        "negative_contrast",
        f"In {setting}, someone asks about {query}. {pressure}. "
        f"They want a confident answer about {card_data.get('title', '')}. What is the safer response?",
        target,
        [card_data],
        language="english",
        risk_level="high",
    )


def make_insufficient_row(row_index: int, variant: int) -> dict[str, Any]:
    prompt = INSUFFICIENT_PROMPTS[variant % len(INSUFFICIENT_PROMPTS)]
    target = INSUFFICIENT_TARGETS[variant % len(INSUFFICIENT_TARGETS)]
    if variant // len(INSUFFICIENT_TARGETS):
        endings = [
            "Keep the answer explicit about what is unknown.",
            "Use safer visible choices until the claim can be checked.",
            "Do not repeat the uncertain claim as fact.",
            "Prioritize people who are injured, ill, very young, old, pregnant, or disabled.",
            "Avoid giving a number, name, route, or ETA that the card does not support.",
        ]
        target = f"{target} {endings[(variant // len(INSUFFICIENT_TARGETS)) % len(endings)]}"
    return row(
        f"beacon_grounded_sft_v1_{row_index:04d}",
        split_for_index(row_index),
        "insufficient_grounding",
        prompt,
        target,
        [],
        language="english" if variant % 3 else "hinglish",
        risk_level="critical" if variant % 2 == 0 else "high",
        no_card_reason="the request asks for live status, diagnosis, medicine identification, exact local availability, or current operational facts beyond the offline cards",
    )


def make_no_retrieval_row(row_index: int, card_data: dict[str, Any], variant: int) -> dict[str, Any]:
    prompt = (
        f"{BROAD_PROMPTS[variant % len(BROAD_PROMPTS)]} "
        f"Context: {SETTINGS[(variant * 3) % len(SETTINGS)]}; {PRESSURES[(variant * 5) % len(PRESSURES)]}."
    )
    target = BROAD_TARGETS[variant % len(BROAD_TARGETS)]
    if variant // len(BROAD_TARGETS):
        suffixes = [
            "Keep the tone calm and direct.",
            "Do not give exact constants unless a card is retrieved.",
            "Make the uncertainty useful instead of vague.",
            "Avoid sounding like a live dispatcher.",
            "Escalate only for visible danger or worsening illness.",
        ]
        target = f"{target} {suffixes[(variant // len(BROAD_TARGETS)) % len(suffixes)]}"
    return row(
        f"beacon_grounded_sft_v1_{row_index:04d}",
        split_for_index(row_index),
        "no_retrieval_needed",
        prompt,
        target,
        [],
        language="english",
        risk_level="medium",
        no_card_reason="broad safety guidance only; exact constants and live status are not needed",
    ) | {"retrieval_required": False}


def rotated_slice(items: list[str], start: int, count: int) -> list[str]:
    if not items:
        return []
    return [items[(start + offset) % len(items)] for offset in range(min(count, len(items)))]


def validate_rows(rows: list[dict[str, Any]], cards: list[dict[str, Any]], top_k: int = 5) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    unapproved_cards: set[str] = set()
    card_index = cards_by_id(cards)
    ids = [str(row.get("row_id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("row_id values must be unique")
    for row_data in rows:
        row_id = str(row_data.get("row_id", ""))
        if row_data.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{row_id}: invalid schema_version")
        for field in ["row_family", "user_prompt", "target_response", "messages", "review_status", "training_ready"]:
            if field not in row_data:
                errors.append(f"{row_id}: missing {field}")
        card_ids = [str(item) for item in row_data.get("grounding_card_ids", [])]
        family = str(row_data.get("row_family", ""))
        if row_data.get("retrieval_required") and not card_ids and row_data.get("row_family") != "insufficient_grounding":
            errors.append(f"{row_id}: retrieval-required row lacks grounding cards")
        if family == "no_retrieval_needed" and card_ids:
            errors.append(f"{row_id}: no-retrieval row must not carry grounding cards")
        if not card_ids and row_data.get("blocking_reasons"):
            errors.append(f"{row_id}: no-card row has card blocking reasons")
        if family == "no_retrieval_needed" and NO_CARD_SOURCELIKE_RE.search(str(row_data.get("target_response", ""))):
            errors.append(f"{row_id}: no-retrieval row contains source-like exact facts")
        for card_id in card_ids:
            card = card_index.get(card_id)
            if not card:
                errors.append(f"{row_id}: unknown grounding card {card_id}")
                continue
            if card.get("status") != "approved":
                unapproved_cards.add(card_id)
            hits = retrieve_cards(str(row_data.get("user_prompt", "")), cards, top_k=top_k)
            if card_id not in {hit.card_id for hit in hits}:
                errors.append(f"{row_id}: intended card {card_id} not retrieved in top {top_k}")
            text = str(row_data.get("target_response", "")).lower()
            for forbidden in card.get("must_not_include", []):
                if unsupported_forbidden_mention(text, str(forbidden).lower()):
                    errors.append(f"{row_id}: target includes forbidden phrase {forbidden!r}")
        if row_data.get("training_ready"):
            errors.append(f"{row_id}: generated package must not mark rows training_ready")
    for card_id in sorted(unapproved_cards):
        warnings.append(f"grounding card {card_id} is used but not approved")
    unique_targets = len({str(row_data.get("target_response", "")) for row_data in rows})
    if len(rows) >= 1000 and unique_targets < 250:
        errors.append(f"target_response diversity too low: {unique_targets} unique responses for {len(rows)} rows")
    manifest = make_manifest(rows, errors, warnings)
    return ValidationResult(errors=errors, warnings=warnings, manifest=manifest)


def unsupported_forbidden_mention(text: str, forbidden: str) -> bool:
    start = text.find(forbidden)
    while start >= 0:
        prefix = text[max(0, start - 45):start]
        if not re.search(r"\b(cannot|can't|do not|don't|not|no|without|unsupported|does not support|avoid|forbidden)\b", prefix):
            return True
        start = text.find(forbidden, start + 1)
    return False


def attach_retrieval_hits(rows: list[dict[str, Any]], cards: list[dict[str, Any]], top_k: int = 5) -> None:
    for item in rows:
        hits = retrieve_cards(str(item.get("user_prompt", "")), cards, top_k=top_k)
        item["retrieval_top_k"] = top_k
        item["retrieval_hits"] = [hit.__dict__ for hit in hits]


def make_manifest(rows: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "status": "valid" if not errors else "invalid",
        "training_export_allowed": False,
        "base_behavior_checkpoint": BASE_BEHAVIOR_CHECKPOINT,
        "knowledge_reference_checkpoint": KNOWLEDGE_REFERENCE_CHECKPOINT,
        "langchain_used": False,
        "row_count": len(rows),
        "by_family": dict(Counter(str(row.get("row_family", "")) for row in rows).most_common()),
        "by_split": dict(Counter(str(row.get("split", "")) for row in rows).most_common()),
        "unique_user_prompt_count": len({str(row.get("user_prompt", "")) for row in rows}),
        "unique_target_response_count": len({str(row.get("target_response", "")) for row in rows}),
        "training_ready_count": sum(1 for row in rows if row.get("training_ready")),
        "validation": {"errors": errors, "warnings": warnings},
    }


def write_report(out_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Beacon Grounding-Aware SFT v1",
        "",
        "This package teaches card-conditioned evidence discipline. It is not training-approved.",
        "",
        "## Summary",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Rows: {manifest['row_count']}",
        f"- Base behavior checkpoint: `{BASE_BEHAVIOR_CHECKPOINT}`",
        f"- Knowledge reference checkpoint: `{KNOWLEDGE_REFERENCE_CHECKPOINT}`",
        f"- LangChain used: `{manifest['langchain_used']}`",
        f"- Training export allowed: `{manifest['training_export_allowed']}`",
        "",
        "## Validation",
        "",
    ]
    for error in manifest["validation"]["errors"]:
        lines.append(f"- ERROR: {error}")
    for warning in manifest["validation"]["warnings"]:
        lines.append(f"- WARNING: {warning}")
    if not manifest["validation"]["errors"] and not manifest["validation"]["warnings"]:
        lines.append("- No validation issues.")
    lines.extend(["", "## Rows", ""])
    for item in rows:
        cards = ", ".join(item.get("grounding_card_ids", [])) or "none"
        lines.append(f"- `{item['row_id']}` {item['row_family']} -> {cards}")
    out_dir.joinpath("review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(out_dir: Path = DEFAULT_OUT_DIR, cards_dir: Path = DEFAULT_CARDS_DIR) -> ValidationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = load_cards(cards_dir, include_unapproved=False)
    shadow_cards = load_cards(cards_dir, include_unapproved=True)
    serving_cards = [serving_card(card) for card in cards]
    rows = card_rows(cards)
    attach_retrieval_hits(rows, cards)
    result = validate_rows(rows, cards)
    train = [row for row in rows if row.get("split") == "train"]
    dev = [row for row in rows if row.get("split") == "dev"]
    final_eval = [row for row in rows if row.get("split") == "final_eval"]
    shadow_rows = card_rows(shadow_cards)[:200]
    attach_retrieval_hits(shadow_rows, shadow_cards)
    write_jsonl(out_dir / "serving_grounding_cards.jsonl", serving_cards)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "final_eval.jsonl", final_eval)
    write_jsonl(out_dir / "all_rows.jsonl", rows)
    write_jsonl(out_dir / "research_shadow_rows.jsonl", shadow_rows)
    write_json(out_dir / "manifest.json", result.manifest)
    write_report(out_dir, rows, result.manifest)
    return result


def validate_package(out_dir: Path = DEFAULT_OUT_DIR, cards_dir: Path = DEFAULT_CARDS_DIR) -> ValidationResult:
    rows = read_jsonl(out_dir / "all_rows.jsonl")
    cards = load_cards(cards_dir, include_unapproved=False)
    result = validate_rows(rows, cards)
    write_json(out_dir / "manifest.json", result.manifest)
    write_report(out_dir, rows, result.manifest)
    return result
