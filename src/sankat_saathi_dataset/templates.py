from __future__ import annotations

from .schemas import DpoPair, SftExample, StructuredAnswer
from .sources import GUIDANCE_FACTS


FACT_BY_ID = {fact.fact_id: fact for fact in GUIDANCE_FACTS}

CRISIS_CONTEXTS = [
    {
        "lane": "flood_isolation",
        "prompt": "We are 5 people stuck on the first floor after flooding. No network, 8L water, wet rice bag, one elder has diabetes. What should we do first?",
        "facts": ["water_treat_uncertain_source", "floodwater_food_contact", "medicine_label_uncertainty", "risk_communication_style"],
        "risk": "high",
        "tags": ["flood", "water", "food", "diabetes", "elder"],
    },
    {
        "lane": "power_outage",
        "prompt": "Power has been gone since last night. Milk, eggs, and cooked dal were in the fridge. It smells okay. Can we eat it?",
        "facts": ["power_outage_perishables", "risk_communication_style"],
        "risk": "caution",
        "tags": ["power_outage", "food", "refrigeration"],
    },
    {
        "lane": "floodwater",
        "prompt": "The only water available is from a handpump surrounded by floodwater. It is cloudy but we can boil on a gas stove.",
        "facts": ["water_treat_uncertain_source", "cloudy_water_pre_treatment"],
        "risk": "high",
        "tags": ["water", "flood", "boil"],
    },
    {
        "lane": "first_aid",
        "prompt": "My father cut his foot while walking through floodwater. Bleeding stopped but the wound has mud and he has diabetes.",
        "facts": ["floodwater_contact_wounds", "medicine_label_uncertainty"],
        "risk": "high",
        "tags": ["wound", "floodwater", "diabetes"],
    },
    {
        "lane": "electricity",
        "prompt": "Water has entered the ground floor and an extension board is floating near the door. Can I move it with a wooden stick?",
        "facts": ["electrical_flood_hazard", "risk_communication_style"],
        "risk": "critical",
        "tags": ["electricity", "floodwater", "critical"],
    },
    {
        "lane": "cyclone_shelter",
        "prompt": "We are in a school shelter after a cyclone. A child has vomiting and loose motions after drinking stored water. We have ORS sachets.",
        "facts": ["water_treat_uncertain_source", "risk_communication_style"],
        "risk": "high",
        "tags": ["shelter", "water", "child", "diarrhea"],
    },
]

IMAGE_CASES = [
    {
        "image_id": "cloudy_water_bottle",
        "image_path": "data/images/generated/cloudy_water_bottle.svg",
        "question": "This bottle was filled from a pump near floodwater. Can we drink it?",
        "observations": ["plastic bottle", "cloudy water", "no visible label", "source cannot be verified from image"],
        "facts": ["water_treat_uncertain_source", "cloudy_water_pre_treatment"],
        "risk": "caution",
        "tags": ["vision", "water", "cannot_determine"],
    },
    {
        "image_id": "dented_can",
        "image_path": "data/images/generated/dented_can.svg",
        "question": "This food can was in floodwater and is dented. Is it okay if we wash it?",
        "observations": ["metal can", "visible dent", "possible floodwater exposure"],
        "facts": ["floodwater_food_contact"],
        "risk": "high",
        "tags": ["vision", "food", "packaging"],
    },
    {
        "image_id": "medicine_strip",
        "image_path": "data/images/generated/medicine_strip.svg",
        "question": "This is the medicine strip for my diabetic grandfather. The label is partly wet. Should he take it?",
        "observations": ["medicine strip", "partial label", "water damage possible", "dose cannot be confirmed from image"],
        "facts": ["medicine_label_uncertainty"],
        "risk": "cannot_determine",
        "tags": ["vision", "medicine", "diabetes"],
    },
    {
        "image_id": "wet_rice_bag",
        "image_path": "data/images/generated/wet_rice_bag.svg",
        "question": "Our rice bag got wet during floodwater. Can we dry it and cook it?",
        "observations": ["rice bag", "wet packaging", "floodwater contact cannot be ruled out"],
        "facts": ["floodwater_food_contact"],
        "risk": "high",
        "tags": ["vision", "food", "floodwater"],
    },
    {
        "image_id": "power_cable_water",
        "image_path": "data/images/generated/power_cable_water.svg",
        "question": "There is a cable in water at the entrance. Can we cross quickly?",
        "observations": ["cable", "standing water", "electrical status cannot be verified"],
        "facts": ["electrical_flood_hazard"],
        "risk": "critical",
        "tags": ["vision", "electricity", "critical"],
    },
]


def build_answer(prompt: str, fact_ids: list[str], risk: str, tags: list[str], image_observations: list[str] | None = None) -> StructuredAnswer:
    facts = [FACT_BY_ID[fact_id] for fact_id in fact_ids]
    allowed = [advice for fact in facts for advice in fact.allowed_advice][:5]
    escalation = sorted({trigger for fact in facts for trigger in fact.escalation_triggers})[:7]
    unsafe_items = []
    if "food" in tags:
        unsafe_items.append("Food or packaging that touched floodwater, especially porous packs, screw caps, cartons, or damaged cans.")
    if "water" in tags:
        unsafe_items.append("Untreated water from uncertain, cloudy, flood-affected, or chemically suspicious sources.")
    if "electricity" in tags:
        unsafe_items.append("Standing water near wires, outlets, batteries, or extension boards.")
    if "medicine" in tags:
        unsafe_items.append("Medicine with unreadable label, unknown dose, water damage, or changed appearance.")
    if "wound" in tags:
        unsafe_items.append("Floodwater-exposed wounds with mud, puncture, swelling, redness, fever, or diabetes risk.")
    if not unsafe_items:
        unsafe_items.append("Items whose safety cannot be confirmed from the available context.")

    missing = ["current location safety", "access to clean water/fuel", "age and health risks in the group"]
    if image_observations:
        missing.insert(0, "what happened before the photo and whether floodwater or chemicals touched the item")
    if "medicine" in tags:
        missing.append("medicine name, prescribed dose, meal status, and symptoms")
    if "food" in tags:
        missing.append("time without refrigeration or exact floodwater contact")

    immediate = [
        "Move people away from immediate hazards before trying to save supplies.",
        *allowed[:3],
    ]
    if risk == "critical":
        immediate.insert(0, "Treat this as life-threatening until proven otherwise from a safe place.")
    if "diabetes" in tags:
        immediate.append("For the diabetic/elder person, prioritize safe water, regular food if available, and watch for confusion, fainting, sweating, or unusual behavior.")

    return StructuredAnswer(
        risk_level=risk,  # type: ignore[arg-type]
        immediate_action=immediate[:5],
        resource_plan=[
            "Separate lower-risk sealed supplies from wet, damaged, or uncertain items.",
            "Use treated water for drinking, cooking, hand hygiene, and medicine needs.",
            "Ration by vulnerability: children, elders, pregnancy, diabetes, and sick people first.",
        ],
        unsafe_items=unsafe_items,
        missing_information=missing[:5],
        escalation_signs=escalation or ["rising water", "severe symptoms", "official evacuation order"],
        what_not_to_do=build_what_not_to_do(tags),
        hindi_hinglish=[
            "Pakka safe mat maano; jo cheez floodwater ya unknown source se touch hui hai usme caution rakho.",
            "Sabse pehle logon ko safe jagah rakho, phir paani/khana/medicine ko alag check karo.",
            "Severe symptoms, electric shock risk, rising water, ya diabetic elder ki problem ho to help milte hi escalate karo.",
        ],
        uncertainty_note="I cannot prove safety from limited context or an image alone; this guidance chooses the lower-risk practical action.",
    )


def build_what_not_to_do(tags: list[str]) -> list[str]:
    cautions = ["Do not present uncertain safety as proven from smell, appearance, or a photo."]
    if "water" in tags:
        cautions.append("Do not imply household treatment solves chemical, fuel, or industrial contamination.")
    if "food" in tags:
        cautions.append("Do not rely on cooking, reheating, smell, or taste to approve flood-exposed or poorly refrigerated food.")
    if "medicine" in tags or "diabetes" in tags:
        cautions.append("Do not make medical calls, identify pills, or change prescribed dosing from partial context.")
    if "electricity" in tags:
        cautions.append("Do not touch or move wires, outlets, batteries, or cables near standing water.")
    if "wound" in tags:
        cautions.append("Do not ignore floodwater-exposed wounds, punctures, infection signs, fever, or diabetes risk.")
    cautions.append("Do not replace official emergency or medical advice when help becomes reachable.")
    return cautions[:4]


def make_text_examples(target_count: int, split: str = "train") -> list[SftExample]:
    examples: list[SftExample] = []
    variants = [
        "",
        " Answer in simple English and Hinglish.",
        " We have no internet and only a basic phone battery left.",
        " Please be practical and tell us what not to do.",
    ]
    for idx in range(target_count):
        base = CRISIS_CONTEXTS[idx % len(CRISIS_CONTEXTS)]
        variant = variants[(idx // len(CRISIS_CONTEXTS)) % len(variants)]
        fact_ids = base["facts"]
        examples.append(
            SftExample(
                example_id=f"sft_text_{idx:04d}",
                split=split,  # type: ignore[arg-type]
                modality="text",
                source_ids=sorted({source_id for fact_id in fact_ids for source_id in FACT_BY_ID[fact_id].source_ids}),
                guidance_fact_ids=fact_ids,
                user_prompt=base["prompt"] + variant,
                assistant_response=build_answer(base["prompt"], fact_ids, base["risk"], base["tags"]),
                risk_tags=base["tags"],
                language_mix="bilingual",
            )
        )
    return examples


def make_vision_examples(target_count: int, split: str = "train") -> list[SftExample]:
    examples: list[SftExample] = []
    for idx in range(target_count):
        base = IMAGE_CASES[idx % len(IMAGE_CASES)]
        fact_ids = base["facts"]
        prompt = f"Image uploaded. {base['question']} Context: no network, disrupted conditions, need immediate practical guidance."
        examples.append(
            SftExample(
                example_id=f"sft_vision_{idx:04d}",
                split=split,  # type: ignore[arg-type]
                modality="vision",
                source_ids=sorted({source_id for fact_id in fact_ids for source_id in FACT_BY_ID[fact_id].source_ids}),
                guidance_fact_ids=fact_ids,
                user_prompt=prompt,
                assistant_response=build_answer(prompt, fact_ids, base["risk"], base["tags"], base["observations"]),
                risk_tags=base["tags"],
                language_mix="bilingual",
                image_path=base["image_path"],
                image_observations=base["observations"],
                image_uncertainty="The image supports visible observations only; contamination, dose, electrical status, and safety often cannot be confirmed from pixels alone.",
            )
        )
    return examples


def make_dpo_pairs(target_count: int) -> list[DpoPair]:
    text_examples = make_text_examples(max(target_count, 1))
    rejected_templates = [
        "This is definitely safe. Just boil or cook everything well and continue as normal. No need to worry unless someone feels very sick.",
        "It smells okay, so it should be safe. You can eat it after reheating and drink the water if it looks mostly clear.",
        "I can diagnose from this context: the person is stable. Change the medicine dose today and wait for rescue tomorrow.",
        "Move the wire with a wooden stick and cross quickly. The chance of shock is low if you are careful.",
    ]
    pairs: list[DpoPair] = []
    for idx in range(target_count):
        example = text_examples[idx % len(text_examples)]
        rejected = rejected_templates[idx % len(rejected_templates)]
        pairs.append(
            DpoPair(
                pair_id=f"dpo_{idx:04d}",
                source_ids=example.source_ids,
                guidance_fact_ids=example.guidance_fact_ids,
                prompt=example.user_prompt,
                chosen=example.assistant_response,
                rejected=rejected,
                rejection_reasons=["overconfident", "unsafe certainty", "misses escalation", "not grounded in source rules"],
                risk_tags=example.risk_tags,
            )
        )
    return pairs
