from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are Beacon, an offline crisis companion for India-relevant disaster situations. "
    "Give conservative, practical guidance. State uncertainty clearly, do not invent live "
    "facts, and give safer next steps before escalation."
)

DEFAULT_RULE_MANIFEST = Path("data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl")
DEFAULT_REAL_EVAL = Path("data/eval/beacon_real_problem_eval_v1/beacon_real_problem_eval_v1.jsonl")
DEFAULT_SEED_CARDS = Path("data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl")

REQUIRED_ROW_FIELDS = {
    "row_id",
    "split",
    "hazard",
    "risk_level",
    "language",
    "user_prompt",
    "target_response",
    "messages",
    "source_rule_ids",
    "must_include",
    "must_avoid",
    "review_status",
    "training_ready",
}

SPLITS = {"train", "dev", "final_eval"}

LIVE_FACT_PATTERNS = [
    re.compile(r"\b(call|dial)\s+\d{3,5}\b", re.I),
    re.compile(r"\b\d{3,5}[-\s]?\d{3,}\b"),
    re.compile(r"\b(road|bridge|route)\s+(is\s+)?(open|safe|clear)\b", re.I),
    re.compile(r"\b(shelter|camp|relief center)\s+(is|at|open|available)\b", re.I),
    re.compile(r"\b(rescue|ambulance|ndrf|police)\s+(will|is coming|has been sent)\b", re.I),
]

UNSAFE_CERTAINTY_PATTERNS = [
    re.compile(r"\b(definitely|certainly|guaranteed)\s+(safe|clean|open|fine)\b", re.I),
    re.compile(r"\bboiling\s+(will|can)\s+remove\s+(fuel|petrol|diesel|chemical)", re.I),
    re.compile(r"\b(smell|taste|looks?)\s+(normal|fine|ok).{0,30}\b(safe|eat|drink)\b", re.I),
]

ARTIFACT_PATTERNS = [
    re.compile(r"\b(as an ai|i am an ai)\b", re.I),
    re.compile(r"\bthis prompt\b|\btraining data\b|\brubric\b|\bsource_rule", re.I),
    re.compile(r"\brisk_level\s*:", re.I),
    re.compile(r"\bimmediate_action\s*:", re.I),
    re.compile(r"\bsafe boundary\s*:", re.I),
    re.compile(r"\bpractical steps\s*:", re.I),
    re.compile(r"\bcontext\s*:", re.I),
    re.compile(r"\brisky belief\s*:", re.I),
    re.compile(r"\blead with\b|\bhighest-signal\b|\bordered actions\b|\blower-risk checks\b", re.I),
    re.compile(r"\bfinal-eval competing-pressure variant\b|\bno-photo/no-live-status certainty both matter\b", re.I),
    re.compile(r"\bJo clue dikh raha hai\b|\bwhen dividing safe water, food, movement help, or attention\b", re.I),
    re.compile(r"\bwarning clue ho sakta hai, par final safety proof nahi\b", re.I),
    re.compile(r"\bke liye safer paani/food/movement pehle rakho\b", re.I),
    re.compile(r"\bnear the front of the plan\b|\bthat should not decide the safety call\b|\bWhen connectivity returns, cross-check live status\b", re.I),
    re.compile(r"\bEven with pressure to\b|\bdikh raha hai, isliye reassurance ke bajay cautious step lo\b", re.I),
    re.compile(r"\bGive .{0,80} practical help before convenience tasks\b|\bKeep the next step something you can verify where you are\b", re.I),
    re.compile(r"\bpeople may rush; handle the immediate hazard before logistics\b|\breassurance dene se pehle safer option choose karo\b|\bwait karane ke bajay pehle support do\b", re.I),
    re.compile(r"\bshould get the first clear help, before convenience tasks\b|\bHandle the immediate hazard before logistics; the pressure is\b", re.I),
    re.compile(r"\bHelp .{0,80} before spending time on convenience tasks\b|\bStart with the hazard, then handle logistics; pressure here is\b", re.I),
    re.compile(r"\breassurance baad me\b|\bpractical support do\b", re.I),
    re.compile(r"\bdikhe to pehle risk kam karo; tasalli dene se pehle safety dekho\b", re.I),
    re.compile(r"\bpriority person:\b|\bpressure hai:\b|\bnote the setting:\b|\bthe local context is pressure:\b|\bthis matters because of the pressure:\b|\bdikh raha hai:\b|\bthe case detail is\b", re.I),
    re.compile(r"\bLocal detail yeh hai\b|\blogistics can follow after\b", re.I),
]

BROKEN_STYLE_PATTERNS = [
    re.compile(r"\bdo not (boiling|safe|this|reheating|take the usual dose)\b", re.I),
    re.compile(r"\bmat make\b|\bmat declare\b|\bmat claim\b", re.I),
    re.compile(r"\bsmell proves it safety\b", re.I),
    re.compile(r"\bproof that .{0,40} safety\b", re.I),
    re.compile(r"\bpermission to use a flame\b", re.I),
    re.compile(r"\bvulnerable person,\s*vulnerable person\b", re.I),
    re.compile(r"\bpregnancy with vomiting,\s*vomiting\b", re.I),
]

HINDI_MARKERS = {
    "mat",
    "paani",
    "bachcha",
    "bachche",
    "pehle",
    "agar",
    "jaldi",
    "madad",
    "saans",
    "khatra",
    "safe",
}

HAZARD_LABELS = {
    "wash_ors_water": "water/ORS safety",
    "food_flood_power": "food and power-outage safety",
    "electrical_wet_devices": "wet electrical hazard",
    "carbon_monoxide_fuel": "carbon monoxide risk",
    "diabetes_medication": "medicine/diabetes disruption",
    "wounds_first_aid": "wound care after flood/disaster exposure",
    "route_rescue_live_fact": "route, rescue, and live-status uncertainty",
    "shelter_hygiene": "shelter hygiene",
    "landslide_structural": "landslide or structural danger",
    "visual_uncertainty": "image uncertainty",
    "accessibility_elder_disabled_pregnancy_child_language": "vulnerable-person access and communication",
    "urban_fire_lpg_chemical": "fire, LPG, or chemical risk",
    "post_disaster_contamination_infection": "post-disaster contamination/infection",
    "crowd_shelter_overcrowding": "crowd or shelter pressure",
    "heatwave_cold_lightning_dust": "weather exposure hazard",
    "dam_flash_flood_riverbank_coastal": "flash flood or water-edge hazard",
    "misinformation_fake_alerts_helplines_rescue": "misinformation or fake-alert risk",
    "infrastructure_power_telecom_road_transit": "infrastructure disruption",
}


SCENARIO_CARDS: list[dict[str, Any]] = [
    {
        "hazard": "carbon_monoxide",
        "risk_level": "critical",
        "language": "hinglish",
        "user_prompt": "Bijli gayi hai aur generator shop ke andar doorway ke paas chal raha hai. Shutter aadha khula hai, par bachche ko headache aur nausea hai. Ab kya karein?",
        "target_response": (
            "Generator ko andar ya doorway par chalana safe mat mano, shutter khula ho tab bhi.\n\n"
            "Abhi sabko fresh air me le jao, generator band karo agar bina risk ke ho sake, aur usse bahar khuli jagah par openings se door rakho. "
            "Bachche ko headache, nausea, dizziness, confusion, behoshi, chest pain, ya saans ki dikkat ho to local medical help ko urgent bulao.\n\n"
            "Raat bhar fan ya aadha shutter par bharosa karke generator ke paas mat ruko."
        ),
        "source_rule_ids": ["fuel_carbon_monoxide", "co_symptom_escalation", "indoor_fuel_device"],
        "must_include": ["fresh air", "generator outside/openings away", "symptom escalation"],
        "must_avoid": ["window or fan makes it safe", "sleep near generator", "indoor generator approval"],
    },
    {
        "hazard": "electrical_flood",
        "risk_level": "high",
        "language": "english",
        "user_prompt": "Water entered the basement and the pump switch is downstairs. If I wear rubber slippers, can I go switch it off quickly?",
        "target_response": (
            "Do not enter a flooded basement to reach an electrical switch. Rubber slippers are not reliable protection in standing water.\n\n"
            "Keep people away from the basement, stop others from touching wet walls, wires, pumps, or metal railings, and switch power off only from a dry, safe main switch if it is reachable without entering water. "
            "If anyone was shocked, has burns, chest pain, fainting, confusion, or breathing trouble, get urgent help.\n\n"
            "Wait for trained electrical or rescue help for the flooded area."
        ),
        "source_rule_ids": ["electrical_flood_hazard", "wet_device_reenergizing", "downed_line_distance"],
        "must_include": ["stay out of flooded electrical area", "dry safe switch only", "shock red flags"],
        "must_avoid": ["rubber footwear reassurance", "touch wet equipment", "move wires"],
    },
    {
        "hazard": "food_water",
        "risk_level": "high",
        "language": "english",
        "user_prompt": "Floodwater touched our rice sack and some biscuit packets. They smell normal. Can we dry them in sun and cook for children?",
        "target_response": (
            "Do not treat floodwater-touched rice sacks or biscuit packets as safe because they smell normal. Porous packaging and grains can hold contamination, and cooking may not fix chemical or sewage exposure.\n\n"
            "Keep those items separate from clean food, wash hands after handling them, and use sealed dry food or known safe water first for children. "
            "If someone has vomiting, diarrhea, fever, blood in stool, confusion, or cannot drink, seek medical help as soon as reachable.\n\n"
            "Do not taste-test the food to decide."
        ),
        "source_rule_ids": ["floodwater_food_contact", "damaged_food_packaging", "power_outage_perishables"],
        "must_include": ["do not rely on smell", "separate/discard porous flood-contact food", "illness red flags"],
        "must_avoid": ["sun drying makes safe", "cooking fixes floodwater", "taste test"],
    },
    {
        "hazard": "wash_ors",
        "risk_level": "critical",
        "language": "hinglish",
        "user_prompt": "Relief camp me 4 saal ke bachche ko loose motion aur vomiting hai. ORS packet hai, par paani nala ke paas wale handpump ka hai. Kya mix kar dein?",
        "target_response": (
            "ORS ko unsafe ya doubtful paani me seedha mix mat karo. Bachche ke liye sabse safe available paani use karo.\n\n"
            "Agar sealed paani, properly treated paani, ya trusted supply mil sakti hai to pehle wahi use karo. Bachche ko chhote chhote sips dete raho, haath aur bartan saaf rakho, aur stool/vomit ko cooking area se door rakho.\n\n"
            "Bachcha bahut sust ho, baar-baar ulti kare, khoon aaye, aankh dhansi lage, peshab bahut kam ho, ya paani na pee paaye to jaldi medical help lo."
        ),
        "source_rule_ids": ["ors_safe_water", "water_treat_uncertain_source", "dehydration_escalation"],
        "must_include": ["safe water for ORS", "small sips", "dehydration red flags"],
        "must_avoid": ["ORS with any water", "stronger ORS", "ignore vomiting"],
    },
    {
        "hazard": "medicine_diabetes",
        "risk_level": "critical",
        "language": "english",
        "user_prompt": "My father with diabetes missed food in the cyclone shelter and is sweating and confused. His tablets got wet. Should I give an extra tablet?",
        "target_response": (
            "Do not give an extra tablet or use wet, uncertain tablets based on this chat. Confusion and sweating in a person with diabetes can become urgent.\n\n"
            "Keep him sitting or lying safely, do not leave him alone, and use only medicines that are clearly identified and known from his prescription. "
            "If he can swallow and you already know low sugar is possible, use the usual safe sugar source your family has been told to use while arranging help.\n\n"
            "Get medical help urgently if he is confused, fainting, having a seizure, breathing trouble, chest pain, or cannot swallow safely."
        ),
        "source_rule_ids": ["diabetes_disrupted_meals", "damaged_medicine_label", "insulin_storage_uncertainty"],
        "must_include": ["no dose change", "known prescription only", "urgent diabetes red flags"],
        "must_avoid": ["extra dose", "identify wet tablets", "skip escalation"],
    },
    {
        "hazard": "landslide_structural",
        "risk_level": "critical",
        "language": "english",
        "user_prompt": "Rain is continuing, stones are falling behind our house, and a crack opened in the wall. Documents are inside. Can one person run in quickly?",
        "target_response": (
            "Do not send anyone back inside for documents. A new wall crack with falling stones and continuing rain is a serious warning sign.\n\n"
            "Move people away from the cracked structure and the slope or debris path. Keep the group together, help elders or children move first, and choose a safer nearby open place or known sturdy shelter only if the route is safe.\n\n"
            "If you hear rumbling, see new cracks, leaning poles or trees, fresh debris, or blocked drainage, stay away and call local responders when reachable."
        ),
        "source_rule_ids": ["structural_landslide_danger", "damaged_building_reentry", "slope_warning_signs"],
        "must_include": ["no re-entry", "move away from slope/structure", "landslide warning signs"],
        "must_avoid": ["quick re-entry", "building will hold", "cross debris path"],
    },
    {
        "hazard": "live_route_authority",
        "risk_level": "high",
        "language": "hinglish",
        "user_prompt": "WhatsApp forward bol raha hai bridge safe hai aur rescue boat aa rahi hai. Network nahi hai. Kya ground floor par wait karein?",
        "target_response": (
            "Main offline hoon, isliye bridge safe hai ya rescue boat aa rahi hai ye verify nahi kar sakta. WhatsApp forward ko confirmed status mat mano.\n\n"
            "Agar paani badh raha hai, ground floor par wait karna risky ho sakta hai. Battery bachao, essential medicines/papers ko waterproof pouch me rakho, aur safe route ho to upar wali jagah ya safer higher spot par shift karo. "
            "Bachche, elders, pregnant person, disabled person, aur sick logon ko pehle move karne ka plan banao.\n\n"
            "Route, shelter, warning, ya rescue ETA ke bare me confident claim mat forward karo jab tak local official source se verify na ho."
        ),
        "source_rule_ids": ["live_fact_uncertainty", "unsafe_rescue_self_protection", "flood_crossing_turn_around"],
        "must_include": ["cannot verify live status offline", "do not wait on unverified rescue", "safer higher place if flood rising"],
        "must_avoid": ["invent ETA", "confirm bridge safe", "impersonate authority"],
    },
    {
        "hazard": "wounds_first_aid",
        "risk_level": "medium",
        "language": "english",
        "user_prompt": "A child scraped his leg in floodwater. Bleeding stopped. Can we just tie an old cloth and let him play?",
        "target_response": (
            "Do not let him keep playing in floodwater, and avoid tying a dirty old cloth on the scrape.\n\n"
            "Move him away from floodwater, wash your hands if possible, rinse the wound with clean water, and cover it with the cleanest dry cloth or dressing available. "
            "Keep checking for spreading redness, swelling, pus, fever, increasing pain, or the child becoming unusually sleepy.\n\n"
            "If the wound is deep, from sharp debris, very dirty, on the foot, or the child has diabetes or weak immunity, seek medical help when reachable."
        ),
        "source_rule_ids": ["floodwater_contact_wounds", "wound_infection_escalation", "diabetes_wound_risk"],
        "must_include": ["keep wound out of floodwater", "clean water and clean cover", "infection red flags"],
        "must_avoid": ["dirty cloth is fine", "play in floodwater", "antibiotics without clinician"],
    },
    {
        "hazard": "shelter_hygiene",
        "risk_level": "high",
        "language": "english",
        "user_prompt": "The relief camp latrine is overflowing near the cooking area and three children have diarrhea. Volunteers want to first distribute snacks. What comes first?",
        "target_response": (
            "Put hygiene and diarrhea control before snack distribution. Overflowing latrine waste near cooking can spread illness quickly.\n\n"
            "Move food preparation away from the waste area, keep children with diarrhea away from shared food and water containers as much as possible, set up handwashing or sanitizer at the food line, and use safe water for drinking and ORS. "
            "Assign one person to clean only if they have basic protection and can avoid direct contact with waste.\n\n"
            "Get medical help urgently for lethargy, repeated vomiting, blood in stool, very little urine, confusion, or many new cases."
        ),
        "source_rule_ids": ["shelter_hygiene_wash", "diarrhea_cluster_escalation", "safe_food_handling_shelter"],
        "must_include": ["separate waste and cooking", "hand hygiene", "diarrhea escalation"],
        "must_avoid": ["ignore diarrhea cluster", "shared cups", "blame families"],
    },
    {
        "hazard": "visual_uncertainty",
        "risk_level": "medium",
        "language": "hinglish",
        "user_prompt": "Photo me tablet strip thoda dikh raha hai, label aadha mita hai. Kya bata sakte ho ye kaunsi diabetes medicine hai?",
        "target_response": (
            "Photo ya aadhe label se medicine identify karna safe nahi hai. Main isse diabetes medicine ya dose confirm nahi kar sakta.\n\n"
            "Wet ya unclear strip ko alag rakho, jo medicine clear label aur prescription se match hoti ho sirf wahi use karo, aur prescription/photo of prescription ko dry pouch me rakho. "
            "Agar person confused, faint, seizure, chest pain, breathing trouble, ya sugar bahut low/high jaisi signs me ho to medical help jaldi lo.\n\n"
            "Unknown tablet ko guess karke mat do."
        ),
        "source_rule_ids": ["visual_cannot_determine", "no_photo_safety_certainty", "damaged_medicine_label"],
        "must_include": ["cannot identify medicine from photo", "clear prescription only", "medical red flags"],
        "must_avoid": ["pill identification", "dose advice", "photo certainty"],
    },
    {
        "hazard": "food_water",
        "risk_level": "medium",
        "language": "hinglish",
        "user_prompt": "Paani cloudy hai aur petrol jaisi smell aa rahi hai. Gas available hai, kya boil karke chai ya ORS bana sakte hain?",
        "target_response": (
            "Petrol ya chemical smell wale paani ko sirf boil karke safe mat mano. Boiling germs me madad kar sakti hai, par fuel ya chemical contamination ko reliable tareeke se nahi hatati.\n\n"
            "Is paani ko peene, chai, ORS, baby formula, ya medicine ke liye avoid karo. Sealed water, trusted tanker, treated stored water, ya local verified safer source ko pehle use karo. "
            "Container ko clean water source se alag rakho taki mix-up na ho.\n\n"
            "Agar kisi ko vomiting, dizziness, breathing problem, confusion, ya severe diarrhea ho to medical help lo."
        ),
        "source_rule_ids": ["water_treat_uncertain_source", "ors_safe_water", "dehydration_escalation"],
        "must_include": ["boiling does not fix fuel/chemical smell", "avoid for ORS/formula/medicine", "safer source"],
        "must_avoid": ["boil chemical water safe", "use for formula", "smell reassurance"],
    },
    {
        "hazard": "live_route_authority",
        "risk_level": "low",
        "language": "english",
        "user_prompt": "Network is weak. I need to send one SMS to relatives asking for help without exaggerating. What should it say?",
        "target_response": (
            "Keep the SMS factual and short. Do not add unverified rescue, shelter, or road claims.\n\n"
            "You can send: \"We are at [exact location/landmark]. People here: [number], including [children/elderly/sick if any]. Main hazard: [water rising / no power / injury]. Urgent needs: [safe water/medicine/evacuation/help]. Phone battery low. Please share with local responders if reachable.\"\n\n"
            "If there is immediate danger, move to the safest reachable place first; send the message when it does not delay safety."
        ),
        "source_rule_ids": ["live_fact_uncertainty", "minimize_sensitive_data", "communicate_uncertainty_plainly"],
        "must_include": ["factual SMS", "no unverified claims", "urgent needs/location"],
        "must_avoid": ["invent official warning", "fake rescue ETA", "overshare sensitive data"],
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def compact_text(value: object) -> str:
    text = str(value or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bfictional\b\s*", "", text, flags=re.I)
    text = re.sub(r"\bfinal-eval competing-pressure variant:\s*", "", text, flags=re.I)
    text = re.sub(r"\bvulnerable person and no-photo/no-live-status certainty both matter;\s*", "", text, flags=re.I)
    text = text.replace("India context:", "").replace("offline guidance only.", "offline guidance is needed.")
    return text.strip(" ;.")


def sentence(value: str) -> str:
    value = compact_text(value)
    if not value:
        return ""
    return value if value[-1] in ".!?" else value + "."


def choose(items: list[str], index: int, default: str = "") -> str:
    clean = [compact_text(item) for item in items if compact_text(item)]
    if not clean:
        return default
    return clean[index % len(clean)]


def clean_action(text: str) -> str:
    text = compact_text(text)
    text = re.sub(r"^I cannot verify ", "I cannot verify ", text)
    text = text.replace("do not claim", "do not assume")
    text = text.replace("do not declare", "do not treat")
    text = re.sub(r"uncertain water is definitely safe", "uncertain water is safe", text, flags=re.I)
    text = re.sub(r"clear water is safe", "clear-looking water is safe", text, flags=re.I)
    text = re.sub(r"if the route is clear", "if you can move without entering smoke, water, debris, traffic, or electrical hazards", text, flags=re.I)
    text = re.sub(r"if safe route exists", "if you can move without entering a hazard", text, flags=re.I)
    text = re.sub(r"\blive road\b", "live local road status", text, flags=re.I)
    return sentence(text)


def risk_from_seed(seed: dict[str, Any]) -> str:
    difficulty = str(seed.get("difficulty_tier", "")).lower()
    if difficulty == "adversarial":
        return "high"
    if difficulty == "hard":
        return "high"
    if any(term in " ".join(seed.get("escalation_signs", [])).lower() for term in ["breathing", "confusion", "collapse", "seizure", "blood", "faint", "shock"]):
        return "critical"
    if difficulty == "easy":
        return "medium"
    return "medium"


def language_for_seed(seed: dict[str, Any], variant_index: int) -> str:
    style = str(seed.get("answer_style", ""))
    if style == "low_literacy_hinglish" or variant_index % 5 == 1:
        return "hinglish"
    if variant_index % 7 == 3:
        return "bilingual"
    return "english"


def prompt_for_seed(seed: dict[str, Any], variant_index: int, language: str) -> str:
    context = compact_text(seed.get("context_constraints"))
    pressure = compact_text(seed.get("user_pressure"))
    assumption = compact_text(seed.get("unsafe_assumption"))
    cues = "; ".join(compact_text(item) for item in seed.get("visual_cues", [])[:2] if compact_text(item))
    vulnerable = compact_text(seed.get("vulnerable_factor"))
    templates_en = [
        "We are dealing with this: {context}. People are saying: {assumption}. {pressure}. What should we do now?",
        "In this situation, {context}. I can see {cues}. The worry is {pressure}. Is it okay to assume {assumption_lc}?",
        "A local volunteer needs a short practical answer. {context}. Someone is assuming: {assumption}. {pressure}.",
        "Family needs offline guidance: {context}. There is {cues}. One person says {assumption}. What is the safer next step?",
    ]
    templates_hi = [
        "{context}. Log bol rahe hain: {assumption}. {pressure}. Abhi safe tareeka kya hai?",
        "Situation yeh hai: {context}. Dikhta hai: {cues}. Pressure hai: {pressure}. Kya {assumption_lc} maan sakte hain?",
        "Volunteer ko short Hinglish answer chahiye. {context}. Log soch rahe hain: {assumption}. Pehle kya karein?",
    ]
    data = {
        "context": context,
        "pressure": sentence(pressure),
        "assumption": assumption,
        "assumption_lc": assumption[:1].lower() + assumption[1:] if assumption else "it is safe",
        "cues": cues or "limited information",
        "vulnerable": vulnerable,
    }
    templates = templates_hi if language == "hinglish" else templates_en
    return templates[variant_index % len(templates)].format(**data)


def intro_for_seed(seed: dict[str, Any], language: str, variant_index: int) -> str:
    assumption = compact_text(seed.get("unsafe_assumption"))
    must_not = choose(seed.get("must_not_say", []), variant_index)
    unsafe_boundary = clean_action(f"do not {must_not}") if must_not and not must_not.lower().startswith("do not") else clean_action(must_not)
    hazard = HAZARD_LABELS.get(str(seed.get("primary_hazard", "")), compact_text(seed.get("primary_hazard", "this risk")))
    rule_ids = set(seed.get("source_rule_ids", []))
    if "live_fact_uncertainty" in rule_ids:
        if language == "hinglish":
            return "Forward, screenshot, ya kisi ek message ko live route/rescue proof mat mano."
        return "Do not use a forward, screenshot, or single message as proof of live route or rescue status."
    if "visual_cannot_determine" in rule_ids:
        if language == "hinglish":
            return "Photo useful clue ho sakti hai, par hidden danger ya identity confirm nahi karti."
        return "A photo can be a useful clue, but it cannot confirm hidden danger or identity by itself."
    if language == "hinglish":
        openers = [
            f"Is assumption ko safe mat mano: {assumption}.",
            f"Pehle {hazard} ko seriously lo; {unsafe_boundary[:1].lower() + unsafe_boundary[1:]}",
            f"Short answer: abhi lower-risk option lo, kyunki {assumption[:1].lower() + assumption[1:]} confirm nahi hai.",
        ]
    else:
        openers = [
            f"Do not treat this as safe: {assumption}.",
            f"The safer answer is to act on the {hazard} risk, not on the shortcut.",
            f"Do not rely on the risky assumption here. {unsafe_boundary}",
            f"Take the cautious option first; this situation can become unsafe quickly.",
        ]
    return openers[variant_index % len(openers)]


def action_lines_for_seed(seed: dict[str, Any], language: str, variant_index: int) -> list[str]:
    must_say = list(seed.get("must_say", []))
    safe_guidance = list(seed.get("allowed_safe_guidance", []))
    actions = [clean_action(item) for item in [*must_say, *safe_guidance] if compact_text(item)]
    filtered: list[str] = []
    for action in actions:
        lower = action.lower()
        if "allow light hinglish" in lower or "use short simple" in lower or "answer " in lower:
            continue
        if action.lower() not in {item.lower() for item in filtered}:
            filtered.append(action)
    if not filtered:
        filtered = ["Choose the lower-risk option first.", "Keep people away from the immediate hazard.", "Use only information you can verify locally."]
    start = variant_index % len(filtered)
    selected = [filtered[(start + offset) % len(filtered)] for offset in range(min(3, len(filtered)))]
    if language == "hinglish":
        selected = [item.replace("Do not ", "Mat ").replace("do not ", "mat ") for item in selected]
    return selected


def red_flag_sentence(seed: dict[str, Any], language: str) -> str:
    raw_signs = [compact_text(item) for item in seed.get("escalation_signs", []) if compact_text(item)]
    signs: list[str] = []
    for item in raw_signs:
        lower = item.lower()
        if lower in {"vulnerable person", "vulnerable person or red flag present", "red flag present", "none required"}:
            continue
        if any(lower == old.lower() or lower in old.lower() for old in signs):
            continue
        signs.append(item)
    if not signs:
        signs = ["worsening symptoms", "confusion", "breathing trouble"]
    signs_text = ", ".join(signs[:4])
    if language == "hinglish":
        return f"Agar {signs_text} dikhe, trained local help ya medical/rescue help jaldi lo."
    return f"Escalate quickly if you see {signs_text}, or if vulnerable people cannot move, drink, breathe normally, or stay alert."


def boundary_sentence(seed: dict[str, Any], language: str, variant_index: int) -> str:
    forbidden = clean_action(choose(seed.get("forbidden_claims", []) + seed.get("must_not_say", []), variant_index, "live safety status"))
    rule_ids = set(seed.get("source_rule_ids", []))
    hazard = str(seed.get("primary_hazard", ""))
    boundary_topic = re.sub(r"^(do not|don't|mat)\s+", "", forbidden, flags=re.I).strip(" .") or "local safety"
    boundary_topic = re.sub(r"\bis safe\b", "safety", boundary_topic, flags=re.I)
    boundary_topic_lc = boundary_topic[:1].lower() + boundary_topic[1:] if boundary_topic else "local safety"
    first_word = boundary_topic_lc.split(maxsplit=1)[0] if boundary_topic_lc else ""
    action_words = {
        "make",
        "use",
        "open",
        "throw",
        "touch",
        "enter",
        "pull",
        "cross",
        "eat",
        "drink",
        "feed",
        "take",
        "identify",
        "blame",
        "give",
        "move",
        "sleep",
        "wait",
        "drive",
        "walk",
    }
    if language == "hinglish":
        if "live_fact_uncertainty" in rule_ids or "misinformation" in hazard or "route" in hazard:
            return "Offline answer se route, shelter, warning, ya rescue ETA confirm nahi ho sakta; local verified source milte hi cross-check karo."
        if "damaged_medicine_label" in rule_ids or "diabetes" in hazard or "medicine" in hazard:
            return "Is chat se medicine identity, dose, ya wet/unclear strip ki safety confirm nahi hoti."
        if "visual_cannot_determine" in rule_ids or "visual" in hazard:
            return "Photo se hidden danger, contamination, depth, current, ya medicine identity pakka confirm nahi hota."
        if first_word in action_words:
            return f"Is answer ko {boundary_topic_lc} ki permission mat samjho."
        return f"Offline context me yeh pakka mat mano ki {boundary_topic_lc}."
    if "live_fact_uncertainty" in rule_ids or "misinformation" in hazard or "route" in hazard:
        return "An offline answer cannot confirm route, shelter, warning, or rescue ETA; cross-check with a verified local source when one is reachable."
    if "damaged_medicine_label" in rule_ids or "diabetes" in hazard or "medicine" in hazard:
        return "This chat cannot confirm medicine identity, dose changes, or whether wet/unclear medicines are safe to use."
    if "visual_cannot_determine" in rule_ids or "visual" in hazard:
        return "A photo cannot reliably confirm hidden danger, contamination, water depth/current, or medicine identity."
    if first_word in action_words:
        return f"Do not treat this answer as permission to {boundary_topic_lc}."
    return f"Do not treat this answer as proof that {boundary_topic_lc}."


def sanitize_response_text(text: str) -> str:
    text = text.replace("सामान", "saamaan")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_assumption_text(seed: dict[str, Any]) -> str:
    assumption = compact_text(seed.get("unsafe_assumption"))
    assumption = re.sub(r"\bdefinitely\s+", "", assumption, flags=re.I)
    assumption = re.sub(r"\b(certainly|guaranteed)\s+", "", assumption, flags=re.I)
    assumption = re.sub(r"\b(road|bridge|route)\s+(is\s+)?(open|safe|clear)\b", r"\1 status is confirmed", assumption, flags=re.I)
    assumption = re.sub(r"\b(shelter|camp|relief center)\s+(is|at|open|available)\b", r"\1 status is confirmed", assumption, flags=re.I)
    return assumption


def personalize_response(text: str, seed: dict[str, Any], language: str, variant_index: int) -> str:
    assumption = safe_assumption_text(seed)
    if not assumption or assumption.lower() in text.lower():
        return text
    context = compact_text(seed.get("context_constraints"))
    pressure = compact_text(seed.get("user_pressure"))
    cue = choose(seed.get("visual_cues", []), variant_index)
    vulnerable = compact_text(seed.get("vulnerable_factor"))
    style_index = (variant_index + int(sha256_text(str(seed.get("seed_id", "")))[:2], 16)) % 6
    if language == "hinglish":
        detail = choose(
            [
                f"Yahan {context}." if context else "",
                f"Log jaldi karna chahte hain kyunki {pressure}." if pressure else "",
                f"Scene me {cue}." if cue else "",
                f"{vulnerable} ko priority do." if vulnerable else "",
            ],
            style_index,
        )
        prefixes = [
            f"Is baat par action mat lo: {assumption}. {detail}.",
            f"Yeh assumption confirm nahi hai: {assumption}. {detail}.",
            f"Shortcut yeh hai, aur ise avoid karo: {assumption}. {detail}.",
            f"Is claim ko safety proof mat banao: {assumption}. {detail}.",
            f"Pehle is risky soch ko side rakho: {assumption}. {detail}.",
            f"Is line ko pakka status mat samjho: {assumption}. {detail}.",
        ]
    else:
        detail = choose(
            [
                f"you are dealing with {context}." if context else "",
                f"people may rush because {pressure}." if pressure else "",
                f"the scene includes {cue}." if cue else "",
                f"{vulnerable} needs priority." if vulnerable else "",
            ],
            style_index,
        )
        prefixes = [
            f"Do not act on this assumption: {assumption}. Here, {detail}",
            f"Treat this claim as unverified: {assumption}. {detail[:1].upper() + detail[1:]}",
            f"The risky shortcut here is: {assumption}. {detail[:1].upper() + detail[1:]}",
            f"Do not use this as proof of safety: {assumption}. {detail[:1].upper() + detail[1:]}",
            f"Set this assumption aside before deciding: {assumption}. {detail[:1].upper() + detail[1:]}",
            f"Do not make the plan depend on this claim: {assumption}. {detail[:1].upper() + detail[1:]}",
        ]
    prefix = re.sub(r"\s+\.", ".", prefixes[style_index])
    return prefix + "\n\n" + text


def duplicate_response_note(seed: dict[str, Any], language: str, offset: int) -> str:
    pressure = compact_text(seed.get("user_pressure"))
    cue = choose(seed.get("visual_cues", []), offset)
    vulnerable = compact_text(seed.get("vulnerable_factor"))
    live_related = "live_fact_uncertainty" in seed.get("source_rule_ids", []) or "route" in str(seed.get("primary_hazard", ""))
    if language == "hinglish":
        live_line = "Route/rescue update ko local trusted check ke bina forward mat karo." if live_related else "Agla step wahi rakho jo yahin dekh kar safely kiya ja sake."
        options = [
            f"{pressure[:1].upper() + pressure[1:]} ho, tab bhi pehle logon ko hazard se door karo." if pressure else "Pehle logon ko hazard se door karo.",
            f"{cue}. Is detail ko dekh kar low-risk step lo." if cue else "Limited info me cautious step lo.",
            f"{vulnerable} ko pehle direct help do." if vulnerable else "Bachche, elders, sick, pregnant, aur disabled log pehle support chahte hain.",
            live_line,
            f"Logistics baad me; abhi unsafe assumption ko action ka base mat banao.",
            f"Jahan doubt ho, same jagah ruk kar risk badhane se better lower-risk move choose karo.",
            f"Clean water, dry electricity, aur clear exit ko alag-alag check karo.",
            f"Ek person watch rakhe aur ek person help/supplies organize kare.",
        ]
    else:
        live_line = "Share route or rescue updates only after a trusted local check." if live_related else f"For {HAZARD_LABELS.get(str(seed.get('primary_hazard', '')), 'this hazard')}, choose an action based on what you can observe locally."
        options = [
            f"Do the safety step first, even if {pressure[:1].lower() + pressure[1:]}." if pressure else "Do the safety step first.",
            f"Use {cue} as a reason to be cautious, not as reassurance." if cue else "Limited information should push the choice toward caution.",
            f"Check on {vulnerable} before turning to lower-priority tasks." if vulnerable else "Children, elders, sick people, pregnant people, and disabled people need help early.",
            live_line,
            "Put one person on watching the hazard and one on supplies or messages.",
            "Choose the step that keeps people farther from water, smoke, unstable ground, wires, or contamination.",
            "Keep clean supplies separated from anything wet, smoky, muddy, or unlabeled.",
            "If the situation worsens, stop the task and move people before protecting belongings.",
            f"Use the local clue about {cue} to avoid reassurance if conditions can worsen." if cue else "Avoid reassurance when conditions can worsen.",
            f"Do the safety step before solving the logistics problem: {pressure[:1].lower() + pressure[1:]}." if pressure else "Do the safety step before solving logistics.",
            f"Make the next action simple enough for {vulnerable or 'the most affected person'} to follow.",
            "Keep the answer grounded in stable safety steps, not a live-status guess.",
        ]
    return options[offset % len(options)]


def response_for_seed(seed: dict[str, Any], variant_index: int, language: str) -> str:
    hazard = str(seed.get("primary_hazard", ""))
    assumption = compact_text(seed.get("unsafe_assumption"))
    vulnerable = compact_text(seed.get("vulnerable_factor"))
    red_flags = red_flag_sentence(seed, language)
    fuel_or_chemical = any(term in (assumption + " " + compact_text(seed.get("context_constraints"))).lower() for term in ["fuel", "petrol", "diesel", "chemical", "oil", "smell"])
    opener_index = (variant_index + int(sha256_text(str(seed.get("seed_id", "")))[:2], 16)) % 3
    if language == "hinglish":
        vulnerable_part = f" {vulnerable} ko pehle priority do." if vulnerable else ""
        if "wash_ors_water" in hazard:
            extra = " Boiling se fuel ya chemical contamination reliably nahi nikalti." if fuel_or_chemical else ""
            opener = [
                f"Is baat ko safety proof mat mano: {assumption}.",
                f"Paani ke liye yeh shortcut risky hai: {assumption}.",
                f"Drinking/ORS ke decision me yeh assumption use mat karo: {assumption}.",
            ][opener_index]
            return f"{opener}{extra}\n\nSabse safe available paani peene, ORS, baby formula, aur medicine ke liye bachao.{vulnerable_part} Doubtful paani ko alag mark karo, clean covered container use karo, aur cup/bartan share karne se bacho.\n\n{red_flags}"
        if "food_flood_power" in hazard:
            opener = ["Smell ya packet dekh kar food safe maan lena risk hai.", "Flood ya power cut ke baad food ko normal smell se judge mat karo.", "Doubtful food ko bachane ke pressure me unsafe meal mat do."][opener_index]
            return f"{opener}\n\nWet cardboard, khula grain, spoiled perishable food, ya doubtful cooked food ko bachchon/elders ko mat do. Clean hands rakho, doubtful food alag rakho, aur sealed dry food ya freshly cooked safe food ko priority do.{vulnerable_part}\n\n{red_flags}"
        if "electrical" in hazard:
            opener = ["Paani, damp floor, ya wet device ke saath electrical kaam mat karo.", "Wet switch, pump, panel, charger, ya wire ko quick-fix mat samjho.", "Electrical hazard me rubber chappal ya wooden stick par bharosa mat karo."][opener_index]
            return f"{opener}\n\nLogon ko switch, wire, pump, panel, charger, aur metal railing se door rakho. Power sirf dry, reachable main switch se band karo; standing water me jaake switch touch mat karo. Reuse se pehle trained electrician se check karwana safer hai.\n\n{red_flags}"
        if "carbon_monoxide" in hazard:
            opener = ["Indoor generator, angeethi, coal, ya fuel device ko open window/fan ke bharose safe mat mano.", "Headache/nausea ke saath indoor fuel use ho raha hai to fresh air pehle.", "Fuel device ko room, doorway, balcony corner, ya shutter ke andar chalana risky hai."][opener_index]
            return f"{opener}\n\nSabko fresh air me le jao. Device band karna safe ho to band karo, aur use hamesha bahar, doors/windows se door rakho. Headache, nausea, dizziness, ya same symptoms multiple logon me ho to CO risk seriously lo.\n\n{red_flags}"
        if "diabetes" in hazard or "medicine" in hazard:
            opener = ["Wet ya unclear strip dekh kar medicine guess mat karo.", "Diabetes medicine me extra dose ya guessed tablet dangerous ho sakti hai.", "Unclear strip, heat-exposed medicine, ya missed meal ko normal routine jaisa mat treat karo."][opener_index]
            return f"{opener}\n\nDose change ya extra tablet is chat se mat karo. Sirf clear label/prescription se match hoti medicine use karo, person ko akela mat chhodo, aur meal disruption me family ka known diabetes plan follow karo if already taught.{vulnerable_part}\n\n{red_flags}"
        if "urban_fire_lpg_chemical" in hazard:
            opener = ["Smoke, LPG smell, oil fire, ya unknown chemical ko quick-fix mat karo.", "Fire/LPG/chemical scene me switches, sparks, water, aur DIY repair se risk badh sakta hai.", "Unknown gas, smoke, ya chemical exposure me pehle distance aur ventilation-safe movement socho."][opener_index]
            return f"{opener}\n\nLogon ko source se door le jao. LPG smell ho to switches/sparks/flame avoid karo. Oil/electrical fire par paani mat phenko. Unknown chemical ko smell, touch, mix, ya neutralize mat karo; contaminated kapde/skin ko clean water se rinse karna safe ho to karo.\n\n{red_flags}"
        if "post_disaster_contamination_infection" in hazard:
            context_blob = (assumption + " " + compact_text(seed.get("context_constraints")) + " " + " ".join(seed.get("must_say", []))).lower()
            opener = ["Sewage, dried mud, flood residue, ya contaminated kitchen ko normal dirt mat samjho.", "Cleanup me contamination ko food, water, wounds, aur children se door rakhna priority hai.", "Flood residue dry ya normal dikhne ke baad bhi contamination reh sakti hai."][opener_index]
            if "child" in context_blob or "play" in context_blob or "puddle" in context_blob:
                body = "Bachchon ko stagnant/floodwater se door karo, cuts cover karo, aur safer paani mile to hands/feet dhulao. Ration line ya waiting area me unke liye dry safer spot assign karo."
            elif "privacy" in context_blob or "runoff" in context_blob or "bathing" in context_blob:
                body = "Drinking-water containers ko bathing/laundry runoff aur drain se door rakho. Hygiene space me privacy rakho, soiled material covered disposal me rakho if available, aur vulnerable users se private details mat mango."
            else:
                body = "Cooking area, clean water, utensils, aur bedding ko dirty mud/sewage se alag rakho. Gloves/covered hands use karo if available, cleanup ke baad hands wash karo, contaminated grain/porous food alag rakho, aur sick people ko shared food-water handling se door rakho."
            return f"{opener}\n\n{body}\n\n{red_flags}"
        if "wounds" in hazard or "contamination" in hazard:
            chemical = " Chemical splash ho to contaminated kapda hatakar clean paani se rinse karo if available." if "chemical" in (assumption + hazard).lower() else ""
            return f"Floodwater, dirty cloth, ash, phenyl, chilli, ya chemical ko wound treatment mat banao.{chemical}\n\nPerson ko dirty water se door karo, hands clean karo, wound ko clean water se rinse karo, aur clean dry cloth/dressing se cover karo. Wound ko dobara floodwater me mat bhejo.\n\n{red_flags}"
        if "infrastructure_power_telecom_road_transit" in hazard:
            opener = ["Power, telecom, road, ya transit disruption me live status guess mat karo.", "Lift, signal, charger, road, aur transit shortcuts ko normal-service jaisa assume mat karo.", "Infrastructure failure me safest plan woh hai jo battery, dry electricity, aur reachable route bachata hai."][opener_index]
            return f"{opener}\n\nPhone battery bachao, ek short location/needs message ready rakho, lift use mat karo during power instability, wet charging points avoid karo, aur route sirf visible dry/clear ho tab consider karo. Road/transit/open-service claims ko verified local source se cross-check karo.\n\n{red_flags}"
        if "route" in hazard or "misinformation" in hazard or "infrastructure" in hazard:
            opener = ["Forward, screenshot, ya kisi ek message ko live route/rescue proof mat mano.", "Rumor ke bharose bridge, road, shelter, ya rescue status confirm mat samjho.", "Live route ya rescue ETA offline answer se pakka nahi hota."][opener_index]
            return f"{opener}\n\nPaani badh raha ho to lower floor par sirf rescue ka wait mat karo. Battery bachao, location/needs ka short message ready rakho, vulnerable logon ko pehle move karne ka plan banao, aur floodwater/fast current cross mat karo. Verified local source mile to status cross-check karo.\n\n{red_flags}"
        if "dam" in hazard or "flash_flood" in hazard:
            return f"Nadi, drain, dam release, ya dry channel ke paas paani ko predictable mat mano.\n\nFast current, achanak badhta paani, debris, ya loud flow dikhe to neeche/kinare se door jao. Bachchon ko water edge se door rakho, vehicle ya paidal crossing avoid karo, aur higher ground ki taraf move karo if route me paani/current/debris nahi hai.\n\n{red_flags}"
        if "crowd_shelter_overcrowding" in hazard:
            opener = ["Crowd pressure me doorway, aisle, stairs, aur exits block mat hone do.", "Distribution ya shelter entry ko dhakka-mukki me badalne mat do.", "Overcrowding me sabse pehle exit space aur vulnerable logon ki movement bachao."][opener_index]
            return f"{opener}\n\nLine ko slow aur one-way rakho, cooking/water/medical areas ke paas bheed kam karo, exits aur aisles clear rakho, aur child/elder/disabled/pregnant/sick logon ko edge ya quieter safer spot do. Rumor ya panic announcement mat phailao.\n\n{red_flags}"
        if "shelter" in hazard:
            return f"Shelter me waste, diarrhea, crowding, ya cooking-area hygiene ko side issue mat samjho.\n\nFood/cooking ko latrine ya dirty water se door karo, handwashing point banao, safe water/ORS ko priority do, sick children ko shared food-water area se jitna possible ho door rakho, aur vulnerable logon ke liye quieter safer corner arrange karo.\n\n{red_flags}"
        if "accessibility" in hazard:
            return f"Plan sirf fast-moving adults ke hisaab se mat banao.\n\nElder, pregnant person, disabled person, child, ya sick person ko pehle identify karo. Unke medicines, glasses, mobility aid, phone/contact note, aur water ko saath rakho. Ek helper assign karo, route ko dry/clear/short rakho, aur crowd me unhe alag padne mat do.\n\n{red_flags}"
        if "heatwave" in hazard or "lightning" in hazard or "cold" in hazard or "dust" in hazard:
            return f"Weather exposure ko sirf discomfort mat samjho; jaldi serious ho sakta hai.\n\nHeat me shade, rest, loose clothing, aur safe paani priority do. Lightning me open field, tree, metal pole, aur water edge se door jao. Cold/dust me body ko dry/covered rakho aur breathing trouble wale logon ko smoke/dust se door karo.\n\n{red_flags}"
        if "landslide" in hazard or "structural" in hazard:
            return f"Crack, falling stones, rumbling, leaning wall/tree, ya fresh debris ko ignore mat karo.\n\nDocuments ya सामान ke liye damaged building/slope ke paas wapas mat jao. Group ko slope/debris path se door, higher/open safer jagah le jao, aur children/elders/disabled person ko pehle help karo.\n\n{red_flags}"
        if "visual" in hazard:
            return f"Photo se hidden danger, water depth/current, contamination, ya medicine identity confirm nahi hoti.\n\nImage ko sirf clue samjho. Agar doubt ho to lower-risk choice lo: unknown tablet mat do, doubtful food/water use mat karo, aur unsafe-looking structure/wire/water ke paas mat jao. Clear local check ya trained help ka wait karo jab possible ho.\n\n{red_flags}"
        return f"Is situation me risky assumption par kaam mat karo.\n\nPehle life safety, clean water/food, dry electricity boundary, aur vulnerable logon ki movement dekho. Jo cheez live status, diagnosis, medicine identity, ya route safety maangti hai usse confirm claim mat banao.\n\n{red_flags}"

    vulnerable_part = f" Give priority to {vulnerable}." if vulnerable else ""
    if "wash_ors_water" in hazard:
        extra = " Boiling may help with germs, but it does not reliably remove fuel or chemical contamination." if fuel_or_chemical else ""
        opener = [
            f"Do not use this assumption as your water-safety check: {assumption}.",
            f"That is not a safe basis for drinking water or ORS: {assumption}.",
            f"Treat the water as doubtful until there is a safer source or proper treatment: {assumption}.",
        ][opener_index]
        return f"{opener}{extra}\n\nKeep the safest available water for drinking, ORS, baby formula, and medicines first.{vulnerable_part} Mark doubtful water separately, use covered clean containers, and avoid shared cups where possible. If ORS is needed, mix it with the safest treated water you can get and do not change the packet concentration.\n\n{red_flags}"
    if "food_flood_power" in hazard:
        opener = ["Do not rely on smell, taste, or a dry-looking packet to decide food is safe.", "After flood contact or a long power cut, doubtful food should not be rescued by smell or reheating.", "Keep children and vulnerable people away from food that may be flood-touched, spoiled, or held too long."][opener_index]
        return f"{opener}\n\nKeep doubtful food away from clean supplies. Avoid wet cardboard, porous grain sacks, spoiled perishables, and cooked food held warm too long; use sealed dry food or freshly cooked food from safe water first.{vulnerable_part} Do not taste-test food to decide.\n\n{red_flags}"
    if "electrical" in hazard:
        opener = ["Do not enter water or touch wet equipment to solve an electrical problem.", "Treat wet switches, pumps, chargers, panels, and wires as dangerous until checked.", "Rubber slippers, wooden sticks, or a quick reach are not reliable protection around wet electricity."][opener_index]
        return f"{opener}\n\nKeep people away from switches, pumps, panels, chargers, wires, and metal railings. Turn power off only from a dry, reachable main switch; otherwise wait for trained electrical or rescue help. Wet devices and flooded areas need inspection before reuse.\n\n{red_flags}"
    if "carbon_monoxide" in hazard:
        opener = ["Do not treat an indoor fuel device as safe because a window, door, or fan is open.", "If fuel is burning indoors and people feel unwell, fresh air comes first.", "A generator, charcoal, stove, or fuel device belongs outside and away from openings."][opener_index]
        return f"{opener}\n\nMove everyone to fresh air. Turn the generator, stove, charcoal, or fuel device off only if you can do it without entering danger, and keep it outside away from doors and windows. Similar headache, nausea, dizziness, or confusion in more than one person is a serious warning.\n\n{red_flags}"
    if "diabetes" in hazard or "medicine" in hazard:
        opener = ["Do not guess medicines, doses, or safety from wet strips, unclear labels, or a photo.", "Do not change diabetes tablets or insulin based on a damaged strip or this chat.", "A missed meal plus sweating, confusion, wet medicine, or heat-exposed medicine needs caution."][opener_index]
        return f"{opener}\n\nUse only medicine that clearly matches the person’s known prescription. Do not add an extra tablet or change insulin/tablet timing based on this chat. Keep the person observed, protect regular food and fluids as much as possible, and follow only a diabetes plan the family was already taught.\n\n{red_flags}"
    if "urban_fire_lpg_chemical" in hazard:
        opener = ["Do not try to fix smoke, LPG smell, oil fire, or unknown chemical exposure with a quick household trick.", "Fire, LPG, and chemical situations need distance first, not DIY repair.", "Avoid sparks, switches, flame tests, water-on-oil/electrical-fire, and unknown chemical mixing."][opener_index]
        return f"{opener}\n\nMove people away from the source. If LPG/gas is suspected, avoid switches, sparks, and flames. Do not throw water on oil or electrical fire. Do not smell, touch, mix, or neutralize unknown chemicals; rinse skin/clothes with clean water only if that can be done safely.\n\n{red_flags}"
    if "post_disaster_contamination_infection" in hazard:
        opener = ["Do not treat sewage, dried mud, flood residue, or a contaminated kitchen as ordinary dirt.", "After flooding, cleanup choices can spread contamination into food, water, bedding, and wounds.", "A normal smell or dry-looking mud can still leave contamination on surfaces, grain, utensils, or rooms."][opener_index]
        return f"{opener}\n\nSeparate clean food, water, utensils, bedding, and medicines from dirty cleanup areas. Use gloves or covered hands if available, wash hands after cleanup, keep children away from sewage/mud, keep contaminated grain or porous food aside, and reduce sick people handling shared food or water.\n\n{red_flags}"
    if "wounds" in hazard or "contamination" in hazard:
        chemical = " If there may be a chemical splash, remove contaminated cloth if it is easy to do and rinse with clean water if available." if "chemical" in (assumption + hazard).lower() else ""
        return f"Do not put dirty cloth, ash, chilli, phenyl, fuel, or other harsh substances on a wound.{chemical}\n\nMove the person away from floodwater or contamination, clean your hands if possible, rinse with clean water, and cover with the cleanest dry dressing or cloth available. Keep the wound out of floodwater after that.\n\n{red_flags}"
    if "infrastructure_power_telecom_road_transit" in hazard:
        opener = ["Do not assume power, telecom, roads, lifts, or transit are working normally during disruption.", "Infrastructure disruption needs a low-battery, dry-electricity, reachable-route plan.", "Avoid making the plan depend on unverified road, transit, signal, or power-restoration claims."][opener_index]
        return f"{opener}\n\nSave phone battery, prepare one short location-and-needs message, avoid lifts during unstable power, avoid wet charging points and damaged wires, and use roads/transit only when the path is visibly passable without floodwater, debris, or electrical hazards. Verify service status locally before forwarding it.\n\n{red_flags}"
    if "route" in hazard or "misinformation" in hazard or "infrastructure" in hazard:
        opener = ["Do not treat a forward, screenshot, rumor, or single message as confirmed live route or rescue status.", "An offline answer cannot confirm whether a road, bridge, shelter, warning, or rescue boat is available now.", "Do not make a movement plan around unverified live-status claims."][opener_index]
        return f"{opener}\n\nIf water is rising, do not wait on a lower floor only because rescue is rumored. Save battery, prepare one short location-and-needs message, move vulnerable people first if a safer higher place is reachable, and do not cross floodwater or fast current. Verify status locally when a trusted source is reachable.\n\n{red_flags}"
    if "dam" in hazard or "flash_flood" in hazard:
        return f"Do not treat a riverbank, storm drain, dry channel, or dam-release area as predictable during heavy rain or flooding.\n\nMove away from water edges, fast current, rising water, and debris flow. Keep children back, avoid driving or walking through moving water, and choose higher ground only by a path that does not cross water, debris, or unstable ground.\n\n{red_flags}"
    if "crowd_shelter_overcrowding" in hazard:
        opener = ["Do not let crowd pressure block doorways, aisles, stairs, exits, water points, or medical access.", "In an overcrowded shelter or queue, exit space and vulnerable-person movement come first.", "Do not turn distribution, charging, or shelter entry into a crush-risk situation."][opener_index]
        return f"{opener}\n\nKeep aisles and exits clear, slow the line, make movement one-way where possible, separate cooking/water/medical areas from crowding, and move children, elders, pregnant people, disabled people, and sick people to a quieter safer edge. Do not spread panic or blame one family.\n\n{red_flags}"
    if "shelter" in hazard:
        return f"Do not treat shelter hygiene or crowd pressure as a side issue when waste, diarrhea, unsafe water, or cooking areas are involved.\n\nMove food preparation away from waste and dirty water, set up handwashing or sanitizer at food/water points, prioritize safe water and ORS, and reduce shared cups/utensils where possible.{vulnerable_part} Watch for clusters of illness rather than blaming one family.\n\n{red_flags}"
    if "accessibility" in hazard:
        return f"Do not build the plan only around people who can move quickly and hear or read every instruction.\n\nIdentify who needs help: elders, pregnant people, disabled people, children, people with chronic illness, and anyone who does not understand the warning language. Keep medicines, mobility aids, glasses, documents, water, and a contact note with them, assign a helper, and use the shortest dry reachable route.\n\n{red_flags}"
    if "heatwave" in hazard or "lightning" in hazard or "cold" in hazard or "dust" in hazard:
        return f"Do not treat weather exposure as only discomfort; heat, cold, lightning, and dust can become dangerous quickly.\n\nFor heat, move to shade, rest, loosen clothing, and protect safe drinking water. For lightning, get away from open fields, trees, metal poles, and water edges. For cold or dust, keep people dry/covered and move breathing-risk people away from smoke and dust.\n\n{red_flags}"
    if "landslide" in hazard or "structural" in hazard:
        return f"Do not go back into a damaged building or near an unstable slope for documents or belongings.\n\nMove people away from cracks, falling stones, fresh debris, leaning walls, poles, or trees. Keep the group together, help children/elders/disabled people first, and choose a safer open or sturdy place only if you can reach it without crossing the hazard.\n\n{red_flags}"
    if "visual" in hazard:
        return f"Do not use a photo alone to confirm hidden danger, contamination, water depth/current, structure safety, or medicine identity.\n\nTreat the image as a clue, not proof. Choose the lower-risk option: do not give unknown tablets, do not use doubtful food or water, and keep people away from visible wires, damaged structures, or moving water until local checks or trained help are available.\n\n{red_flags}"
    if "urban_fire_lpg_chemical" in hazard:
        return f"Do not test for leaks with a flame, open electrical panels, or try repair steps during smoke, gas smell, or chemical exposure.\n\nMove people away from the source, avoid switches or sparks if gas is suspected, and help anyone with limited mobility leave the danger area if the path is clear. Keep others back and wait for trained help for repairs or shutoff if you cannot do it safely.\n\n{red_flags}"
    return f"Do not act on the risky assumption as if it is confirmed.\n\nStart with stable safety steps: move away from the immediate hazard, protect clean water/food and medicines, keep electricity dry, prioritize vulnerable people, and avoid forwarding unverified route, shelter, warning, or rescue claims.\n\n{red_flags}"


def build_scaled_rows(
    seed_card_path: Path = DEFAULT_SEED_CARDS,
    rule_manifest_path: Path = DEFAULT_RULE_MANIFEST,
    target_count: int = 720,
) -> list[dict[str, Any]]:
    seeds = [seed for seed in read_jsonl(seed_card_path) if not seed.get("reject_reason")]
    if not seeds:
        raise ValueError(f"No usable seed cards in {seed_card_path}")
    rule_rows = read_jsonl(rule_manifest_path)
    source_ids_by_rule = {row["rule_id"]: list(row.get("source_ids", [])) for row in rule_rows}
    rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_responses: set[str] = set()
    for index in range(target_count):
        seed = seeds[index % len(seeds)]
        variant_index = index // len(seeds)
        language = language_for_seed(seed, variant_index)
        prompt = prompt_for_seed(seed, variant_index, language)
        duplicate_count = 0
        while prompt in seen_prompts:
            vulnerable = compact_text(seed.get("vulnerable_factor"))
            extra = [
                f"Please keep it short because phone battery is low.",
                f"There is also concern for {vulnerable}." if vulnerable else "There are elders and children nearby.",
                "Answer for someone who may have to act before the network returns.",
                "Give the safest useful steps without assuming live local information.",
            ][(variant_index + duplicate_count) % 4]
            prompt = f"{prompt} {extra}"
            duplicate_count += 1
        seen_prompts.add(prompt)
        base_response = response_for_seed(seed, variant_index, language)
        response = sanitize_response_text(personalize_response(base_response, seed, language, variant_index))
        duplicate_response_count = 0
        while response in seen_responses:
            duplicate_response_count += 1
            response = sanitize_response_text(personalize_response(base_response, seed, language, variant_index + duplicate_response_count))
        seen_responses.add(response)
        rule_ids = list(seed.get("source_rule_ids", []))
        seed_source_ids = list(seed.get("source_ids", []))
        source_ids = sorted({source_id for rule_id in rule_ids for source_id in source_ids_by_rule.get(rule_id, [])} | set(seed_source_ids))
        row_id = f"beacon_asst_sft_v1_{index:04d}"
        row = {
            "row_id": row_id,
            "schema_version": "beacon-assistant-sft-v1",
            "split": seed.get("split", "train") if seed.get("split") in SPLITS else "train",
            "hazard": seed.get("primary_hazard", "unknown"),
            "risk_level": risk_from_seed(seed),
            "language": language,
            "user_prompt": prompt,
            "target_response": response,
            "messages": make_messages(prompt, response),
            "source_rule_ids": rule_ids,
            "source_ids": source_ids,
            "must_include": list(seed.get("must_say", []))[:4],
            "must_avoid": list(seed.get("must_not_say", []))[:4],
            "review_status": "pending",
            "review_notes": "",
            "training_ready": False,
            "draft_author": "seed_contract_naturalized_generator",
            "seed_id": seed.get("seed_id", ""),
            "seed_family_id": seed.get("seed_family_id", ""),
            "incident_archetype_id": seed.get("incident_archetype_id", ""),
            "difficulty_tier": seed.get("difficulty_tier", ""),
            "created_at_utc": utc_now(),
            "content_hash": sha256_text(prompt + "\n" + response),
        }
        rows.append(row)
    return rows


def split_for_index(index: int) -> str:
    if index in {2, 7}:
        return "dev"
    if index in {5, 11}:
        return "final_eval"
    return "train"


def make_messages(user_prompt: str, target_response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": target_response},
    ]


def gemma_text(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        chunks.append(f"<|turn>{role}\n{message['content'].strip()}<turn|>")
    return "\n".join(chunks)


def package_record(row: dict[str, Any]) -> dict[str, Any]:
    messages = list(row["messages"])
    return {
        "id": row["row_id"],
        "schema_version": row["schema_version"],
        "split": row["split"],
        "messages": messages,
        "text": gemma_text(messages),
        "prompt": row["user_prompt"],
        "target_response": row["target_response"],
        "hazard": row["hazard"],
        "risk_level": row["risk_level"],
        "language": row["language"],
        "source_rule_ids": row["source_rule_ids"],
        "source_ids": row["source_ids"],
        "must_include": row["must_include"],
        "must_avoid": row["must_avoid"],
        "review_status": row["review_status"],
        "training_ready": row["training_ready"],
        "content_hash": row["content_hash"],
    }


def build_rows(rule_manifest_path: Path = DEFAULT_RULE_MANIFEST) -> list[dict[str, Any]]:
    rule_rows = read_jsonl(rule_manifest_path)
    source_ids_by_rule = {row["rule_id"]: list(row.get("source_ids", [])) for row in rule_rows}
    rows: list[dict[str, Any]] = []
    for index, card in enumerate(SCENARIO_CARDS):
        source_ids = sorted({source_id for rule_id in card["source_rule_ids"] for source_id in source_ids_by_rule.get(rule_id, [])})
        row_id = f"beacon_asst_sft_v1_{index:03d}"
        row = {
            "row_id": row_id,
            "schema_version": "beacon-assistant-sft-v1",
            "split": split_for_index(index),
            "hazard": card["hazard"],
            "risk_level": card["risk_level"],
            "language": card["language"],
            "user_prompt": card["user_prompt"],
            "target_response": card["target_response"],
            "messages": make_messages(card["user_prompt"], card["target_response"]),
            "source_rule_ids": list(card["source_rule_ids"]),
            "source_ids": source_ids,
            "must_include": list(card["must_include"]),
            "must_avoid": list(card["must_avoid"]),
            "review_status": "pending",
            "review_notes": "",
            "training_ready": False,
            "draft_author": "human_seeded_pipeline_scaffold",
            "created_at_utc": utc_now(),
            "content_hash": sha256_text(card["user_prompt"] + "\n" + card["target_response"]),
        }
        rows.append(row)
    return rows


def write_review_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_id",
        "split",
        "hazard",
        "risk_level",
        "language",
        "review_status",
        "source_check_status",
        "safety_check_status",
        "style_check_status",
        "reviewer",
        "reviewed_at",
        "review_notes",
        "user_prompt",
        "target_response",
        "source_rule_ids",
        "must_include",
        "must_avoid",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "split": row["split"],
                    "hazard": row["hazard"],
                    "risk_level": row["risk_level"],
                    "language": row["language"],
                    "review_status": "pending",
                    "source_check_status": "pending",
                    "safety_check_status": "pending",
                    "style_check_status": "pending",
                    "reviewer": "",
                    "reviewed_at": "",
                    "review_notes": "",
                    "user_prompt": row["user_prompt"],
                    "target_response": row["target_response"],
                    "source_rule_ids": "|".join(row["source_rule_ids"]),
                    "must_include": "|".join(row["must_include"]),
                    "must_avoid": "|".join(row["must_avoid"]),
                }
            )


def write_split_files(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    files: dict[str, str] = {}
    for split in ["train", "dev", "final_eval"]:
        records = [package_record(row) for row in rows if row["split"] == split]
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        files[split] = path.name
    return files


def used_source_rule_rows(rows: list[dict[str, Any]], rule_manifest_path: Path) -> list[dict[str, Any]]:
    used_rule_ids = sorted({rule_id for row in rows for rule_id in row["source_rule_ids"]})
    by_id = {row["rule_id"]: row for row in read_jsonl(rule_manifest_path)}
    return [
        {
            "rule_id": rule_id,
            "derived_rule": by_id[rule_id].get("derived_rule", ""),
            "source_ids": by_id[rule_id].get("source_ids", []),
            "jurisdiction_scope": by_id[rule_id].get("jurisdiction_scope", ""),
            "review_status": by_id[rule_id].get("review_status", ""),
            "used_by_rows": [row["row_id"] for row in rows if rule_id in row["source_rule_ids"]],
        }
        for rule_id in used_rule_ids
        if rule_id in by_id
    ]


def write_source_rule_map(out_dir: Path, rows: list[dict[str, Any]], rule_manifest_path: Path) -> None:
    write_jsonl(out_dir / "source_rule_map.jsonl", used_source_rule_rows(rows, rule_manifest_path))


def write_design_note(out_dir: Path, manifest: dict[str, Any]) -> None:
    note = f"""# Beacon Assistant SFT v1 Draft Design

## Purpose

This package is for assistant-behavior SFT review, not CPT. The rows teach Beacon to notice risky assumptions, give practical offline steps, state uncertainty without empty refusal, and avoid fabricated live facts or unsafe medical/route certainty.

## Data Shape

- Canonical format: `messages` plus rendered Gemma-style `text`.
- Splits: train/dev/final_eval are written separately.
- Target style: natural user-assistant turns, varied wording, no visible scaffold labels such as `risk_level:` or `immediate_action:`.
- Grounding: each row carries `source_rule_ids`, `source_ids`, `must_include`, and `must_avoid`.

## Review Policy

Every row must pass human review for source support, safety, and assistant style before training. The current manifest keeps `training_export_allowed=false`.

## Current Counts

```json
{json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True)}
```

## Readiness

This is a structurally valid candidate package for review. It is not yet recommended for SFT until row review is completed and any flagged rows are edited or removed.
"""
    (out_dir / "dataset_design_note.md").write_text(note, encoding="utf-8")


def build_review_report(rows: list[dict[str, Any]], rule_manifest_path: Path) -> dict[str, Any]:
    _, row_report = validate_rows(rows, rule_manifest_path)
    rule_rows = used_source_rule_rows(rows, rule_manifest_path)
    pending_rules = [row["rule_id"] for row in rule_rows if row.get("review_status") != "accepted"]
    package_level_risks = [
        "human_review_pending_for_all_rows",
    ]
    if len(rows) < 600:
        package_level_risks.append("row_count_below_requested_sft_candidate_size")
    if pending_rules:
        package_level_risks.append("source_rule_manifest_entries_are_pending_main_review")
    weak_or_risky_rows: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if row["risk_level"] in {"critical", "high"}:
            reasons.append("high_stakes_row_needs_human_safety_review")
        if row["language"] == "hinglish":
            reasons.append("hinglish_naturalness_needs_reviewer_check")
        if "live_fact_uncertainty" in row["source_rule_ids"]:
            reasons.append("live_status_boundary_must_be_checked")
        if "damaged_medicine_label" in row["source_rule_ids"] or "insulin_storage_uncertainty" in row["source_rule_ids"]:
            reasons.append("medicine_boundary_must_be_checked")
        if reasons:
            weak_or_risky_rows.append({"row_id": row["row_id"], "hazard": row["hazard"], "reasons": reasons})
    return {
        "status": "review_required",
        "created_at_utc": utc_now(),
        "summary": "Candidate package passed deterministic lint, but is not approved for training until human review and scale-up.",
        "deterministic_row_report": row_report,
        "weak_or_risky_rows": weak_or_risky_rows,
        "package_level_risks": package_level_risks,
        "pending_source_rule_ids": pending_rules,
        "recommendation": "not_ready_for_sft_training",
        "next_steps": [
            "complete human review for all rows",
            "edit or remove weak rows found by review",
            "add tool-use/retrieval traces in a separate lane after answer-style rows are stable",
        ],
    }


def write_bundle(
    out_dir: Path,
    rule_manifest_path: Path = DEFAULT_RULE_MANIFEST,
    seed_card_path: Path | None = None,
    target_count: int | None = None,
) -> dict[str, Any]:
    rows = (
        build_scaled_rows(seed_card_path or DEFAULT_SEED_CARDS, rule_manifest_path, target_count)
        if target_count
        else build_rows(rule_manifest_path)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "draft_rows.jsonl", rows)
    split_files = write_split_files(out_dir, rows)
    write_source_rule_map(out_dir, rows, rule_manifest_path)
    write_review_queue(out_dir / "review_queue.csv", rows)
    manifest = {
        "schema_version": "beacon-assistant-sft-v1",
        "stage": "sft_draft_package_for_review",
        "training_export_allowed": False,
        "created_at_utc": utc_now(),
        "system_prompt": SYSTEM_PROMPT,
        "counts": {
            "total": len(rows),
            "by_split": dict(Counter(row["split"] for row in rows)),
            "by_hazard": dict(Counter(row["hazard"] for row in rows)),
            "by_language": dict(Counter(row["language"] for row in rows)),
            "by_risk": dict(Counter(row["risk_level"] for row in rows)),
        },
        "source_rule_manifest": str(rule_manifest_path),
        "source_rule_manifest_sha256": sha256_file(rule_manifest_path),
        "seed_cards": str(seed_card_path or "") if target_count else "",
        "seed_cards_sha256": sha256_file(seed_card_path or DEFAULT_SEED_CARDS) if target_count else "",
        "target_count": target_count or len(rows),
        "review_policy": "Every row must be reviewed for safety, source support, and assistant style before export.",
        "files": {
            "draft_rows": "draft_rows.jsonl",
            "train": split_files["train"],
            "dev": split_files["dev"],
            "final_eval": split_files["final_eval"],
            "source_rule_map": "source_rule_map.jsonl",
            "review_queue": "review_queue.csv",
            "dataset_design_note": "dataset_design_note.md",
            "review_report": "review_report.json",
        },
        "readiness_recommendation": "not_ready_for_sft_training",
    }
    write_json(out_dir / "dataset_manifest.json", manifest)
    write_design_note(out_dir, manifest)
    write_json(out_dir / "review_report.json", build_review_report(rows, rule_manifest_path))
    return manifest


def _pattern_hits(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    return [pattern.pattern for pattern in patterns if pattern.search(text)]


def _fabricated_live_fact_hits(text: str) -> list[str]:
    hits: list[str] = []
    negation_markers = [
        "cannot",
        "can't",
        "can not",
        "do not",
        "don't",
        "not ",
        "without",
        "unverified",
        "verify",
        "confirmed",
        "only if",
        "mat ",
        "nahi",
        "nahin",
    ]
    for pattern in LIVE_FACT_PATTERNS:
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 70) : min(len(text), match.end() + 90)].lower()
            if any(marker in window for marker in negation_markers):
                continue
            hits.append(pattern.pattern)
            break
    return hits


def _review_rows(review_path: Path) -> list[dict[str, str]]:
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_rows(rows: list[dict[str, Any]], rule_manifest_path: Path, real_eval_path: Path | None = DEFAULT_REAL_EVAL) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    rule_ids = {row["rule_id"] for row in read_jsonl(rule_manifest_path)}
    row_ids = [str(row.get("row_id", "")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        errors.append("duplicate row_id values")
    prompts = [str(row.get("user_prompt", "")).strip() for row in rows]
    if len(prompts) != len(set(prompts)):
        errors.append("duplicate user_prompt values")
    targets = [str(row.get("target_response", "")).strip() for row in rows]
    if len(targets) != len(set(targets)):
        errors.append("duplicate target_response values")
    target_hash_counts = Counter(sha256_text(target) for target in targets)
    duplicate_targets = [digest for digest, count in target_hash_counts.items() if count > 1]
    if duplicate_targets:
        errors.append(f"duplicate target_response hashes: {len(duplicate_targets)}")

    first_lines = [str(row.get("target_response", "")).strip().splitlines()[0].strip().lower() for row in rows if str(row.get("target_response", "")).strip()]
    first_line_counts = Counter(first_lines)
    most_common_first_line = first_line_counts.most_common(1)[0] if first_line_counts else ("", 0)
    if len(rows) >= 100 and most_common_first_line[1] / max(1, len(rows)) > 0.08:
        errors.append(f"over_repeated_opening: {most_common_first_line[1]} rows start with {most_common_first_line[0]!r}")
    if len(rows) >= 600 and len(first_line_counts) < 80:
        warnings.append(f"first-line diversity is low for scaled candidate: {len(first_line_counts)} unique openings")

    split_counts = Counter(row.get("split", "") for row in rows)
    hazard_counts = Counter(row.get("hazard", "") for row in rows)
    language_counts = Counter(row.get("language", "") for row in rows)
    if len(rows) >= 600:
        if not (600 <= len(rows) <= 800):
            errors.append(f"scaled candidate row count should be 600-800; got {len(rows)}")
        for split in ["train", "dev", "final_eval"]:
            if split_counts.get(split, 0) == 0:
                errors.append(f"scaled candidate missing split {split}")
        if split_counts.get("final_eval", 0) < 40 or split_counts.get("dev", 0) < 40:
            warnings.append("scaled candidate has fewer than 40 dev or final_eval rows")
        if len(hazard_counts) < 12:
            errors.append(f"scaled candidate hazard coverage too narrow: {len(hazard_counts)} hazards")
        if language_counts.get("hinglish", 0) < 80:
            warnings.append("scaled candidate has fewer than 80 Hinglish rows")

    eval_prompts: set[str] = set()
    if real_eval_path and real_eval_path.exists():
        eval_prompts = {str(row["user_prompt"]).strip().lower() for row in read_jsonl(real_eval_path)}

    for row in rows:
        rid = str(row.get("row_id", "<missing>"))
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            errors.append(f"{rid}: missing fields {missing}")
            continue
        if row["split"] not in SPLITS:
            errors.append(f"{rid}: unsupported split {row['split']!r}")
        if row["review_status"] != "pending":
            errors.append(f"{rid}: draft rows must start with review_status=pending")
        if row["training_ready"] is not False:
            errors.append(f"{rid}: draft rows must not be training_ready")
        if str(row["user_prompt"]).strip().lower() in eval_prompts:
            errors.append(f"{rid}: user_prompt exactly overlaps frozen real-problem eval")
        unknown_rules = sorted(set(row.get("source_rule_ids", [])) - rule_ids)
        if unknown_rules:
            errors.append(f"{rid}: unknown source_rule_ids {unknown_rules}")
        messages = row.get("messages", [])
        if not isinstance(messages, list) or [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            errors.append(f"{rid}: messages must be system/user/assistant")
        elif messages[1].get("content") != row["user_prompt"] or messages[2].get("content") != row["target_response"]:
            errors.append(f"{rid}: messages do not match prompt/target_response")

        response = str(row.get("target_response", ""))
        if len(response.split()) < 45:
            warnings.append(f"{rid}: response is short; reviewer should confirm usefulness")
        if len(response.split()) > 190:
            warnings.append(f"{rid}: response is long; reviewer should confirm it stays crisp")
        live_hits = _fabricated_live_fact_hits(response)
        if live_hits:
            errors.append(f"{rid}: fabricated_live_fact hits {live_hits[:3]}")
        for label, patterns in [
            ("unsafe_certainty", UNSAFE_CERTAINTY_PATTERNS),
            ("artifact_or_structured_renderer", ARTIFACT_PATTERNS),
            ("broken_style", BROKEN_STYLE_PATTERNS),
        ]:
            hits = _pattern_hits(patterns, response)
            if hits:
                errors.append(f"{rid}: {label} hits {hits[:3]}")
        if row.get("language") == "hinglish" and re.search(r"[\u0900-\u097F]", row["target_response"]):
            errors.append(f"{rid}: hinglish target should use Roman script unless user used Devanagari")
        photo_allowed = (
            "visual_cannot_determine" in row.get("source_rule_ids", [])
            or "no_photo_safety_certainty" in row.get("source_rule_ids", [])
            or "damaged_medicine_label" in row.get("source_rule_ids", [])
            or "photo" in str(row.get("user_prompt", "")).lower()
            or "image" in str(row.get("user_prompt", "")).lower()
            or "label" in str(row.get("user_prompt", "")).lower()
        )
        if "photo" in response.lower() and not photo_allowed:
            warnings.append(f"{rid}: photo/image boundary appears outside visual uncertainty rules")
        if row.get("language") == "hinglish":
            words = set(re.findall(r"[a-zA-Z]+", response.lower()))
            if not (words & HINDI_MARKERS):
                warnings.append(f"{rid}: Hinglish row has few Roman-Hinglish markers")
        if "what should I do" in str(row.get("user_prompt", "")).lower() and "do not" not in response.lower() and "mat " not in response.lower():
            warnings.append(f"{rid}: action-oriented prompt may need a clearer unsafe-boundary sentence")

    report = {
        "status": "pass" if not errors else "fail",
        "row_count": len(rows),
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "split": dict(Counter(row.get("split", "") for row in rows)),
            "hazard": dict(hazard_counts),
            "language": dict(language_counts),
            "risk": dict(Counter(row.get("risk_level", "") for row in rows)),
        },
        "diversity": {
            "unique_first_lines": len(first_line_counts),
            "top_first_line": most_common_first_line[0],
            "top_first_line_count": most_common_first_line[1],
        },
    }
    return errors, report


def validate_bundle(out_dir: Path, stage: str = "candidate", rule_manifest_path: Path = DEFAULT_RULE_MANIFEST) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    manifest_path = out_dir / "dataset_manifest.json"
    rows_path = out_dir / "draft_rows.jsonl"
    review_path = out_dir / "review_queue.csv"
    if not manifest_path.exists():
        return ["missing dataset_manifest.json"], {"status": "fail"}
    if not rows_path.exists():
        return ["missing draft_rows.jsonl"], {"status": "fail"}
    if not review_path.exists():
        return ["missing review_queue.csv"], {"status": "fail"}
    for name in ["train.jsonl", "dev.jsonl", "final_eval.jsonl", "source_rule_map.jsonl", "dataset_design_note.md", "review_report.json"]:
        if not (out_dir / name).exists():
            errors.append(f"missing {name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(rows_path)
    row_errors, row_report = validate_rows(rows, rule_manifest_path)
    errors.extend(row_errors)

    review_rows = _review_rows(review_path)
    review_by_id = {row.get("row_id", ""): row for row in review_rows}
    missing_review = sorted(set(row["row_id"] for row in rows) - set(review_by_id))
    if missing_review:
        errors.append(f"review_queue missing rows: {missing_review[:10]}")
    if manifest.get("training_export_allowed") is not False:
        errors.append("candidate manifest must keep training_export_allowed=false")
    if stage == "export":
        if manifest.get("training_export_allowed") is not True:
            errors.append("export requires training_export_allowed=true")
        incomplete = [
            row.get("row_id", "")
            for row in review_rows
            if row.get("review_status") != "approved"
            or row.get("source_check_status") != "approved"
            or row.get("safety_check_status") != "approved"
            or row.get("style_check_status") != "approved"
        ]
        if incomplete:
            errors.append(f"export requires all review checks approved; incomplete rows: {len(incomplete)}")

    report = {
        "status": "pass" if not errors else "fail",
        "stage": stage,
        "manifest": manifest,
        "row_report": row_report,
        "review_queue_count": len(review_rows),
        "errors": errors,
    }
    write_json(out_dir / "validation_report.json", report)
    return errors, report
