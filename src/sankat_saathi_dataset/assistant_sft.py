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
]


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

This is a structurally valid draft package and review seed. It is not yet recommended for SFT because the row count is intentionally small and review statuses are pending.
"""
    (out_dir / "dataset_design_note.md").write_text(note, encoding="utf-8")


def build_review_report(rows: list[dict[str, Any]], rule_manifest_path: Path) -> dict[str, Any]:
    _, row_report = validate_rows(rows, rule_manifest_path)
    rule_rows = used_source_rule_rows(rows, rule_manifest_path)
    pending_rules = [row["rule_id"] for row in rule_rows if row.get("review_status") != "accepted"]
    package_level_risks = [
        "small_pilot_size_not_enough_for_behavior_sft",
        "human_review_pending_for_all_rows",
    ]
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
            "expand toward the target 500-600 reviewed rows with the same gates",
            "add tool-use/retrieval traces in a separate lane after answer-style rows are stable",
        ],
    }


def write_bundle(out_dir: Path, rule_manifest_path: Path = DEFAULT_RULE_MANIFEST) -> dict[str, Any]:
    rows = build_rows(rule_manifest_path)
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
        live_hits = _fabricated_live_fact_hits(response)
        if live_hits:
            errors.append(f"{rid}: fabricated_live_fact hits {live_hits[:3]}")
        for label, patterns in [
            ("unsafe_certainty", UNSAFE_CERTAINTY_PATTERNS),
            ("artifact_or_structured_renderer", ARTIFACT_PATTERNS),
        ]:
            hits = _pattern_hits(patterns, response)
            if hits:
                errors.append(f"{rid}: {label} hits {hits[:3]}")
        if row.get("language") == "hinglish" and re.search(r"[\u0900-\u097F]", row["target_response"]):
            errors.append(f"{rid}: hinglish target should use Roman script unless user used Devanagari")
        if "what should I do" in str(row.get("user_prompt", "")).lower() and "do not" not in response.lower() and "mat " not in response.lower():
            warnings.append(f"{rid}: action-oriented prompt may need a clearer unsafe-boundary sentence")

    report = {
        "status": "pass" if not errors else "fail",
        "row_count": len(rows),
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "split": dict(Counter(row.get("split", "") for row in rows)),
            "hazard": dict(Counter(row.get("hazard", "") for row in rows)),
            "language": dict(Counter(row.get("language", "") for row in rows)),
            "risk": dict(Counter(row.get("risk_level", "") for row in rows)),
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
