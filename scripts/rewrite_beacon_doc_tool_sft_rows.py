from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1"
OUT_PACKAGE = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1_rewritten"


CONTEXT_RE = re.compile(r"Context:\s*([^;]+);\s*([^;]+);\s*([^\.]+)\.", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def context_parts(user_prompt: str) -> tuple[str, str, str]:
    match = CONTEXT_RE.search(user_prompt)
    if not match:
        return "this situation", "people need a clear answer", "now"
    return tuple(part.strip() for part in match.groups())  # type: ignore[return-value]


def row_number(row: dict[str, Any]) -> int:
    row_id = str(row.get("row_id", "0"))
    digits = "".join(ch for ch in row_id if ch.isdigit())
    return int(digits[-4:] or "0")


def citations(row: dict[str, Any]) -> str:
    if str(row.get("row_family", "")).endswith("no_support"):
        return ""
    section_ids = list(row.get("section_ids", []))[:2]
    if not section_ids:
        return ""
    doc_id = str(row.get("doc_ids", [""])[0]) if row.get("doc_ids") else section_ids[0].split("_chunk_", 1)[0]
    pairs = []
    for section_id in section_ids:
        sec_doc = section_id.split("_chunk_", 1)[0] if "_chunk_" in section_id else doc_id
        pairs.append(f"{sec_doc}:{section_id}")
    return "\n\nEvidence: " + ", ".join(pairs) + "."


def lead(row: dict[str, Any]) -> str:
    context, pressure, timing = context_parts(str(row.get("user_prompt", "")))
    variants = [
        f"In the {context}, short answer:",
        f"Tell the group in the {context}:",
        f"At the {context} {timing}, use this boundary:",
        f"Because {pressure} in the {context}, be conservative:",
        f"In the {context}, do not guess:",
        f"Give people in the {context} this clear answer:",
        f"Keep it simple at the {context}:",
        f"This is a safety decision in the {context}:",
        f"With {pressure} at the {context}, use the safer rule:",
        f"Right now at the {context}:",
        f"The practical answer for the {context}:",
        f"Start with the safer assumption in the {context}:",
        f"Use a calm boundary in the {context}:",
    ]
    return variants[row_number(row) % len(variants)]


def context_close(row: dict[str, Any]) -> str:
    context, pressure, timing = context_parts(str(row.get("user_prompt", "")))
    variants = [
        f"That keeps the decision usable in the {context} while {pressure}.",
        f"At {timing}, avoid adding any detail the group cannot verify.",
        f"That is enough for the {context} without pretending to know live conditions.",
        f"If people need a quick answer, repeat the safety boundary and stop there.",
        f"Keep the group away from choices that depend on guesswork.",
        f"This gives the {context} a safer next step without overpromising.",
        f"Because {pressure}, do not add extra claims.",
        f"Use visible conditions and trusted local confirmation before acting further.",
        "Say what is known, say what is not known, and stop before guessing.",
        "Keep the advice useful, but do not fill gaps with memory.",
        "If the group must act now, choose the lower-risk option.",
        "Do not let urgency turn an unsupported claim into a fact.",
        "Repeat the practical step instead of adding new assumptions.",
        "The answer should help people pause before taking a risky shortcut.",
        "Keep the boundary firm even if someone wants a faster yes.",
        "Use the document result as a limit, not a launch point for guesses.",
        "A short cautious answer is better than a confident unsupported one.",
    ]
    return variants[(row_number(row) // 5) % len(variants)]


def finish(row: dict[str, Any]) -> str:
    context, pressure, timing = context_parts(str(row.get("user_prompt", "")))
    variants = [
        "Do not rely on smell, rumor, or appearance when the safety threshold matters.",
        "If the situation changes, choose the safer visible option and verify locally when possible.",
        "Keep children, elders, medicines, and clean water away from the risky area while deciding.",
        "If someone is pushing a shortcut, explain the boundary once and move to the safer action.",
        "Do not add route, shelter, rescue, or medicine details that the offline documents do not support.",
        "Keep the instruction short and conservative.",
        "Avoid decisions that depend on unverified live status.",
        "The safest next step is the one that does not require guessing.",
    ]
    return variants[(row_number(row) // 7) % len(variants)]


def key_fact_from_answer(answer: str) -> str:
    match = re.search(r"The key document-backed point is:\s*([^\.]+)\.", answer)
    return match.group(1).strip() if match else ""


def doc_title_from_answer(answer: str) -> str:
    match = re.search(r"use\s+(.+?)\s+as the offline source", answer, re.I)
    return match.group(1).strip() if match else "the offline document"


def rewrite_known_case(row: dict[str, Any]) -> str | None:
    case_id = str(row.get("case_family_id", ""))
    l = lead(row)
    f = finish(row)
    known: dict[str, str] = {
        "fridge_4h_40f": f"{l} keep the refrigerator and freezer doors closed. Refrigerator food is generally safe for about 4 hours, and perishables at 40 degrees or higher should be thrown out. {f}",
        "freezer_48_24": f"{l} do not assume a half-full freezer is safe for 72 hours. Use about 48 hours for a full freezer and about 24 hours for a half-full freezer if the door stayed closed. {f}",
        "boil_water_1_3": f"{l} bring clear emergency drinking water to a rolling boil. Use 1 minute normally and 3 minutes at higher elevation when that applies; do not mix this up with bleach standing time. {f}",
        "bleach_wait_30": f"{l} do not drink bleach-treated water immediately. Let treated water stand for 30 minutes, and do not invent a bleach amount in chat. {f}",
        "diabetes_quick_carbs": f"{l} do not remove quick sugar from a diabetes emergency kit. Keep glucose tablets or quick carbohydrates for low blood sugar; use the 15 grams rule only when it is part of the diabetes guidance. Do not change insulin or medicines from chat.",
        "generator_20ft": f"{l} do not run the generator on a balcony or near open doors. Keep it outdoors and at least 20 feet from windows, doors, and vents; carbon monoxide can still build up even when smoke seems to go outside.",
        "winter_road_hazard": f"{l} winter storms are not only a house-fire risk. Slippery or dangerous roads and traffic crashes are part of the hazard, so avoid unnecessary travel and do not minimize road danger.",
        "rewrite_hinglish_fridge_milk": f"{l} doodh ki smell theek lagna safety proof nahi hai. Power outage ke baad fridge food ke liye about 4 hours aur 40 degrees ka boundary rakho; doubt ho to use mat karo.",
        "rewrite_half_freezer_3_days": f"{l} aadha bhara freezer 3 din safe assume mat karo. Full freezer about 48 hours, half-full freezer about 24 hours tak maan sakte ho if door closed raha.",
        "rewrite_water_boiling_hill": f"{l} saaf dikhne wala paani bhi emergency me rolling boil tak lao. Normal rule 1 minute hai; high elevation me 3 minutes use karo. 30-minute boil mat invent karo.",
        "rewrite_bleach_now_drink": f"{l} bleach/disinfection ke baad paani turant mat pilao. Treated water ko 30 minutes stand karne do, aur bleach quantity chat se guess mat karo.",
        "rewrite_diabetes_sugar_bad": f"{l} diabetes kit se quick sugar hatao mat. Low blood sugar ke liye glucose tablets ya quick carbohydrates useful ho sakte hain; 15 grams rule ko guidance se hi use karo. Insulin dose chat se change mat karo.",
        "rewrite_generator_balcony": f"{l} generator balcony me chalana safe nahi hai. Use outdoors rakho, windows/doors/vents se at least 20 feet door. Khula darwaza carbon monoxide risk khatam nahi karta.",
        "rewrite_food_60f_myth": f"{l} rumor ka warmer threshold safe mat mano. Fridge food ke liye 40 degrees aur about 4 hours ka rule use karo; smell par bharosa mat karo.",
        "rewrite_fuel_smell_water_ors": f"{l} petrol ya chemical smell wale paani se ORS mat banao. Fuel/toxic chemical ka doubt ho to boiling ya disinfecting se paani safe nahi hota; bottled ya doosra safer source use karo.",
    }
    if case_id in known:
        return known[case_id] + " " + context_close(row) + citations(row)
    return None


def rewrite_no_support(row: dict[str, Any]) -> str:
    case_id = str(row.get("case_family_id", ""))
    l = lead(row)
    context, pressure, _ = context_parts(str(row.get("user_prompt", "")))
    variants = {
        "live_route": "offline documents cannot verify whether the bridge is open, whether a shelter has beds, or when rescue will arrive. Treat the forward as unverified and choose only actions that are safe from what people can see locally.",
        "medicine_photo": "offline documents cannot identify a wet or unknown tablet from chat or give a dose. Keep packaging if available, separate wet/unknown medicines, and ask a pharmacist, doctor, or health worker when reachable.",
        "structural_photo": "a photo or offline document cannot certify a cracked wall, slope, bridge, or building as safe. Keep people away from the risky area and wait for trained local assessment when possible.",
        "lightning_gap": "this local offline result is not enough to prove the exact safest lightning shelter. Do not guess; move away from exposed water/open areas if a safer sheltered place is visibly reachable.",
        "relief_stock_status": "offline documents cannot confirm whether food packets or supplies are available right now. Treat the message as unverified and check a physically present organizer or trusted local channel.",
        "unknown_injection": "offline documents cannot identify an unlabeled injection or tell anyone to use it. Keep it aside and ask a qualified health worker or pharmacist when reachable.",
        "slope_reentry": "offline documents cannot certify a cracked slope or nearby home as safe for tonight. Keep people away from the slope-facing risk area and avoid re-entry for belongings.",
        "open_wire_photo": "offline documents cannot certify from a photo that a wet wire is de-energized. Keep people away and do not touch it.",
        "rescue_eta": "offline documents cannot confirm a rescue ETA. Do not repeat the one-hour claim as verified; plan for safer waiting or movement based on visible conditions.",
        "pharmacy_substitute": "offline documents cannot choose a prescription substitute. Do not swap medicines from chat; preserve the old strip or prescription and ask a pharmacist, doctor, or health worker.",
        "bridge_crack": "offline documents cannot certify a cracked bridge as safe to cross. Keep people off it and use a safer known route only if it is visibly safe.",
        "shelter_bed_count": "offline documents cannot know tonight's shelter bed count. Do not promise availability; verify with people managing the shelter when reachable.",
    }
    for key, body in variants.items():
        if key in case_id:
            return f"{l} {body} {context_close(row)}"
    return f"{l} offline documents cannot verify that specific claim. Avoid guessing and choose the safer visible option. {context_close(row)}"


def rewrite_doc_index(row: dict[str, Any]) -> str:
    hazard = str(row.get("hazard", "crisis safety")).replace("_", " ")
    title = doc_title_from_answer(str(row.get("target_response", "")))
    fact = key_fact_from_answer(str(row.get("target_response", "")))
    l = lead(row)
    f = finish(row)
    if fact:
        bodies = [
            f"{l} the offline {title} guidance gives one stable point for {hazard}: {fact}. Use it as a boundary, not as a live local-status claim. {f}",
            f"{l} for {hazard}, the useful point from {title} is {fact}. Keep the answer limited to that stable safety point and avoid adding current local details. {f}",
            f"{l} the practical takeaway for {hazard} is {fact}. That is a document-based safety detail, not a promise about what is happening nearby. {f}",
            f"{l} use {fact} as the key {hazard} detail from {title}. Do not expand it into route, shelter, rescue, or medicine claims. {f}",
        ]
    else:
        bodies = [
            f"{l} {title} gives general offline guidance for {hazard}. Summarize only the stable safety point from the document and do not add current local status. {f}",
            f"{l} for {hazard}, keep the answer tied to the offline {title} guidance. Avoid exact claims unless the document section actually gives them. {f}",
            f"{l} use the {title} guidance as a stable reference for {hazard}, then give a cautious next step that does not depend on live conditions. {f}",
            f"{l} the helpful answer is a short {hazard} safety boundary from {title}, without guessing about local roads, shelters, or services. {f}",
        ]
    return bodies[row_number(row) % len(bodies)] + " " + context_close(row) + citations(row)


def rewrite_no_tool(row: dict[str, Any]) -> str:
    case_id = str(row.get("case_family_id", ""))
    l = lead(row)
    context, pressure, _ = context_parts(str(row.get("user_prompt", "")))
    if "broad_cleanup" in case_id:
        return f"{l} start with people, not belongings. Move away from visible hazards, keep children and elders with the group, protect clean water and medicines, and do not spread unverified route or shelter claims. {context_close(row)}"
    if "calm_rumor" in case_id:
        return f"{l} do not repeat the forward as fact. Ask what people can verify nearby, avoid choices that depend on live status, and choose the option with fewer visible hazards. {context_close(row)}"
    if "general_outage" in case_id:
        return f"{l} keep the priorities broad: heat, smoke, wet electricity, fumes, medicines, and clean water. Exact food or medicine thresholds should be checked in official documents. {context_close(row)}"
    if "children_shelter" in case_id:
        return f"{l} keep children with known caregivers, away from damaged areas and dirty cleanup zones. Use calm, concrete instructions and do not promise supplies or beds. {context_close(row)}"
    if "neighbor_checkin" in case_id:
        return f"{l} check that elders are away from wires, floodwater, heat, falls, and damaged walls. Keep medicines and drinking water together, and do not promise rescue or services. {context_close(row)}"
    if "camp_queue" in case_id:
        return f"{l} acknowledge the stress, keep families together, share only what nearby organizers have confirmed, and avoid rumors about supplies or space. {context_close(row)}"
    if "bus_stand" in case_id:
        return f"{l} do not guess movement status. Keep people away from water, wires, unstable structures, and traffic; conserve phone battery and decide from visible hazards. {context_close(row)}"
    if "family_supplies" in case_id:
        return f"{l} prioritize people, medicines, clean water, dry clothes, basic contacts, and safer movement. Leave exact food, water-treatment, and medicine rules for a checked lookup. {context_close(row)}"
    return f"{l} give broad safety priorities, keep people away from visible hazards, and avoid guesses while {pressure}. {context_close(row)}"


def rewrite_row(row: dict[str, Any]) -> dict[str, Any]:
    new = json.loads(json.dumps(row))
    family = str(row.get("row_family", ""))
    if family == "no_tool_needed":
        answer = rewrite_no_tool(row)
    elif family.endswith("no_support"):
        answer = rewrite_no_support(row)
    else:
        answer = rewrite_known_case(row) or rewrite_doc_index(row)
    new["target_response"] = answer
    new["rewrite_pass"] = {
        "name": "beacon_doc_tool_sft_v1_rewrite_v2",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "method": "deterministic natural-language rewrite of assistant final turns; tool calls/results preserved",
    }
    messages = list(new.get("messages", []))
    if messages and messages[-1].get("role") == "assistant":
        messages[-1]["content"] = answer
        new["messages"] = messages
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PACKAGE)
    parser.add_argument("--out", type=Path, default=OUT_PACKAGE)
    args = parser.parse_args()
    rows = read_jsonl(args.source / "all_rows.jsonl")
    rewritten = [rewrite_row(row) for row in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "all_rows.jsonl", rewritten)
    for split in ["train", "dev", "final_eval"]:
        write_jsonl(args.out / f"{split}.jsonl", [row for row in rewritten if row.get("split") == split])
    manifest = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_package": str(args.source),
        "row_count": len(rewritten),
        "by_split": dict(Counter(str(row.get("split", "")) for row in rewritten).most_common()),
        "by_family": dict(Counter(str(row.get("row_family", "")) for row in rewritten).most_common()),
        "training_export_allowed": False,
        "status": "candidate_rewritten_needs_review",
    }
    write_json(args.out / "rewrite_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
