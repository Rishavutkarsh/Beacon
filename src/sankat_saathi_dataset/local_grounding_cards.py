from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .local_grounding_research import (
    LIVE_STATUS_PATTERNS,
    MEDICINE_DOSE_PATTERNS,
    REQUIRED_HAZARD_FAMILIES,
    ROOT,
    read_jsonl,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "beacon-local-grounding-card-v1"
DEFAULT_RESEARCH_DIR = ROOT / "data" / "local_grounding" / "source_research_v1"
DEFAULT_CHUNKS_PATH = ROOT / "data" / "source_corpus" / "retrieval_chunks" / "retrieval_chunks.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "local_grounding" / "cards_v1"
CARD_STATUSES = {"draft", "needs_revision", "approved", "rejected"}
REVIEW_AXES = ["source_support", "safety", "retrieval_usefulness", "live_fact_fabrication"]


EXTRA_LIVE_PATTERNS = [
    "nearest shelter",
    "official warning says",
    "warning is active",
    "rescue is coming",
    "boat is coming",
    "hospital is open",
    "supplies are available",
    "phone line is working",
    "bus is running",
    "train is running",
]
EXTRA_MEDICINE_PATTERNS = [
    "insulin unit",
    "insulin units",
    "tablet count",
    "pediatric dose",
    "antibiotic",
    "skip dose",
    "skip the dose",
]


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    manifest: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def card(
    card_id: str,
    hazard_family: str,
    hazards: list[str],
    title: str,
    queries: list[str],
    core_guidance: str,
    why: str,
    safe_actions: list[str],
    red_flags: list[str],
    uncertainty_note: str,
    must_include: list[str],
    must_not_include: list[str],
    evidence: list[tuple[str, list[str], list[str]]],
    jurisdiction_scope: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "schema_version": SCHEMA_VERSION,
        "status": "draft",
        "hazard_family": hazard_family,
        "hazards": hazards,
        "title": title,
        "jurisdiction_scope": jurisdiction_scope or ["india_relevant", "global_applicable_when_local_status_unknown"],
        "audience": ["general_public", "volunteers"],
        "intent_tags": ["what_to_do", "what_not_to_do", "when_to_seek_help"],
        "retrieval_queries": queries,
        "answer_template": {
            "core_guidance": core_guidance,
            "why": why,
            "safe_actions": safe_actions,
            "red_flags": red_flags,
            "uncertainty_note": uncertainty_note,
        },
        "must_include": must_include,
        "must_not_include": must_not_include,
        "source_evidence": [
            {
                "document_id": document_id,
                "chunk_ids": chunk_ids,
                "support_type": "direct",
                "supported_claims": claims,
            }
            for document_id, chunk_ids, claims in evidence
        ],
        "review": {
            "required_review_count": 2,
            "final_status": "pending",
            "notes": [],
        },
    }


def draft_cards() -> list[dict[str, Any]]:
    cards = [
        card("flood_route_no_crossing_v1", "flood_route", ["flood", "floodwater", "route_safety"], "Do not cross floodwater", ["flooded road safe to cross", "walk through flood water", "drive through water"], "Do not walk, swim, or drive through floodwater. Turn back and choose higher ground or wait for a route that does not cross moving or unknown water.", "Floodwater can hide depth, current, debris, contamination, and electrical hazards.", ["Keep children away from water edges.", "Avoid bridges over fast-moving water.", "Preserve phone battery for a short location-and-needs message."], ["person swept away", "vehicle stuck", "fast current", "downed wire in water"], "This card cannot verify current road or bridge status.", ["avoid floodwater", "choose higher ground"], ["specific route status", "bridge is open"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0000"], ["avoid driving in flooded areas", "floodwater hazards"]), ("nws_turn_around", ["nws_turn_around_chunk_0000"], ["turn around behavior"])]),
        card("flood_route_vehicle_stall_v1", "flood_route", ["flood", "route_safety"], "Vehicle stuck near floodwater", ["car stalled in floodwater", "vehicle stuck water rising", "leave car in flood"], "If a vehicle is stalled in rising or moving water, focus on getting people away from the water if it is safe to exit. Do not try to save luggage or push through current.", "Vehicles can be swept away or stall in floodwater.", ["Move people before belongings.", "Avoid stepping into fast current.", "Signal location when communication is possible."], ["water entering vehicle", "fast current", "children or elders trapped"], "This card cannot estimate rescue arrival or road status.", ["people before belongings", "avoid current"], ["rescue will arrive", "route is safe"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0000"], ["cars do not protect from floodwater"]), ("ready_floods", ["ready_floods_chunk_0000"], ["move to higher ground"])]),
        card("floodwater_contact_basic_v1", "flood_route", ["floodwater", "contamination"], "Floodwater exposure", ["touched floodwater", "children playing flood water", "flood water contamination"], "Treat floodwater as contaminated even if it looks clear. Keep children out of it and wash exposed skin with soap and clean water when available.", "Floodwater can contain sewage, chemicals, sharp objects, animals, and germs.", ["Separate clean water and food from dirty areas.", "Wash hands before meals.", "Keep toys from floodwater away from children."], ["fever", "vomiting", "diarrhoea", "wound redness or swelling"], "Appearance alone cannot prove floodwater is safe.", ["contamination", "wash exposed skin"], ["clear water is safe", "photo proves safe"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0001", "cdc_floodwater_safety_chunk_0003"], ["floodwater contamination", "handwashing and children safety"])]),
        card("flood_reenter_daylight_power_v1", "wounds_cleanup", ["floodwater", "electrical", "mold"], "Re-entering a flooded home", ["return to flooded house", "enter house after flood", "cleanup after water recedes"], "Re-enter during daylight if possible and avoid using electrical tools or switches where water is present. If power cannot be shut off from a dry safe place, wait for trained help.", "Flooded buildings can have sewage, mold, gas leaks, and electrical hazards.", ["Use battery lights instead of candles or fuel lamps.", "Air out closed rooms briefly before staying.", "Leave if you smell gas or see structural danger."], ["gas smell", "standing water near power", "cracks", "breathing trouble"], "This card cannot certify a building is safe to enter.", ["daylight", "dry safe power shutoff"], ["building is safe", "turn power on in water"], [("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0000", "cdc_reenter_flooded_home_chunk_0001"], ["daylight reentry", "electrical and gas hazards"])]),
        card("flood_cleanup_mold_v1", "wounds_cleanup", ["flood_cleanup", "mold", "indoor_air"], "Mold and damp cleanup", ["mold after flood", "wet house cleanup", "musty smell after flood"], "Dry wet spaces as soon as safe, keep vulnerable people away from moldy areas, and avoid spreading mold-contaminated air through fans or systems that were flooded.", "Wet materials can grow mold and make indoor air unhealthy.", ["Ventilate when weather and safety allow.", "Keep clean bedding, medicines, and food away from cleanup dust.", "Use trained cleanup help for large contamination."], ["asthma attack", "breathing difficulty", "fever after exposure"], "This card cannot judge hidden mold from a photo or smell alone.", ["mold risk", "ventilate safely"], ["smell proves safe", "turn flooded HVAC on"], [("epa_flood_cleanup_iaq", ["epa_flood_cleanup_iaq_chunk_0000"], ["mold and flood cleanup risk"]), ("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0002", "cdc_reenter_flooded_home_chunk_0003"], ["drying and HVAC caution"])]),
        card("flood_propane_batteries_v1", "power_co_electrical", ["floodwater", "electrical", "fire"], "Propane tanks and batteries after flood", ["propane tank in flood", "car battery in water", "cleanup fuel cylinder"], "Do not move unknown propane tanks or damaged batteries found in floodwater. Keep people away and get local trained help when reachable.", "Flooded tanks and batteries can create fire, explosion, acid, and electrical hazards.", ["Mark the area from a distance.", "Keep children away.", "Avoid metal tools and wet contact."], ["leaking gas smell", "hissing cylinder", "sparks", "chemical burn"], "This card cannot identify whether a tank or battery is safe from appearance.", ["do not move tanks", "avoid batteries"], ["move cylinder yourself", "safe if not leaking"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0004"], ["propane and car battery hazards"])]),
        card("wet_electrical_switches_v1", "power_co_electrical", ["electrical", "floodwater"], "Wet switches and outlets", ["switch wet floor", "turn off power standing water", "outlet got wet"], "Do not touch switches, outlets, pumps, chargers, or appliances while standing in water. Shut power only from a dry reachable main switch; otherwise wait for trained help.", "Water can carry electrical current and wet equipment may remain dangerous.", ["Keep people away from wet electrical areas.", "Use battery lights.", "Do not reuse wet appliances until checked."], ["shock", "sparks", "burning smell", "unconscious person"], "This card cannot confirm power is off or a device is dry.", ["dry location only", "avoid wet equipment"], ["safe if switch looks dry", "use cloth to touch wire"], [("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0000", "cdc_reenter_flooded_home_chunk_0001"], ["do not use electric tools in water"]), ("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0004"], ["downed line and electrical hazards"])]),
        card("water_drinking_priority_v1", "water_wash", ["water_safety", "wash"], "Prioritize safest drinking water", ["drinking water after flood", "water supply disrupted", "safe water for child"], "Use the safest available water first for drinking, ORS, baby formula, medicines, and vulnerable people. Keep doubtful water separate and covered.", "After emergencies, tap or stored water may not be safe or available.", ["Use sealed bottled water first if available.", "Use clean covered containers.", "Avoid dipping dirty cups or hands into stored water."], ["unable to drink", "lethargy", "repeated vomiting", "infant without safe formula water"], "Clear appearance does not prove water is safe.", ["safest water priority", "covered storage"], ["clear water is safe", "untreated water for formula"], [("cdc_emergency_water", ["cdc_emergency_water_chunk_0000"], ["safe water after emergency"]), ("who_wash_emergencies", ["who_wash_emergencies_chunk_0000"], ["WASH risk in emergencies"])]),
        card("water_boil_disinfect_v1", "water_wash", ["water_safety", "disinfection", "boil_water"], "Boiling or disinfecting water", ["boil water emergency", "disinfect water bleach", "cloudy water safe"], "If bottled water is unavailable and water is not chemically contaminated, boiling is the safer first choice. If water is cloudy, let particles settle and filter through clean cloth before treatment.", "Disinfection works less well when water is cloudy and does not remove many chemicals.", ["Use clean covered containers after treatment.", "Follow product labels for disinfection tablets.", "Do not treat water with fuel or chemical smell as safe."], ["chemical smell", "oil sheen", "industrial spill", "severe diarrhoea"], "This card does not replace local boil-water advisories.", ["boil", "filter cloudy water"], ["boiling removes chemicals", "exact bleach dose from chat"], [("epa_emergency_disinfection", ["epa_emergency_disinfection_chunk_0000", "epa_emergency_disinfection_chunk_0001"], ["boil and filter cloudy water", "chemical caveat"]), ("cdc_emergency_water", ["cdc_emergency_water_chunk_0001"], ["bottled boiled treated water"])]),
        card("water_chemical_caveat_v1", "water_wash", ["water_safety", "chemical_contamination"], "Chemical smell or oil sheen", ["water smells like fuel", "oil sheen drinking water", "chemical contaminated water"], "Do not drink, cook with, or brush teeth with water that smells of fuel, has an oil sheen, or may be chemically contaminated. Use a different safer source.", "Boiling or simple disinfection may not remove many chemical contaminants.", ["Keep the suspect source labeled.", "Use sealed water if available.", "Avoid using it for baby formula or medicines."], ["burning throat", "vomiting after exposure", "chemical spill nearby"], "This card cannot identify a chemical or declare water safe from smell or appearance.", ["chemical caveat", "use different source"], ["boil chemicals away", "smell normal means safe"], [("epa_emergency_disinfection", ["epa_emergency_disinfection_chunk_0000"], ["boiling/disinfection does not remove many chemicals"]), ("cdc_emergency_water", ["cdc_emergency_water_chunk_0002"], ["unsafe water caveats"])]),
        card("water_storage_containers_v1", "water_wash", ["water_safety", "wash"], "Clean water storage", ["store emergency water", "container for drinking water", "dirty bucket water"], "Store drinking water in clean covered containers. Avoid containers that held fuel, pesticides, cleaners, or other toxic substances.", "Dirty or toxic containers can contaminate treated water again.", ["Label drinking water.", "Use a clean utensil each time.", "Keep containers away from chemicals and floodwater."], ["chemical taste", "container previously held pesticide", "child vomiting"], "This card cannot certify a reused container is food-safe.", ["clean covered container", "avoid toxic containers"], ["any plastic container is fine"], [("cdc_emergency_water", ["cdc_emergency_water_chunk_0002", "cdc_emergency_water_chunk_0003"], ["safe water storage containers"])]),
        card("wash_hand_hygiene_shelter_v1", "water_wash", ["wash", "sanitation", "shelter_hygiene"], "Hand hygiene in crowded shelters", ["handwash shelter", "hygiene after flood", "shared toilets disaster"], "In crowded shelters or cleanup areas, hand hygiene protects shared food and water. Wash hands after toilet use, cleanup, floodwater contact, and before preparing food.", "Poor sanitation and unsafe water increase diarrhoeal disease risk during emergencies.", ["Keep drinking water and handwashing water separate.", "Keep children away from sewage or mud.", "Reduce sick people handling shared food."], ["bloody stool", "many people vomiting", "dehydration signs"], "This card cannot confirm whether a shelter has safe sanitation now.", ["hand hygiene", "shared food protection"], ["crowd makes hygiene impossible"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0003"], ["wash hands after floodwater"]), ("who_wash_emergencies", ["who_wash_emergencies_chunk_0003", "who_wash_emergencies_chunk_0004"], ["WASH and hygiene importance"])]),
        card("ors_diarrhoea_dehydration_v1", "water_wash", ["water_safety", "diarrhoea", "ors", "dehydration"], "Diarrhoea and ORS", ["child diarrhoea flood", "ORS after disaster", "dehydration signs"], "For diarrhoea, replace fluids early. Use ORS prepared with the safest treated water available and do not change packet concentration.", "The biggest danger from diarrhoea is dehydration, especially for children.", ["Continue safe fluids and food as tolerated.", "Keep hands and utensils clean.", "Watch children, elders, and pregnant people closely."], ["blood in stool", "lethargy", "unable to drink", "repeated vomiting", "signs of shock"], "This card cannot diagnose the cause of diarrhoea.", ["ORS", "dehydration"], ["change ORS concentration", "ignore blood in stool"], [("who_diarrhoea", ["who_diarrhoea_chunk_0000", "who_diarrhoea_chunk_0001", "who_diarrhoea_chunk_0002"], ["dehydration and ORS"])]),
        card("food_flood_contact_v1", "food_safety", ["food_safety", "floodwater"], "Food touched by floodwater", ["rice sack flood water", "food touched floodwater", "carton got wet"], "Do not eat food that may have touched floodwater unless it is an undamaged waterproof commercial container that can be cleaned and sanitized.", "Floodwater can carry sewage, chemicals, and germs; cooking may not fix contamination history.", ["Separate doubtful food from clean supplies.", "Prioritize sealed dry food.", "Do not taste-test to decide."], ["vomiting", "diarrhoea", "fever", "child under five ill"], "This card cannot prove food safety from smell, heat, or a photo.", ["discard flood-contact food", "sealed dry food"], ["cooking fixes floodwater", "smell proves safe"], [("fda_food_water_floods", ["fda_food_water_floods_chunk_0000", "fda_food_water_floods_chunk_0001"], ["flood-contact food safety"]), ("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0006"], ["food may be unsafe even normal"])]),
        card("food_power_outage_fridge_v1", "food_safety", ["food_safety", "power_outage"], "Refrigerated food after power loss", ["fridge power cut food", "milk after outage", "food smells okay power outage"], "Do not rely on smell or taste after a long or uncertain power cut. Use shelf-stable food first and discard perishables when time or temperature safety is unclear.", "Unsafe food can still look, smell, and taste normal.", ["Keep fridge/freezer doors closed during outage.", "Feed infants, elders, and sick people lower-risk food first.", "Keep doubtful food away from shared meals."], ["vomiting", "diarrhoea", "fever", "dehydration"], "This card cannot reconstruct fridge temperature history.", ["do not rely on smell", "shelf-stable first"], ["reheating always fixes", "smell normal means safe"], [("cdc_food_after_emergency", ["cdc_food_after_emergency_chunk_0000"], ["food after emergency"]), ("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0006"], ["unsafe food can look normal"])]),
        card("food_relief_kitchen_v1", "food_safety", ["food_safety", "shelter_hygiene"], "Relief or community kitchen food", ["relief rice safe", "community kitchen disaster", "food distribution flood"], "Serve freshly cooked food made with safe water when possible, and keep cooked food protected from dirty hands, floodwater, and long warm holding.", "Crowding and unsafe water can turn shared food into a disease risk.", ["Separate cooking, toilet, and waste areas.", "Use clean utensils.", "Do not serve food from wet cardboard or porous sacks."], ["cluster of vomiting", "bloody stool", "fever", "dehydration"], "This card cannot confirm a specific kitchen is safe.", ["safe water cooking", "clean utensils"], ["hot means always safe"], [("fda_food_water_floods", ["fda_food_water_floods_chunk_0001"], ["food and water safety"]), ("who_wash_emergencies", ["who_wash_emergencies_chunk_0004"], ["hygiene and food safety control"])]),
        card("food_infant_formula_v1", "food_safety", ["food_safety", "water_safety"], "Infant formula after flood or outage", ["baby formula floodwater", "formula water disaster", "infant feeding flood"], "Do not use formula containers, bottles, or water that may be contaminated. Use sealed supplies and the safest treated water available for infant needs.", "Infants are highly vulnerable to unsafe water and contaminated packaging.", ["Keep infant supplies dry and separate.", "Use clean feeding utensils.", "Escalate quickly if safe feeding cannot be maintained."], ["infant not drinking", "repeated vomiting", "diarrhoea", "lethargy"], "This card cannot certify formula or water safety from packaging appearance.", ["infant priority", "safe water"], ["rinse flooded formula box", "dilute formula"], [("cdc_emergency_water", ["cdc_emergency_water_chunk_0000"], ["safe water for formula"]), ("fda_food_water_floods", ["fda_food_water_floods_chunk_0000"], ["flood-contact food packaging"])]),
        card("generator_co_distance_v1", "power_co_electrical", ["carbon_monoxide", "generators", "power_outage"], "Generator carbon monoxide", ["generator indoors rain", "generator in balcony", "CO symptoms outage"], "Keep generators and fuel-burning engines outside and away from doors, windows, vents, garages, stairwells, and sleeping areas. Fresh air comes first if people feel unwell.", "Carbon monoxide is invisible and can build up quickly from fuel-burning devices.", ["Move people to fresh air.", "Turn off the device only if safe.", "Watch for similar symptoms in more than one person."], ["confusion", "sleepiness", "breathing difficulty", "collapse", "multiple people dizzy"], "This card cannot measure CO levels or make indoor use safe.", ["outside away from openings", "fresh air"], ["open garage is safe", "one night is safe"], [("cdc_power_outage", ["cdc_power_outage_chunk_0000"], ["CO and generators during outage"]), ("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0004", "cdc_reenter_flooded_home_chunk_0005"], ["generator distance and enclosed spaces"])]),
        card("charcoal_stove_indoor_v1", "power_co_electrical", ["carbon_monoxide", "power_outage"], "Charcoal or stove indoors", ["charcoal indoors power cut", "cook inside during rain", "stove for heat outage"], "Do not use charcoal, camp stoves, grills, or other fuel-burning devices inside rooms, tents, garages, or enclosed spaces for cooking, heat, or light.", "Fuel-burning devices can cause carbon monoxide poisoning even when there is no smoke smell.", ["Move cooking outside away from openings.", "Keep sleeping areas separate from fumes.", "Move unwell people to fresh air."], ["headache with dizziness", "confusion", "vomiting", "fainting"], "This card cannot judge ventilation safety from windows or doors being open.", ["no fuel burning indoors", "fresh air"], ["smell proves safe", "window open is enough"], [("cdc_power_outage", ["cdc_power_outage_chunk_0000", "cdc_power_outage_chunk_0001"], ["fuel burning and CO risk"])]),
        card("fuel_siphoning_storage_v1", "power_co_electrical", ["power_outage", "fire"], "Fuel siphoning and storage", ["siphon petrol outage", "fuel storage disaster", "gasoline shortage"], "Do not siphon fuel by mouth or move fuel into unsafe containers. Keep fuel away from flames, cigarettes, sparks, children, and sleeping areas.", "Fuel vapors can poison, burn, ignite, or explode.", ["Use only proper containers if available.", "Ventilate fuel storage away from living spaces.", "Stop if there is dizziness or strong fumes."], ["burns", "confusion", "breathing trouble", "fuel swallowed"], "This card cannot make improvised fuel transfer safe.", ["do not siphon", "avoid sparks"], ["mouth siphon", "store fuel indoors"], [("cdc_power_outage", ["cdc_power_outage_chunk_0007"], ["siphoning gasoline risk"])]),
        card("outage_phone_battery_v1", "power_co_electrical", ["power_outage", "preparedness"], "Phone battery during outage", ["save phone battery disaster", "no network power cut", "send location message"], "Save battery for short essential messages. Prepare one concise message with location, people count, urgent needs, and hazards, then send when a trusted connection is available.", "Power and network disruption can make repeated messaging unreliable.", ["Lower screen brightness.", "Use SMS/voice only when needed.", "Keep one phone available for emergency contact."], ["rising water", "medical emergency", "trapped person"], "This card cannot confirm network coverage or rescue response.", ["battery conservation", "short location message"], ["rescue will see message"], [("ready_power_outages", ["ready_power_outages_chunk_0000"], ["power outage planning"]), ("who_risk_comm", ["who_risk_comm_chunk_0000"], ["clear risk communication"])]),
        card("outage_heat_cold_vulnerable_v1", "power_co_electrical", ["power_outage", "heat_cold", "vulnerable_people"], "Vulnerable people during outage", ["elder power outage heat", "child cold power cut", "disabled person outage"], "During outages, check infants, elders, pregnant people, disabled people, and people with chronic illness first for heat, cold, hydration, and medicine needs.", "Power loss can disrupt cooling, heating, food storage, medicines, and communication.", ["Move vulnerable people to the safest available room.", "Protect water, medicines, and assistive devices.", "Pair each vulnerable person with a helper if possible."], ["confusion", "not drinking", "breathing trouble", "very hot or very cold body"], "This card cannot identify an open cooling center or shelter.", ["vulnerable first", "medicine needs"], ["nearest shelter available"], [("cdc_power_outage", ["cdc_power_outage_chunk_0000"], ["power outage hazards"]), ("ready_heat", ["ready_heat_chunk_0000"], ["vulnerable heat risk"])]),
        card("wet_device_charging_v1", "power_co_electrical", ["electrical", "power_outage"], "Charging devices after water exposure", ["charge phone wet floor", "wet charger disaster", "phone battery flood"], "Do not charge phones or batteries from wet outlets, wet extension cords, or while standing on a damp floor. Use only a dry known-safe charging point.", "Wet charging setups can shock people or start a fire.", ["Keep chargers off wet floors.", "Dry hands before touching devices.", "Prioritize one shared phone if safe charging is limited."], ["shock", "sparks", "burning smell", "swollen battery"], "This card cannot confirm an outlet or charger is safe from appearance.", ["dry known-safe charging", "avoid wet cords"], ["stool makes charger safe"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0004"], ["electrical hazards"]), ("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0005"], ["electrical equipment dry before reuse"])]),
        card("medicine_wet_unknown_v1", "medicine_diabetes", ["medicine_disruption", "drug_safety"], "Wet or unknown medicines", ["medicine got wet flood", "tablet label unreadable", "identify pill photo"], "Do not take or give medicine when the label, identity, dose, or contamination history is uncertain. Protect known prescribed medicines and prescription information from water.", "Damaged labels and flood exposure can make medicine identity and safety uncertain.", ["Keep wet or unknown medicine separate.", "Use only clearly known prescribed medicine.", "Seek pharmacist or clinician help when reachable for critical medicines."], ["seizure medicine uncertain", "diabetes medicine uncertain", "confusion", "chest pain", "breathing difficulty"], "This card cannot identify pills from photos or partial labels.", ["unknown medicine boundary", "known prescription"], ["identify pill", "dose from photo"], [("fda_drugs_disaster", ["fda_drugs_disaster_chunk_0000"], ["safe drug use after disaster"]), ("cdc_diabetes_emergencies", ["cdc_diabetes_emergencies_chunk_0000"], ["prescription and medicine records"])]),
        card("insulin_temperature_disruption_v1", "medicine_diabetes", ["diabetes", "insulin", "medicine_disruption"], "Insulin storage disruption", ["insulin power outage heat", "insulin got warm", "diabetes medicine flood"], "If insulin storage was disrupted by heat, freezing, or floodwater, do not guess safety or change dosing from this card. Follow the person’s existing plan and seek clinician/pharmacist help when reachable.", "Insulin and diabetes supplies can be affected by emergency storage conditions.", ["Keep insulin as cool and dry as safely possible.", "Keep prescription details with the person.", "Watch food intake and symptoms closely."], ["unable to eat", "vomiting", "confusion", "fainting", "very sleepy"], "This card cannot decide whether a specific insulin vial is usable.", ["do not guess safety", "existing plan"], ["insulin units", "correction dose"], [("cdc_insulin_emergency", ["cdc_insulin_emergency_chunk_0000"], ["insulin emergency handling"]), ("cdc_diabetes_emergencies", ["cdc_diabetes_emergencies_chunk_0001"], ["diabetes emergency kit"])]),
        card("diabetes_meals_disrupted_v1", "medicine_diabetes", ["diabetes", "medicine_disruption"], "Diabetes when meals are disrupted", ["diabetes no food disaster", "missed meal insulin", "blood sugar emergency shelter"], "When meals are disrupted, avoid making medicine changes from guesswork. Keep the person observed, protect regular food and fluids as much as possible, and follow only a plan already taught to them.", "Diabetes emergencies can become dangerous when food, medicine, and monitoring are disrupted.", ["Keep fast sugar source if already part of their plan.", "Keep medicines and records together.", "Prioritize safe water and regular meals."], ["confusion", "fainting", "unable to eat", "repeated vomiting", "sweating with weakness"], "This card cannot prescribe diabetes medicine changes.", ["observe", "known plan"], ["double dose", "skip dose"], [("cdc_diabetes_emergencies", ["cdc_diabetes_emergencies_chunk_0000", "cdc_diabetes_emergencies_chunk_0002"], ["diabetes care during emergencies"])]),
        card("medicine_evacuate_go_bag_v1", "medicine_diabetes", ["medicine_disruption", "emergency_kit"], "Medicines during evacuation", ["leave medicines behind evacuation", "medicine go bag flood", "prescription emergency"], "If evacuation is safe and time allows, carry known critical medicines, prescription details, glasses, assistive devices, and basic supplies in a dry bag. Do not risk life to retrieve items.", "Emergencies can interrupt access to refills and medical records.", ["Keep medicines with the person who needs them.", "Separate wet unknown medicine.", "Record names and usual instructions if already known."], ["critical medicine missing", "seizure", "chest pain", "breathing difficulty"], "This card cannot promise refill access after evacuation.", ["critical medicines", "do not risk life"], ["refills always available"], [("cdc_diabetes_emergencies", ["cdc_diabetes_emergencies_chunk_0000"], ["emergency kit and records"]), ("fda_drugs_disaster", ["fda_drugs_disaster_chunk_0000"], ["drug safety after disaster"])]),
        card("cyclone_prepare_inside_v1", "cyclone_coastal", ["cyclone", "preparedness"], "Cyclone preparation indoors", ["cyclone coming what to do", "prepare home cyclone", "high wind warning"], "Before strong cyclone winds arrive, move loose outdoor items if safe, keep essentials together, charge devices from dry safe points, and stay away from windows during high winds.", "Cyclones can bring damaging winds, flooding, power outages, and debris.", ["Keep water, food, medicines, lights, and documents ready.", "Choose an interior safer area.", "Do not go outside to inspect damage during high winds."], ["roof damage", "flying debris", "water entering", "injury"], "This card cannot verify the current cyclone track or warning level.", ["cyclone preparation", "away from windows"], ["current warning level", "storm has passed"], [("ndma_cyclone", ["ndma_cyclone_chunk_0000"], ["cyclone preparedness"]), ("ready_floods", ["ready_floods_chunk_0000"], ["flood and outage hazards"])]),
        card("cyclone_coastal_evacuation_v1", "cyclone_coastal", ["cyclone", "coastal_evacuation", "shelter"], "Coastal evacuation uncertainty", ["storm surge evacuation", "coastal cyclone shelter", "should we wait by sea"], "If coastal flooding, storm surge, or evacuation is possible, do not wait in a low-lying place only because a rumor says help is coming. Move early if a safer higher place is reachable without crossing water.", "Coastal cyclone impacts can change quickly and low-lying areas may become hard to leave.", ["Move vulnerable people first.", "Carry medicines and documents if safe.", "Verify official instructions when reachable."], ["rising seawater", "fast water", "debris", "blocked exit"], "This card cannot confirm shelter availability, route status, or rescue timing.", ["higher place", "verify official instructions"], ["shelter is available", "rescue will arrive"], [("ndma_cyclone_guidelines_pdf", ["ndma_cyclone_guidelines_pdf_chunk_0020", "ndma_cyclone_guidelines_pdf_chunk_0021"], ["coastal evacuation and shelter planning"]), ("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0000"], ["floodwater danger"])]),
        card("cyclone_after_damage_v1", "cyclone_coastal", ["cyclone", "floodwater", "electrical"], "After cyclone damage", ["after cyclone go outside", "check damage after storm", "post cyclone safety"], "After severe wind or rain, avoid downed wires, unstable structures, floodwater, and damaged trees. Check people first before property.", "Post-cyclone injuries often come from hidden electrical, structural, and flood hazards.", ["Use daylight if possible.", "Keep children away from debris.", "Treat floodwater as contaminated."], ["downed wire", "gas smell", "unstable wall", "severe injury"], "This card cannot say the storm is over or an area is safe.", ["people first", "avoid wires and structures"], ["area is safe now"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0004"], ["electrical hazards after disasters"]), ("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0000"], ["daylight and safety after flood"])]),
        card("shelter_crowding_exits_v1", "shelter_vulnerable", ["shelter", "shelter_hygiene", "vulnerable_people"], "Crowded shelter exits", ["shelter crowded exits blocked", "sleep in stairwell shelter", "overcrowded relief camp"], "Do not block doors, stairs, aisles, water points, or medical access in a shelter. Keep movement paths open even when space is crowded.", "Blocked exits and crowding can turn a shelter into a new hazard.", ["Place children, elders, pregnant people, disabled people, and sick people near safer quieter edges.", "Separate cooking, waste, and sleeping areas where possible.", "Use calm one-way movement for queues."], ["exit blocked", "breathing distress", "crowd conflict", "fire or smoke"], "This card cannot confirm a specific shelter has space.", ["clear exits", "vulnerable people"], ["any indoor space is safer"], [("ndma_cyclone_guidelines_pdf", ["ndma_cyclone_guidelines_pdf_chunk_0050"], ["shelter management"]), ("unicef_wash_emergencies", ["unicef_wash_emergencies_chunk_0001"], ["WASH and vulnerable support"])]),
        card("shelter_children_hygiene_v1", "shelter_vulnerable", ["children", "wash", "shelter_hygiene"], "Children in shelters", ["children shelter hygiene", "kids flood camp", "child sick relief shelter"], "Keep children away from sewage, floodwater, waste, cooking fires, and crowded exits. Protect handwashing, safe drinking water, and clean feeding areas.", "Children are more vulnerable to dehydration, diarrhoeal disease, injury, and crowding.", ["Assign an adult to watch small children.", "Keep clean utensils for children.", "Separate sick children from shared food handling."], ["child lethargy", "repeated vomiting", "bloody stool", "breathing trouble"], "This card cannot judge disease spread in a specific shelter.", ["children away from hazards", "safe water"], ["toys from floodwater are safe"], [("cdc_floodwater_safety", ["cdc_floodwater_safety_chunk_0003"], ["children and floodwater"]), ("unicef_wash_emergencies", ["unicef_wash_emergencies_chunk_0000", "unicef_wash_emergencies_chunk_0001"], ["children and WASH in emergencies"])]),
        card("shelter_cooking_fire_v1", "shelter_vulnerable", ["shelter", "fire", "carbon_monoxide"], "Cooking in shelters", ["cook inside shelter", "stove in classroom shelter", "charcoal relief camp"], "Keep cooking and fuel-burning devices away from sleeping rooms, crowded areas, doors, and exits. Do not use charcoal or generators indoors.", "Fire, smoke, crowding, and carbon monoxide can spread quickly in shelters.", ["Create a separate cooking area if possible.", "Keep fuel away from children.", "Keep exits clear."], ["smoke indoors", "breathing trouble", "burns", "crowd panic"], "This card cannot certify any indoor cooking setup as safe.", ["separate cooking", "no fuel indoors"], ["cook in closed room"], [("cdc_power_outage", ["cdc_power_outage_chunk_0000"], ["fuel-burning CO risk"]), ("ready_wildfires", ["ready_wildfires_chunk_0000"], ["fire evacuation readiness"])]),
        card("landslide_warning_signs_v1", "landslide_structural", ["landslide", "structural"], "Landslide warning signs", ["landslide signs", "cracks after rain", "slope moving"], "Fresh cracks, leaning trees or poles, new seepage, rumbling, falling rocks, or fresh debris after rain are warning signs. Move away from the slope if a safer route exists.", "Saturated slopes can fail with little warning.", ["Avoid the base of steep slopes.", "Keep children away from fresh debris.", "Do not cross fresh mud or rockfall to save time."], ["rumbling", "fast debris flow", "house cracking", "blocked escape route"], "This card cannot certify slope stability.", ["warning signs", "move away"], ["familiar shortcut is safe"], [("ready_landslides", ["ready_landslides_chunk_0000", "ready_landslides_chunk_0001"], ["landslide signs and safety"])]),
        card("landslide_blocked_route_v1", "landslide_structural", ["landslide", "route_safety"], "Blocked route by debris", ["mudslide blocking road", "walk over landslide debris", "shelter across debris"], "Do not walk or drive over fresh landslide debris, flowing mud, or rockfall areas. Choose a different route or wait in a safer place away from the slope.", "Fresh debris can move again, hide water flow, or collapse under weight.", ["Stay upslope only if it does not put you near unstable ground.", "Watch for continuing rain.", "Warn others from a distance."], ["debris moving", "water flowing through mud", "new cracks", "injury"], "This card cannot confirm an alternate route is open.", ["avoid fresh debris", "wait safer place"], ["cross quickly"], [("ready_landslides", ["ready_landslides_chunk_0001", "ready_landslides_chunk_0002"], ["avoid landslide/debris flow danger"])]),
        card("structural_damage_return_v1", "landslide_structural", ["structural", "floodwater"], "Cracked or damaged building", ["wall crack after flood", "return cracked house", "building unsafe rain"], "Do not stay in or re-enter a building with new major cracks, leaning walls, shifting floors, gas smell, or water near electrical systems unless trained local help clears it.", "Floods, landslides, and storms can leave hidden structural and electrical hazards.", ["Move people out first if exit is safe.", "Avoid lighting flames if gas is suspected.", "Keep distance from damaged walls."], ["collapse sounds", "gas smell", "sparks", "trapped person"], "This card cannot inspect or approve a structure remotely.", ["do not reenter damaged building", "trained help"], ["looks okay from outside"], [("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0000", "cdc_reenter_flooded_home_chunk_0001"], ["reentry hazards"])]),
        card("heat_illness_first_steps_v1", "heat_cold_lightning", ["heatwave", "vulnerable_people"], "Heat illness first steps", ["heat stroke signs", "person dizzy in heat", "heat exhaustion disaster"], "Move the person to shade or a cooler place, loosen heavy clothing, cool the body with wet cloths, and give water or ORS only if the person is awake and able to drink.", "Heat illness can become life-threatening, especially with confusion, seizures, or unconsciousness.", ["Check elders, children, pregnant people, outdoor workers, and people with chronic illness.", "Rest during peak heat where possible.", "Avoid alcohol or caffeine for rehydration."], ["confusion", "seizure", "unconscious", "very high body heat", "worsening symptoms"], "This card cannot diagnose heat stroke.", ["cool place", "awake to drink"], ["force fluids unconscious"], [("ndma_heat_wave", ["ndma_heat_wave_chunk_0003"], ["heat illness steps"]), ("ready_heat", ["ready_heat_chunk_0000"], ["heat safety"])]),
        card("heat_prevention_vulnerable_v1", "heat_cold_lightning", ["heatwave", "vulnerable_people"], "Heat prevention for vulnerable people", ["protect elder heatwave", "baby heat wave", "outdoor worker heat"], "During heat, reduce exertion, rest in shade, drink safe fluids regularly, and check vulnerable people often. Do not leave children, elders, or sick people in hot closed spaces.", "Heat can overwhelm the body before people realize they are in danger.", ["Move activity to cooler hours.", "Use light clothing.", "Check urine, alertness, and ability to drink."], ["fainting", "confusion", "no sweating with very hot body", "unable to drink"], "This card cannot tell whether a local heat alert is active.", ["rest shade fluids", "check vulnerable"], ["wait until collapse"], [("ready_heat", ["ready_heat_chunk_0000", "ready_heat_chunk_0001"], ["heat prevention"]), ("ndma_heat_wave", ["ndma_heat_wave_chunk_0000"], ["heatwave public guidance"])]),
        card("cold_outage_hypothermia_v1", "heat_cold_lightning", ["cold_wave", "winter_storm", "power_outage"], "Cold during outage", ["cold wave power outage", "hypothermia signs", "keep warm no power"], "During cold outages, keep people dry, layer clothing and blankets, block drafts safely, and avoid fuel-burning heaters indoors. Check elders, infants, and sick people first.", "Cold stress and carbon monoxide can both become dangerous during outages.", ["Share warmth in one safe room.", "Keep medicines and water from freezing where possible.", "Move to fresh air if fuel fumes cause symptoms."], ["confusion", "shivering stops", "blue lips", "breathing trouble"], "This card cannot identify an open warming shelter.", ["keep dry layered", "no fuel indoors"], ["charcoal indoors for heat"], [("ready_winter", ["ready_winter_chunk_0000"], ["winter outage safety"]), ("cdc_power_outage", ["cdc_power_outage_chunk_0000"], ["CO risk during outages"])]),
        card("lightning_storm_safety_v1", "heat_cold_lightning", ["lightning", "storm"], "Lightning and storm safety", ["lightning outside shelter", "storm open field", "safe under tree lightning"], "During lightning, move inside a substantial building or hard-topped vehicle if reachable. Avoid open fields, isolated trees, water, metal objects, and rooftops.", "Lightning can strike before or after heavy rain and can travel through water or metal.", ["Wait before resuming outdoor work.", "Keep groups spread slightly while moving.", "Avoid using wired electrical items during a storm."], ["person struck", "unconscious", "burns", "breathing trouble"], "This card cannot predict where lightning will strike.", ["move indoors", "avoid trees/water/metal"], ["tree is safe"], [("nws_lightning", ["nws_lightning_chunk_0000"], ["lightning safety"])]),
        card("smoke_fire_evacuation_v1", "heat_cold_lightning", ["storm", "fire", "smoke", "evacuation"], "Smoke and fire movement", ["smoke near shelter", "wildfire smoke disaster", "fire evacuation"], "If smoke or fire is near, move away early by a visible safer path and protect breathing as much as possible. Do not wait for perfect information if conditions are worsening.", "Smoke can reduce visibility and harm breathing before flames arrive.", ["Keep low in smoke if trapped.", "Carry medicines and documents only if safe.", "Keep children and elders together."], ["breathing distress", "trapped by smoke", "burn injury", "fire blocking exit"], "This card cannot confirm fire direction or evacuation route status.", ["move away early", "protect breathing"], ["route is safe"], [("ready_wildfires", ["ready_wildfires_chunk_0000", "ready_wildfires_chunk_0001"], ["fire and smoke safety"])]),
        card("fake_alert_screenshot_v1", "misinformation_live_status", ["misinformation", "live_fact_uncertainty"], "Fake alerts and screenshots", ["whatsapp alert bridge open", "screenshot government logo", "fake disaster warning"], "Treat screenshots, forwards, and cropped alerts as unverified until checked against a trusted current source. Do not make risky movement decisions from a forward alone.", "Old, cropped, or fake messages can spread quickly during disasters.", ["Compare date, place, source, and full message.", "Use safer offline actions while waiting.", "Avoid forwarding uncertain claims."], ["rising water", "evacuation conflict", "crowd panic", "blocked exit"], "This card cannot verify current warnings or official status.", ["unverified screenshot", "do not forward"], ["logo proves true"], [("who_risk_comm", ["who_risk_comm_chunk_0000", "who_risk_comm_chunk_0001"], ["risk communication and trust"]), ("imd_weather_warnings", ["imd_weather_warnings_chunk_0001"], ["official weather portal context"])]),
        card("live_route_shelter_uncertainty_v1", "misinformation_live_status", ["live_fact_uncertainty", "route_safety", "shelter"], "Live route and shelter uncertainty", ["is shelter open", "is bridge open now", "rescue boat coming"], "An offline card cannot confirm whether a route, bridge, shelter, warning, hospital, supply point, or rescue is available now. Choose the safer action that does not depend on that claim.", "Current operational status changes quickly and can be wrong in rumors.", ["Move away from immediate hazards if a safer path is visible.", "Save battery for verified updates.", "Prepare a short location-and-needs message."], ["rising water", "fast current", "trapped person", "medical emergency"], "Verify current status locally or through trusted official channels when reachable.", ["cannot confirm live status", "safer action"], ["shelter available", "rescue timing"], [("who_risk_comm", ["who_risk_comm_chunk_0002"], ["uncertainty-aware communication"]), ("imd_weather_warnings", ["imd_weather_warnings_chunk_0001"], ["live official source context"])]),
        card("image_uncertainty_supplies_v1", "misinformation_live_status", ["live_fact_uncertainty", "food_safety", "water_safety"], "Photo cannot prove safety", ["photo wet medicine safe", "picture food safe", "image water clean"], "Treat photos as clues, not proof. A photo cannot confirm hidden contamination, medicine identity, electrical current, water safety, or structural stability.", "Many crisis hazards are invisible in an image.", ["Use lower-risk supplies when uncertain.", "Keep unknown medicines separate.", "Avoid visible wires, floodwater, and damaged structures."], ["unknown medicine needed", "chemical smell", "sparks", "severe symptoms"], "This card cannot identify safe food, water, medicine, or routes from an image.", ["photo not proof", "hidden hazards"], ["photo proves safe"], [("who_risk_comm", ["who_risk_comm_chunk_0000"], ["uncertainty communication"]), ("fda_drugs_disaster", ["fda_drugs_disaster_chunk_0000"], ["drug safety after disaster"])]),
    ]
    cards.extend(precision_anchor_cards())
    return cards


def precision_anchor_cards() -> list[dict[str, Any]]:
    return [
        card(
            "precision_food_fridge_temperature_v1",
            "food_safety",
            ["food_safety", "power_outage"],
            "Refrigerator and freezer safety temperatures",
            ["fridge safe temperature after outage", "refrigerator 40 F", "freezer 0 F"],
            "Use 40 F or below as the refrigerator safety threshold and 0 F or below as the freezer target. If food has been above 40 F for 4 hours or more, treat perishable food as unsafe.",
            "Food can become unsafe without smelling spoiled, and temperature history matters during outages.",
            ["Keep appliance thermometers if available.", "Keep fridge and freezer doors closed.", "Use shelf-stable food first when temperature history is uncertain."],
            ["vomiting", "diarrhoea", "fever", "dehydration"],
            "This card cannot reconstruct exact temperature history without a thermometer.",
            ["40 F or below", "0 F or below", "above 40 F for 4 hours or more"],
            ["60 F threshold", "smell proves safe", "reheating always fixes"],
            [("cdc_food_after_emergency", ["cdc_food_after_emergency_chunk_0000", "cdc_food_after_emergency_chunk_0001"], ["refrigerator 40 F or below", "freezer 0 F or below", "above 40 F discard guidance"])],
        ),
        card(
            "precision_food_fridge_freezer_times_v1",
            "food_safety",
            ["food_safety", "power_outage"],
            "Fridge and freezer outage times",
            ["fridge outage 4 hours", "half full freezer 24 hours", "full freezer 48 hours"],
            "If doors stay closed, a refrigerator keeps food safe for about 4 hours, a full freezer for about 48 hours, and a half-full freezer for about 24 hours.",
            "Fridge and freezer timelines differ, so mixing them up can make unsafe food look acceptable.",
            ["Keep doors closed as much as possible.", "Move refrigerated perishables to a cooler with ice if available after 4 hours.", "Discard perishables when time or temperature is unclear."],
            ["vomiting", "diarrhoea", "fever", "dehydration"],
            "This card cannot tell whether the door stayed closed or whether food remained cold.",
            ["fridge 4 hours", "full freezer 48 hours", "half-full freezer 24 hours"],
            ["fridge 24 hours", "half-full freezer 72 hours", "smell proves safe"],
            [("cdc_food_after_emergency", ["cdc_food_after_emergency_chunk_0001"], ["4 hours refrigerator", "48 hours full freezer", "24 hours half-full freezer"]), ("cdc_power_outage", ["cdc_power_outage_chunk_0002"], ["freezer and refrigerator outage times"])],
        ),
        card(
            "precision_water_stored_supply_v1",
            "water_wash",
            ["water_safety", "preparedness"],
            "Stored emergency water supply",
            ["how many days emergency water", "stored water supply disaster", "3 day water supply"],
            "For household preparedness, keep at least a 3-day supply of drinking water when possible, while still following local safety guidance after a disaster.",
            "Stored water reduces pressure to use doubtful water after floods, outages, or supply disruption.",
            ["Use sealed water first if available.", "Keep stored water covered and away from chemicals.", "Do not use water suspected of fuel or chemical contamination for drinking, cooking, brushing teeth, medicines, ORS, or formula."],
            ["unable to drink", "lethargy", "repeated vomiting", "blood in stool"],
            "This card cannot verify that stored water remained uncontaminated.",
            ["3-day supply", "safe stored water"],
            ["30-day CDC supply", "clear water is safe", "boiling fixes chemicals"],
            [("cdc_emergency_water", ["cdc_emergency_water_chunk_0000", "cdc_emergency_water_chunk_0001"], ["3-day water supply", "safe bottled boiled treated water"])],
        ),
        card(
            "precision_water_boiling_times_v1",
            "water_wash",
            ["water_safety", "boil_water", "disinfection"],
            "Boiling water times",
            ["boil water 1 minute", "boil water high altitude 3 minutes", "rolling boil emergency"],
            "Bring clear water to a rolling boil for 1 minute. At elevations above 6,500 feet, boil for 3 minutes.",
            "Boiling time is an exact safety constant and should not be confused with bleach wait time.",
            ["Filter cloudy water through clean cloth first or let particles settle.", "Store boiled water in clean covered containers.", "Use another source if fuel or toxic chemicals are suspected."],
            ["chemical smell", "oil sheen", "severe diarrhoea", "unable to drink"],
            "This card does not replace a local boil-water advisory.",
            ["rolling boil 1 minute", "above 6,500 feet 3 minutes"],
            ["30 minutes boiling", "30 seconds high altitude", "boiling removes chemicals"],
            [("cdc_emergency_water", ["cdc_emergency_water_chunk_0001"], ["rolling boil 1 minute", "above 6,500 feet 3 minutes", "chemical caveat"])],
        ),
        card(
            "precision_water_bleach_wait_v1",
            "water_wash",
            ["water_safety", "disinfection"],
            "Bleach disinfection wait time",
            ["bleach wait 30 minutes water", "drink immediately after bleach", "chlorine water stand time"],
            "After correctly disinfecting water with suitable unscented household bleach, let the water stand for at least 30 minutes before drinking.",
            "The wait time is part of the disinfection step; drinking immediately is unsafe.",
            ["Follow the bleach label or trusted local guidance for amount.", "Use clean covered containers after treatment.", "Do not use bleach treatment for water suspected of fuel, toxic chemicals, or radioactive material."],
            ["chemical smell", "vomiting after drinking", "severe diarrhoea", "child lethargic"],
            "This card does not provide a custom bleach dose from chat.",
            ["stand at least 30 minutes", "follow label", "not for chemical contamination"],
            ["drink immediately", "guess bleach amount", "boiling fixes chemicals"],
            [("cdc_emergency_water", ["cdc_emergency_water_chunk_0002"], ["stand at least 30 minutes before drinking", "follow bleach label", "chemical caveat"])],
        ),
        card(
            "precision_diabetes_low_sugar_quick_carbs_v1",
            "medicine_diabetes",
            ["diabetes", "medicine_disruption"],
            "Low blood sugar quick carbs",
            ["diabetes low blood sugar 15g", "quick carbs emergency diabetes", "glucose tablets disaster kit"],
            "For a known low-blood-sugar plan, emergency supplies should include glucose tablets or 15 grams of quick carbs such as juice, hard candy, or honey.",
            "Disrupted meals and evacuation can make diabetes routines fragile, but the assistant must not change medicines or identify pills.",
            ["Use the person's known diabetes plan when available.", "Keep medicines and supplies clearly labeled.", "Escalate if the person is confused, fainting, seizing, or cannot swallow safely."],
            ["confusion", "seizure", "fainting", "cannot swallow", "breathing trouble"],
            "This card cannot diagnose blood sugar level or prescribe medicine doses.",
            ["15 grams quick carbs", "known plan", "cannot swallow red flag"],
            ["no quick carbs", "extra tablet", "identify wet pills"],
            [("cdc_diabetes_emergencies", ["cdc_diabetes_emergencies_chunk_0001"], ["15 grams quick carbs", "diabetes emergency kit"])],
        ),
        card(
            "precision_lightning_sturdy_building_v1",
            "heat_cold_lightning",
            ["lightning", "storm"],
            "Safest place during thunderstorm",
            ["thunderstorm safest place", "lightning sturdy building", "safe under tree lightning"],
            "During lightning, the safer place is a sturdy building or a hard-topped vehicle if one is reachable. Avoid open fields, isolated trees, water, metal objects, and rooftops.",
            "Lightning can strike before heavy rain arrives and unsafe shelters can increase risk.",
            ["Move indoors before the storm is overhead when possible.", "Keep away from water and metal while moving.", "Help groups move without crowding dangerous edges."],
            ["person struck", "unconscious", "burns", "breathing trouble"],
            "This card cannot predict exactly where lightning will strike.",
            ["sturdy building", "hard-topped vehicle", "avoid open fields trees water metal"],
            ["shallow stream", "sheltering under a tree", "using a rooftop as shelter"],
            [("nws_lightning", ["nws_lightning_chunk_0000"], ["lightning safety"])],
        ),
        card(
            "precision_winter_slippery_roads_v1",
            "heat_cold_lightning",
            ["winter_storm", "cold_wave"],
            "Winter storm road hazard",
            ["winter storm road hazard", "slippery road traffic accidents", "blizzard road safety"],
            "Winter storms can make roads dangerous through snow, sleet, freezing rain, whiteout conditions, and slippery-road traffic accidents.",
            "The road hazard is not only house fires or cold exposure; movement itself can become unsafe.",
            ["Avoid unnecessary travel during hazardous winter conditions.", "Keep people warm and dry if sheltering in place.", "Do not use fuel-burning devices indoors for heat."],
            ["confusion", "hypothermia signs", "breathing trouble", "vehicle stranded"],
            "This card cannot verify current road conditions or travel safety.",
            ["slippery-road traffic accidents", "whiteout", "avoid unnecessary travel"],
            ["only house fires", "current road safety claim", "drive because route is familiar"],
            [("nws_winter", ["nws_winter_chunk_0000"], ["slippery roads traffic accidents", "winter storm hazards"])],
        ),
        card(
            "precision_pfa_do_not_minimize_v1",
            "shelter_vulnerable",
            ["pfa", "psychological_first_aid", "risk_communication", "vulnerable_people"],
            "PFA: do not minimize distress",
            ["psychological first aid bad behavior", "pfa weak exaggerating", "support after disaster"],
            "Psychological first aid should not shame people, treat them as weak, or say they are exaggerating. Supportive accompaniment, listening, and practical help are safer.",
            "After disaster stress, dismissive behavior can reduce trust and make people less likely to accept help.",
            ["Listen without forcing details.", "Offer practical help and accompaniment when wanted.", "Protect privacy and dignity."],
            ["self-harm talk", "panic that blocks safety", "violence risk", "unaccompanied child"],
            "This card cannot diagnose a mental health condition.",
            ["do not shame or minimize", "supportive accompaniment", "practical help"],
            ["calling someone weak", "saying they are exaggerating", "force details", "publicly shame"],
            [("who_risk_comm", ["who_risk_comm_chunk_0000", "who_risk_comm_chunk_0001"], ["trust and supportive risk communication"])],
        ),
        card(
            "precision_live_status_offline_boundary_v1",
            "misinformation_live_status",
            ["live_fact_uncertainty", "route_safety", "shelter"],
            "Offline boundary for live route or shelter status",
            ["is shelter open now", "bridge open now offline", "rescue boat coming"],
            "An offline assistant cannot confirm current shelter, route, bridge, hospital, supply, warning, or rescue status. Give safer actions that do not depend on that claim.",
            "Operational status changes quickly and rumors can be wrong or outdated.",
            ["Move away from immediate hazards when a safer visible path exists.", "Save battery for verified updates.", "Prepare a short location-and-needs message if communication returns."],
            ["rising water", "fast current", "trapped person", "medical emergency"],
            "Verify current status locally or through trusted official channels when reachable.",
            ["cannot confirm current status", "safer action without the claim"],
            ["shelter is open", "bridge is open", "rescue will arrive"],
            [("who_risk_comm", ["who_risk_comm_chunk_0002"], ["uncertainty-aware communication"]), ("imd_weather_warnings", ["imd_weather_warnings_chunk_0001"], ["official weather portal context"])],
        ),
        card(
            "precision_cyclone_track_uncertainty_v1",
            "cyclone_coastal",
            ["cyclone", "live_fact_uncertainty"],
            "Cyclone track and warning uncertainty",
            ["cyclone track exact village", "cyclone warning lead time", "landfall forecast uncertainty"],
            "Cyclone forecasts and warnings have uncertainty, especially farther ahead. Do not treat a forwarded track or old warning as exact local safety proof.",
            "Warnings may cover large areas because track and intensity forecasts are uncertain.",
            ["Prepare for wind, rain, flooding, and power disruption when in the possible impact area.", "Follow current trusted local instructions when reachable.", "Avoid coastal or flood-prone movement based only on rumors."],
            ["storm surge", "rising water", "damaged building", "downed wires"],
            "This card cannot verify current cyclone warnings or landfall track.",
            ["forecast uncertainty", "do not use old forward as proof", "verify current warning when reachable"],
            ["exact track certainty", "warning is active now", "safe because landfall moved"],
            [("ndma_cyclone_guidelines_pdf", ["ndma_cyclone_guidelines_pdf_chunk_0079", "ndma_cyclone_guidelines_pdf_chunk_0097"], ["forecast uncertainty", "warning stages"])],
        ),
        card(
            "precision_structural_no_reentry_v1",
            "landslide_structural",
            ["structural", "landslide"],
            "No quick re-entry into damaged structures",
            ["quickly retrieve documents damaged house", "wall crack after rain", "landslide house reentry"],
            "Do not re-enter a damaged building or slope-threatened structure just to retrieve documents, phones, or supplies. People come before belongings.",
            "Floods, storms, and landslides can leave hidden structural, electrical, gas, and debris hazards.",
            ["Move people away from damaged walls, slopes, and debris paths.", "Use supplies already outside the unsafe area.", "Ask trained local help for retrieval when reachable."],
            ["collapse sounds", "new cracks", "gas smell", "sparks", "trapped person"],
            "This card cannot inspect or clear a structure remotely.",
            ["no quick re-entry", "people before belongings", "trained retrieval help"],
            ["run in quickly", "looks safe outside", "documents justify entry"],
            [("cdc_reenter_flooded_home", ["cdc_reenter_flooded_home_chunk_0000", "cdc_reenter_flooded_home_chunk_0001"], ["reentry hazards"]), ("ready_landslides", ["ready_landslides_chunk_0002"], ["avoid landslide/debris flow danger"])],
        ),
    ]


def read_reviews(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def load_chunk_index(path: Path = DEFAULT_CHUNKS_PATH) -> dict[str, dict[str, Any]]:
    return {str(row["chunk_id"]): row for row in read_jsonl(path)}


def load_source_statuses(path: Path) -> dict[str, str]:
    return {str(row["document_id"]): str(row["review_status"]) for row in read_jsonl(path / "candidate_sources.jsonl")}


def reviews_by_card(reviews: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        grouped[str(review.get("card_id", ""))].append(review)
    return grouped


def card_claim_text(card: dict[str, Any]) -> str:
    answer = card.get("answer_template", {})
    fields = [
        card.get("title", ""),
        answer.get("core_guidance", ""),
        answer.get("why", ""),
        " ".join(answer.get("safe_actions", [])),
        " ".join(answer.get("red_flags", [])),
        answer.get("uncertainty_note", ""),
        " ".join(card.get("must_include", [])),
    ]
    return " ".join(str(field) for field in fields).lower()


def validate_cards(
    cards: list[dict[str, Any]],
    reviews: list[dict[str, Any]] | None = None,
    source_research_dir: Path = DEFAULT_RESEARCH_DIR,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> ValidationResult:
    reviews = reviews or []
    errors: list[str] = []
    warnings: list[str] = []
    chunk_index = load_chunk_index(chunks_path)
    source_status = load_source_statuses(source_research_dir)
    grouped_reviews = reviews_by_card(reviews)
    ids = [str(row.get("card_id", "")) for row in cards]
    if len(ids) != len(set(ids)):
        errors.append("card_id values must be unique")
    if not (36 <= len(cards) <= 60):
        warnings.append(f"draft card count is {len(cards)}, expected about 40")

    for row in cards:
        card_id = str(row.get("card_id", ""))
        status = str(row.get("status", ""))
        if row.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{card_id}: invalid schema_version")
        if status not in CARD_STATUSES:
            errors.append(f"{card_id}: invalid status {status!r}")
        family = str(row.get("hazard_family", ""))
        if family not in REQUIRED_HAZARD_FAMILIES:
            errors.append(f"{card_id}: invalid hazard_family {family!r}")
        elif not set(row.get("hazards", [])).intersection(REQUIRED_HAZARD_FAMILIES[family]):
            errors.append(f"{card_id}: hazards do not match hazard_family")
        for key in ["title", "retrieval_queries", "answer_template", "source_evidence"]:
            if not row.get(key):
                errors.append(f"{card_id}: missing {key}")
        answer = row.get("answer_template", {})
        for key in ["core_guidance", "why", "safe_actions", "red_flags", "uncertainty_note"]:
            if not answer.get(key):
                errors.append(f"{card_id}: missing answer_template.{key}")
        if len(str(answer.get("core_guidance", "")).split()) > 70:
            warnings.append(f"{card_id}: core_guidance is long")
        text = card_claim_text(row)
        for pattern in [*LIVE_STATUS_PATTERNS, *EXTRA_LIVE_PATTERNS]:
            if pattern in text:
                errors.append(f"{card_id}: contains live-status pattern {pattern!r}")
        for pattern in [*MEDICINE_DOSE_PATTERNS, *EXTRA_MEDICINE_PATTERNS]:
            if pattern in text:
                errors.append(f"{card_id}: contains medicine-dose pattern {pattern!r}")
        if "license" in text or "training_export" in text or "training export" in text:
            errors.append(f"{card_id}: user-facing card text leaks internal source/export language")

        for evidence in row.get("source_evidence", []):
            document_id = str(evidence.get("document_id", ""))
            if document_id not in source_status:
                errors.append(f"{card_id}: unknown source document {document_id}")
            elif source_status[document_id] in {"deferred", "rejected"}:
                errors.append(f"{card_id}: cites rejected/deferred source {document_id}")
            for chunk_id in evidence.get("chunk_ids", []):
                chunk = chunk_index.get(str(chunk_id))
                if not chunk:
                    errors.append(f"{card_id}: unknown chunk_id {chunk_id}")
                elif str(chunk.get("document_id")) != document_id:
                    errors.append(f"{card_id}: chunk {chunk_id} does not belong to {document_id}")

        if status == "approved":
            card_reviews = grouped_reviews.get(card_id, [])
            approving_reviewers = {
                str(review.get("reviewer_id"))
                for review in card_reviews
                if review.get("recommendation") == "approve" and all(review.get(axis) == "pass" for axis in REVIEW_AXES)
            }
            if len(approving_reviewers) < 2:
                errors.append(f"{card_id}: approved card requires two passing reviewer approvals")
            for review in card_reviews:
                if review.get("source_support") == "unsupported":
                    errors.append(f"{card_id}: approved card has unsupported source review")

    approved = [row for row in cards if row.get("status") == "approved"]
    draft_families = {str(row.get("hazard_family")) for row in cards}
    approved_families = {str(row.get("hazard_family")) for row in approved}
    for family in REQUIRED_HAZARD_FAMILIES:
        if family not in draft_families:
            errors.append(f"draft cards missing hazard family {family}")
        if approved and family not in approved_families:
            warnings.append(f"approved cards missing hazard family {family}")
    if approved and len(approved) < 28:
        warnings.append(f"approved card count below target after review: {len(approved)}")

    manifest = make_manifest(cards, reviews, errors, warnings)
    return ValidationResult(errors=errors, warnings=warnings, manifest=manifest)


def make_manifest(cards: list[dict[str, Any]], reviews: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    approved = [row for row in cards if row.get("status") == "approved"]
    return {
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "status": "valid" if not errors else "invalid",
        "draft_card_count": len(cards),
        "approved_card_count": len(approved),
        "review_count": len(reviews),
        "by_status": dict(Counter(str(row.get("status", "")) for row in cards).most_common()),
        "draft_by_hazard_family": dict(Counter(str(row.get("hazard_family", "")) for row in cards).most_common()),
        "approved_by_hazard_family": dict(Counter(str(row.get("hazard_family", "")) for row in approved).most_common()),
        "validation": {"errors": errors, "warnings": warnings},
    }


def coverage_rows(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in REQUIRED_HAZARD_FAMILIES:
        family_cards = [row for row in cards if row.get("hazard_family") == family]
        approved = [row for row in family_cards if row.get("status") == "approved"]
        rows.append({
            "hazard_family": family,
            "draft_card_count": len(family_cards),
            "approved_card_count": len(approved),
            "draft_card_ids": "|".join(sorted(str(row["card_id"]) for row in family_cards)),
            "approved_card_ids": "|".join(sorted(str(row["card_id"]) for row in approved)),
            "status": "covered" if family_cards else "gap",
        })
    return rows


def write_coverage_csv(path: Path, cards: list[dict[str, Any]]) -> None:
    fieldnames = ["hazard_family", "draft_card_count", "approved_card_count", "draft_card_ids", "approved_card_ids", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows(cards))


def render_report(cards: list[dict[str, Any]], reviews: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    lines = [
        "# Beacon Grounding Cards v1",
        "",
        "Compact local-grounding cards for Beacon. These are not SFT rows.",
        "",
        "## Summary",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Draft cards: {manifest['draft_card_count']}",
        f"- Approved cards: {manifest['approved_card_count']}",
        f"- Reviews: {manifest['review_count']}",
        "",
        "## Approved Cards",
        "",
    ]
    for row in cards:
        if row.get("status") == "approved":
            lines.append(f"- `{row['card_id']}` - {row['title']} ({row['hazard_family']})")
    lines.extend(["", "## Draft Or Held Cards", ""])
    for row in cards:
        if row.get("status") != "approved":
            lines.append(f"- `{row['card_id']}` - {row['title']} [{row['status']}]")
    lines.extend(["", "## Validation", ""])
    for error in manifest["validation"]["errors"]:
        lines.append(f"- ERROR: {error}")
    for warning in manifest["validation"]["warnings"]:
        lines.append(f"- WARNING: {warning}")
    if not manifest["validation"]["errors"] and not manifest["validation"]["warnings"]:
        lines.append("- No validation issues.")
    lines.append("")
    return "\n".join(lines)


def build_bundle(out_dir: Path = DEFAULT_OUT_DIR, reviews_path: Path | None = None) -> ValidationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = draft_cards()
    reviews = read_reviews(reviews_path or (out_dir / "grounding_card_reviews.jsonl"))

    grouped = reviews_by_card(reviews)
    for row in cards:
        card_reviews = grouped.get(str(row["card_id"]), [])
        approvals = {
            str(review.get("reviewer_id"))
            for review in card_reviews
            if review.get("recommendation") == "approve" and all(review.get(axis) == "pass" for axis in REVIEW_AXES)
        }
        if len(approvals) >= 2:
            row["status"] = "approved"
            row["review"]["final_status"] = "approved"
        elif card_reviews:
            row["status"] = "needs_revision"
            row["review"]["final_status"] = "needs_revision"

    validation = validate_cards(cards, reviews)
    write_jsonl(out_dir / "draft_grounding_cards.jsonl", cards)
    write_jsonl(out_dir / "approved_grounding_cards.jsonl", [row for row in cards if row.get("status") == "approved"])
    if not (out_dir / "grounding_card_reviews.jsonl").exists():
        write_jsonl(out_dir / "grounding_card_reviews.jsonl", reviews)
    write_coverage_csv(out_dir / "coverage_matrix.csv", cards)
    write_json(out_dir / "grounding_card_manifest.json", validation.manifest)
    (out_dir / "review_report.md").write_text(render_report(cards, reviews, validation.manifest), encoding="utf-8")
    return validation
