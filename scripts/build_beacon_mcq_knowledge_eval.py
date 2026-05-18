from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUT_DIR = Path("data/eval/beacon_mcq_knowledge_v1")
CPT_DIR = Path("data/dapt_corpus/beacon_crisis_v1_cpt_kaggle")
SCHEMA_PUBLIC = "beacon-mcq-knowledge-v1"
SCHEMA_KEY = "beacon-mcq-answer-key-v1"
PROMPT_TEMPLATE_ID = "mcq_letters_only_v1"
EXPECTED_ROWS = 80
EXPECTED_LABEL_COUNTS = {"A": 20, "B": 20, "C": 20, "D": 20}


@dataclass(frozen=True)
class MCQSpec:
    eval_bucket: str
    cpt_exposure: str
    source_split: str
    hazard_bucket: str
    difficulty: str
    source_family: str
    question_stem: str
    correct: str
    distractors: tuple[str, str, str]
    required_facts: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    unsafe_distractor_indices: tuple[int, ...] = ()
    critical_safety_subset: bool = False


def spec(
    eval_bucket: str,
    source_split: str,
    hazard_bucket: str,
    difficulty: str,
    source_family: str,
    question_stem: str,
    correct: str,
    distractors: list[str],
    required_facts: list[str],
    evidence_terms: list[str],
    unsafe_distractor_indices: list[int] | None = None,
    critical_safety_subset: bool = False,
) -> MCQSpec:
    exposure = {"test": "heldout_test", "train": "train_seen_probe", "dev": "dev_calibration"}[source_split]
    return MCQSpec(
        eval_bucket=eval_bucket,
        cpt_exposure=exposure,
        source_split=source_split,
        hazard_bucket=hazard_bucket,
        difficulty=difficulty,
        source_family=source_family,
        question_stem=question_stem,
        correct=correct,
        distractors=tuple(distractors),
        required_facts=tuple(required_facts),
        evidence_terms=tuple(evidence_terms),
        unsafe_distractor_indices=tuple(unsafe_distractor_indices or ()),
        critical_safety_subset=critical_safety_subset,
    )


SPECS: list[MCQSpec] = [
    # Held-out test: carbon monoxide clinical guidance.
    spec("heldout_test", "test", "carbon_monoxide", "medium", "cdc_co_clinical_disasters", "Which property makes carbon monoxide especially hard for people to detect without an alarm?", "It is odorless and colorless.", ["It has a strong sulfur smell.", "It turns visible in humid air.", "It causes immediate skin burning."], ["CO is odorless", "CO is colorless"], ["odorless", "colorless"], [0], True),
    spec("heldout_test", "test", "carbon_monoxide", "hard", "cdc_co_clinical_disasters", "Why can a generator outside but near an open window still be dangerous?", "Carbon monoxide can enter and build up inside the home.", ["Fresh air from the window neutralizes CO.", "The danger is only electrical shock.", "CO cannot move through open windows."], ["CO can enter openings", "CO can build up indoors"], ["outside near an open window", "poison"], [0, 2], True),
    spec("heldout_test", "test", "carbon_monoxide", "medium", "cdc_co_clinical_disasters", "Which set lists common symptoms of carbon monoxide poisoning from the CDC guidance?", "Headache, dizziness, weakness, nausea, vomiting, chest pain, and altered mental status.", ["Fever, rash, swollen joints, and nosebleeds.", "Burning eyes only, with no systemic symptoms.", "Sneezing, itching, and watery eyes only."], ["common CO symptoms"], ["headache", "dizziness", "altered mental status"], [], True),
    spec("heldout_test", "test", "carbon_monoxide", "hard", "cdc_co_clinical_disasters", "When should clinicians especially consider CO poisoning according to the guidance?", "No fever with symptoms, exposure history, or multiple patients with similar complaints.", ["High fever plus a single isolated patient only.", "Only when a patient smells gas clearly.", "Only after a positive home smoke alarm."], ["no fever", "exposure history", "multiple similar patients"], ["No fever", "multiple patients"], [], True),
    spec("heldout_test", "test", "carbon_monoxide", "hard", "cdc_co_clinical_disasters", "What COHgb level strongly supports CO poisoning in a non-smoker?", "An elevated COHgb level of 2 percent.", ["Any COHgb below 1 percent.", "Only a level above 30 percent.", "A normal oxygen saturation reading."], ["2 percent COHgb for non-smokers"], ["2%", "non-smokers"], [], True),
    spec("heldout_test", "test", "carbon_monoxide", "hard", "cdc_co_clinical_disasters", "What COHgb level strongly supports CO poisoning in smokers?", "A COHgb level greater than 9 percent.", ["A COHgb level of exactly 1 percent.", "Only a level above 60 percent.", "Any pulse oximeter reading above 95 percent."], ["greater than 9 percent COHgb for smokers"], [">9%", "smokers"], [], True),
    spec("heldout_test", "test", "carbon_monoxide", "hard", "cdc_co_clinical_disasters", "What oxygen treatment does the CDC clinical guidance recommend until symptoms resolve?", "Administer 100 percent oxygen, usually for about 4 to 5 hours.", ["Give room air and wait for symptoms to fade.", "Use oxygen only after fever appears.", "Give oxygen for exactly 15 minutes then stop."], ["100 percent oxygen", "about 4 to 5 hours"], ["100% oxygen", "4-5 hours"], [0], True),
    # Held-out test: food, water, volcano, lightning, winter, flood preparedness.
    spec("heldout_test", "test", "food_after_emergency", "medium", "cdc_food_after_emergency", "What refrigerator temperature should emergency food safety planning target?", "The refrigerator should be at 40 F or below.", ["The refrigerator should be at 60 F or below.", "Any cool-feeling refrigerator is safe.", "Temperature does not matter if doors stay closed."], ["refrigerator 40 F or below"], ["40", "refrigerator"], [1, 2], True),
    spec("heldout_test", "test", "food_after_emergency", "medium", "cdc_food_after_emergency", "How long can refrigerator food stay safe if doors remain closed during an outage?", "Up to 4 hours.", ["Up to 24 hours.", "Up to 48 hours.", "Until it smells bad."], ["4 hours refrigerator"], ["4 hours", "refrigerator"], [3], True),
    spec("heldout_test", "test", "food_after_emergency", "medium", "cdc_food_after_emergency", "With doors closed, how long can food stay safe in a full freezer?", "Up to 48 hours.", ["Up to 4 hours.", "Up to 12 hours.", "Until the package sweats."], ["48 hours full freezer"], ["48 hours", "full freezer"], [], True),
    spec("heldout_test", "test", "food_after_emergency", "medium", "cdc_food_after_emergency", "With doors closed, how long can food stay safe in a half-full freezer?", "Up to 24 hours.", ["Up to 4 hours.", "Up to 72 hours.", "Until it smells normal."], ["24 hours half-full freezer"], ["24 hours", "half-full freezer"], [3], True),
    spec("heldout_test", "test", "food_after_emergency", "hard", "cdc_food_after_emergency", "What should you do if you are unsure whether emergency food is safe?", "Throw it out.", ["Taste a small amount first.", "Rinse it and keep it.", "Trust it if it looks normal."], ["when unsure throw it out"], ["unsure", "throw it out"], [0, 2], True),
    spec("heldout_test", "test", "food_after_emergency", "hard", "cdc_food_after_emergency", "Why is food that contacted floodwater or storm water risky?", "It may be unsafe to eat after that contact.", ["Floodwater contact makes sealed-looking food safer.", "Storm water contact is a cleaning step.", "Only salt water can contaminate food."], ["food contact with floodwater may be unsafe"], ["floodwater", "unsafe to eat"], [0, 1], True),
    spec("heldout_test", "test", "emergency_water_supply", "medium", "cdc_emergency_water_supply", "How much emergency water should be stored per person per day?", "At least 1 gallon per person per day.", ["One cup per person per day.", "One gallon for the whole family per week.", "Only bottled juice is needed."], ["1 gallon per person per day"], ["1 gallon", "per person"], [], True),
    spec("heldout_test", "test", "emergency_water_supply", "medium", "cdc_emergency_water_supply", "How many days of water does the CDC emergency supply page say to create and store?", "A 3-day supply.", ["A 3-hour supply.", "A 30-day minimum for everyone.", "No stored water is recommended."], ["3-day supply"], ["3-day supply"], [], True),
    spec("heldout_test", "test", "emergency_water_supply", "hard", "cdc_emergency_water_supply", "What is the safest and most reliable emergency drinking-water source named by CDC?", "Unopened commercially bottled water.", ["Water from a decorative fountain.", "Floodwater filtered through a shirt.", "Any clear-looking standing water."], ["unopened commercially bottled water"], ["Unopened", "bottled water"], [1, 2], True),
    spec("heldout_test", "test", "emergency_water_supply", "medium", "cdc_emergency_water_supply", "How often should stored water in filled containers be replaced?", "Every 6 months.", ["Every 6 years.", "Only after it smells bad.", "Never if the container is sealed."], ["replace stored water every 6 months"], ["replace", "6 months"], [], True),
    spec("heldout_test", "test", "emergency_water_supply", "hard", "cdc_emergency_water_supply", "What bleach concentration range does the CDC storage page mention for emergency water disinfection supplies?", "Unscented household chlorine bleach with 5 to 9 percent sodium hypochlorite.", ["Scented bleach with added cleaners.", "Any powdered detergent.", "Vinegar mixed with ammonia."], ["unscented bleach", "5 to 9 percent sodium hypochlorite"], ["5%", "9%", "sodium hypochlorite"], [0, 3], True),
    spec("heldout_test", "test", "water_disinfection", "hard", "epa_emergency_disinfection", "What limitation does EPA give for boiling or disinfecting emergency drinking water?", "It kills most microbes but will not destroy heavy metals, salts, and most chemicals.", ["It removes every chemical contaminant.", "It makes salty water fresh.", "It removes all heavy metals."], ["boiling/disinfection does not destroy heavy metals salts chemicals"], ["heavy metals", "salts", "chemicals"], [0, 1, 2], True),
    spec("heldout_test", "test", "water_disinfection", "medium", "epa_emergency_disinfection", "How long should water be at a rolling boil at normal altitudes according to EPA?", "At least 1 minute.", ["At least 5 seconds.", "At least 30 minutes.", "No boiling is needed if water is cloudy."], ["rolling boil at least 1 minute"], ["rolling boil", "one minute"], [3], True),
    spec("heldout_test", "test", "water_disinfection", "hard", "epa_emergency_disinfection", "At altitudes above 5,000 feet, how long should emergency water boil?", "For 3 minutes.", ["For 30 seconds.", "For 1 minute exactly.", "For 30 minutes."], ["above 5000 feet boil 3 minutes"], ["5,000 feet", "three minutes"], [], True),
    spec("heldout_test", "test", "water_disinfection", "medium", "epa_emergency_disinfection", "What should happen before boiling cloudy water?", "Let it settle and filter it through a clean cloth, paper towel, or coffee filter.", ["Add fuel to separate particles.", "Drink it if particles sink.", "Skip boiling because cloudy water is safer."], ["settle and filter cloudy water"], ["cloudy", "settle", "coffee filter"], [1, 2], True),
    spec("heldout_test", "test", "volcano_ash", "medium", "epa_volcanoes", "Which volcanic gases does EPA identify as major potential hazards?", "Sulfur dioxide, carbon dioxide, and hydrogen fluoride.", ["Oxygen, nitrogen, and helium.", "Steam only.", "Methane only."], ["sulfur dioxide carbon dioxide hydrogen fluoride"], ["sulfur dioxide", "hydrogen fluoride"], [], True),
    spec("heldout_test", "test", "volcano_ash", "medium", "epa_volcanoes", "How far can volcanic ash travel downwind according to EPA?", "Hundreds to thousands of miles.", ["Only a few feet.", "Only inside the crater.", "It cannot travel downwind."], ["ash can travel hundreds to thousands of miles"], ["hundreds", "thousands of miles"], [], False),
    spec("heldout_test", "test", "volcano_ash", "hard", "epa_volcanoes", "Who can volcanic ash particularly trouble even though it is not highly toxic?", "Infants, the elderly, and people with respiratory ailments.", ["Only trained volcanologists.", "Only healthy adults outdoors.", "Only people near seawater."], ["ash troubles infants elderly respiratory ailments"], ["infants", "elderly", "respiratory"], [], True),
    spec("heldout_test", "test", "volcano_ash", "hard", "epa_volcanoes", "What can volcanic ash do to drinking water and wastewater facilities?", "Clog or damage equipment and force shutdowns.", ["Sterilize equipment automatically.", "Improve filtration performance.", "Remove all contaminants from water."], ["ash can clog or damage water facility equipment"], ["drinking water", "wastewater", "clogging"], [], True),
    spec("heldout_test", "test", "winter_storm", "medium", "nws_winter", "What conditions define a blizzard in the NWS winter safety source?", "Strong wind causing blowing snow and whiteout conditions.", ["Any cold rain event.", "Any snowfall under calm wind.", "A clear but cold day."], ["strong wind", "blowing snow", "whiteout"], ["Blizzards", "strong wind", "whiteout"], [], True),
    spec("heldout_test", "test", "winter_storm", "medium", "nws_winter", "Why are blizzards dangerous for roads according to NWS?", "They can make roads impassable.", ["They melt ice instantly.", "They improve road visibility.", "They only affect sidewalks."], ["roads impassable"], ["roads impassable"], [], True),
    spec("heldout_test", "test", "winter_storm", "medium", "nws_winter", "What winter-storm road hazard injures or kills thousands each year according to NWS?", "Traffic accidents related to slippery roads.", ["Sunburn during snow melt.", "Only lightning strikes.", "Only house fires."], ["slippery road traffic accidents"], ["traffic accidents", "slippery roads"], [], True),
    spec("heldout_test", "test", "lightning", "medium", "ready_lightning", "When thunder roars, what does the Ready lightning source say to do?", "Go indoors.", ["Stand under a lone tree.", "Move into open water.", "Hold a metal pole outside."], ["go indoors when thunder roars"], ["thunder roars", "go indoors"], [0, 1, 2], True),
    spec("heldout_test", "test", "lightning", "medium", "ready_lightning", "What is the safest place to be during a thunderstorm in the Ready source?", "A sturdy building.", ["An open field.", "A hilltop under a tree.", "A shallow stream."], ["sturdy building safest"], ["sturdy building", "safest place"], [0, 1, 2], True),
    spec("heldout_test", "test", "lightning", "hard", "ready_lightning", "Why should people avoid running water and landline phones indoors during lightning?", "Electricity can travel through plumbing and phone lines.", ["They attract floodwater indoors.", "They make thunder louder.", "They prevent alerts from arriving."], ["electricity travels through plumbing and phone lines"], ["plumbing", "phone lines"], [], True),
    spec("heldout_test", "test", "lightning", "medium", "ready_lightning", "Which alerts are named as sources of emergency thunderstorm warnings?", "Community warning systems, EAS, and NOAA Weather Radio.", ["Restaurant menus and traffic lights.", "Only social gossip.", "Only handwritten posters."], ["EAS and NOAA Weather Radio"], ["Emergency Alert System", "Weather Radio"], [], False),
    spec("heldout_test", "test", "flood_preparedness", "medium", "cdc_flood_preparedness_spanish", "What should a family practice before a flood according to the CDC flood-preparedness page?", "A family evacuation route.", ["A route toward floodwater.", "A plan to ignore evacuation orders.", "A plan to split up without contacts."], ["practice flood evacuation route"], ["evacu", "familia"], [0, 1, 2], True),
    spec("heldout_test", "test", "flood_preparedness", "medium", "cdc_flood_preparedness_spanish", "What does the CDC flood page say about evacuation orders?", "Never ignore an evacuation order.", ["Ignore it if roads look familiar.", "Wait until floodwater reaches the door.", "Evacuate only after losing phone service."], ["never ignore evacuation order"], ["ignore", "evacu"], [0, 1], True),
    spec("heldout_test", "test", "flood_preparedness", "medium", "cdc_flood_preparedness_spanish", "Whom should a family choose as a contact if separated during a flood?", "An out-of-state family member or friend.", ["Only someone in the flooded neighborhood.", "No contact person is needed.", "A random social media account."], ["out-of-state family contact"], ["estado", "contacto"], [], False),
    # Held-out test: response management, firefighting, FEMA courses, PFA.
    spec("heldout_test", "test", "firefighting_response", "medium", "fema_nrf_esf4", "What is the purpose of ESF #4 in the NRF annex?", "Federal support for detection and suppression of wildland, rural, and urban fires.", ["Federal support for grocery pricing.", "Only post-disaster tax filing.", "Only hospital credentialing."], ["ESF 4 detection and suppression of fires"], ["detection and suppression", "wildland"], [], False),
    spec("heldout_test", "test", "firefighting_response", "medium", "fema_nrf_esf4", "Which agency is listed as the primary agency for ESF #4 Firefighting?", "Department of Agriculture/Forest Service.", ["Department of Treasury.", "Department of Education.", "Food and Drug Administration."], ["Forest Service primary agency"], ["Primary Agency", "Forest Service"], [], False),
    spec("heldout_test", "test", "firefighting_response", "hard", "fema_nrf_esf4", "What kind of firefighting resources does ESF #4 coordinate in support of local and state agencies?", "Personnel, equipment, and supplies.", ["Weather rumors, donations, and insurance rates.", "Court filings, ballots, and school records.", "Only public announcements."], ["personnel equipment supplies"], ["personnel", "equipment", "supplies"], [], False),
    spec("heldout_test", "test", "firefighting_response", "hard", "fema_nrf_esf4", "How does the annex describe firefighting responsibility at local, state, tribal, and territorial levels?", "It is inherently a local responsibility.", ["It is never local.", "It belongs only to foreign governments.", "It is handled only by private insurers."], ["firefighting is inherently local responsibility"], ["inherently local responsibility"], [], False),
    spec("heldout_test", "test", "firefighting_response", "hard", "fema_nrf_esf4", "Under ESF #4, what does the National Weather Service provide in support of fire planning and response?", "Fire/weather support, forecasts, and smoke dispersion forecasts.", ["Legal rulings on arson cases.", "Food inspection reports.", "School attendance records."], ["fire/weather support", "smoke dispersion forecasts"], ["fire/weather", "smoke"], [], False),
    spec("heldout_test", "test", "firefighting_response", "hard", "fema_nrf_esf4", "What does DOD do for firefighting on DOD installations according to the annex?", "Assumes full responsibility for firefighting activities on DOD installations.", ["Transfers all DOD fires to local schools.", "Provides only press releases.", "Does not fight fires on its own installations."], ["DOD assumes full responsibility on installations"], ["DOD installations", "full responsibility"], [], False),
    spec("heldout_test", "test", "fema_training", "medium", "fema_is100c", "What does IS-100.c introduce?", "The Incident Command System and foundational ICS knowledge.", ["Volcanic ash chemistry.", "Food label nutrition facts.", "At-home COVID test expiration dates."], ["IS-100.c introduces ICS"], ["IS-100.c", "Incident Command System"], [], False),
    spec("heldout_test", "test", "fema_training", "medium", "fema_is100c", "What is the Independent Study Program described as?", "An online distance learning program for emergency management and public preparedness.", ["A shelter construction permit system.", "A food recall database.", "A medical device registry."], ["ISP online distance learning"], ["online distance learning", "Independent Study Program"], [], False),
    spec("heldout_test", "test", "fema_training", "hard", "fema_is100c", "What is the goal of IS-251.b for IPAWS Alerting Administrators?", "Guidance on policies, approvals, training, exercises, and effective use of IPAWS to reach the public.", ["Training on cooking after outages.", "Guidance on insulin switching.", "Training on volcanic ash cleanup only."], ["IPAWS administrator guidance"], ["IPAWS", "Alerting Administrators"], [], False),
    spec("heldout_test", "test", "fema_training", "hard", "fema_is100c", "What does IS-393.b define mitigation as?", "Taking action to reduce or eliminate long-term risk from hazards and their effects.", ["Making alerts more dramatic.", "Waiting until recovery to plan.", "Increasing long-term hazard risk."], ["mitigation reduces long-term risk"], ["Mitigation means", "long-term risk"], [], False),
    spec("heldout_test", "test", "fema_training", "hard", "fema_is100c", "What does IS-870.b for the Dams Sector address?", "Crisis management activities and plans for preparedness, protection, recovery, and resilience.", ["Only household bleach storage.", "Only winter weather forecasts.", "Only school lunch programs."], ["Dams Sector crisis management"], ["IS-870.b", "Crisis Management"], [], False),
    spec("heldout_test", "test", "fema_training", "hard", "fema_is100c", "What are the last four ISC Risk Management Process steps covered by IS-1173.a?", "Identify and assess risk, develop and implement a risk strategy, and measure performance.", ["Boil, cool, bottle, and label water.", "Warn, evacuate, shelter, and recover pets.", "Taste, smell, rinse, and freeze food."], ["identify assess risk develop implement strategy measure performance"], ["IS-1173", "Measure Performance"], [], False),
    spec("heldout_test", "test", "pfa_support", "medium", "va_pfa", "Which behavior interferes with giving support after disaster exposure?", "Rushing to tell someone they will be okay.", ["Listening to concerns first.", "Asking what works for the person.", "Encouraging appropriate social support."], ["rushing reassurance interferes"], ["Rushing", "okay"], [], True),
    spec("heldout_test", "test", "pfa_support", "medium", "va_pfa", "What does the PFA source say about advice without listening?", "Giving advice without listening interferes with support.", ["It is always the best first step.", "It replaces asking about concerns.", "It is required before hearing the story."], ["advice without listening interferes"], ["Giving advice", "without listening"], [], True),
    spec("heldout_test", "test", "pfa_support", "hard", "va_pfa", "When support is not enough, what help does the PFA page suggest?", "Encourage talking with a counselor, clergy, or medical professional and offer to accompany them.", ["Tell them to avoid everyone.", "Tell them experts think withdrawal always helps.", "Tell them to stop talking about distress."], ["encourage counselor clergy medical professional"], ["counselor", "medical professional"], [0, 2], True),
    spec("heldout_test", "test", "pfa_support", "medium", "va_pfa", "What does the PFA page say about avoidance and withdrawal?", "They are likely to increase distress, while social support helps recovery.", ["They are the preferred recovery strategy.", "They should be forced on everyone.", "They prove the person is weak."], ["avoidance withdrawal increase distress", "social support helps recovery"], ["avoidance", "withdrawal", "social support"], [0, 2], True),
    spec("heldout_test", "test", "pfa_children", "medium", "va_pfa", "How should adults respond to a child's confusion about what happened?", "Give clear explanations when asked and avoid frightening details.", ["Refuse all explanations.", "Add frightening details to force seriousness.", "Tell the child they are weak for asking."], ["clear explanations", "avoid frightening details"], ["clear explanations", "Avoid details"], [1, 2], True),
    spec("heldout_test", "test", "pfa_children", "medium", "va_pfa", "What does the child-support handout recommend when a child worries after seeing a parent injured?", "Give chances to talk about feelings and stay as calm as possible.", ["Hide all feelings and forbid questions.", "Criticize the child for worrying.", "Make the child responsible for the adult."], ["talk about feelings", "remain calm"], ["talk about their feelings", "Remain as calm"], [1, 2], True),
    spec("heldout_test", "test", "medical_devices", "hard", "fda_medical_devices", "What is the FDA medical-device recall communications pilot meant to minimize?", "The time between FDA initial awareness and public notification of high-risk removals or corrections.", ["The number of public notifications.", "The need to notify the public.", "The time medical devices remain listed."], ["minimize time to public notification"], ["minimize the time", "public notification"], [], False),
    spec("heldout_test", "test", "medical_devices", "medium", "fda_medical_devices", "What should people check before throwing out at-home COVID-19 tests?", "The FDA website for extended expiration dates.", ["A neighbor's guess.", "The color of the test box only.", "Whether the test feels warm."], ["check FDA website extended expiration dates"], ["throwing out", "extended expiration dates"], [], False),
    spec("heldout_test", "test", "medical_devices", "medium", "fda_medical_devices", "Which FDA database is named for medical device experience reports?", "MAUDE, the Manufacturer and User Facility Device Experience Database.", ["The National Weather Service forecast map.", "A freezer temperature chart.", "A flood insurance rate map."], ["MAUDE database"], ["MAUDE", "Device Experience"], [], False),
    spec("heldout_test", "test", "winter_storm", "medium", "nws_winter", "What types of precipitation can winter storms bring across the United States?", "Snow, sleet, and freezing rain.", ["Only warm rain.", "Only volcanic ash.", "Only hail without snow."], ["snow sleet freezing rain"], ["snow", "sleet", "freezing rain"], [], True),
    spec("heldout_test", "test", "lightning", "medium", "ready_lightning", "What should people do with appliances during a thunderstorm according to Ready.gov?", "Unplug appliances.", ["Plug in extra appliances.", "Run water over appliances.", "Move appliances outside."], ["unplug appliances"], ["Unplug appliances"], [0, 2], True),
    spec("heldout_test", "test", "emergency_water_supply", "medium", "cdc_emergency_water_supply", "When should households store more than the baseline water amount?", "For pregnant women, sick people, pets, or hot climates.", ["When water containers held pesticides.", "Only when floodwater looks cloudy.", "Never; the baseline cannot be adjusted."], ["store more water for vulnerable needs and heat"], ["pregnant women", "pets", "hot climate"], [], True),
    # Train-seen probes.
    spec("train_seen_probe", "train", "water_disinfection", "hard", "cdc_emergency_water", "Why is boiling not enough for emergency water that smells like fuel or chemicals?", "Boiling does not make fuel, toxic chemical, or radioactive contamination safe.", ["Boiling longer removes fuel safely.", "Cloudiness proves fuel has settled out.", "Adding salt neutralizes toxic chemicals."], ["boiling does not fix chemical contamination"], ["fuel", "toxic chemicals", "boiling"], [0, 1], True),
    spec("train_seen_probe", "train", "water_disinfection", "medium", "cdc_emergency_water", "What should be done before boiling cloudy emergency water?", "Filter it through a clean cloth, paper towel, or coffee filter, or let it settle.", ["Boil it without settling.", "Add gasoline to separate sediment.", "Drink it if the top looks clear."], ["filter or settle cloudy water"], ["cloudy", "filter", "boil"], [0, 2], True),
    spec("train_seen_probe", "train", "water_disinfection", "hard", "cdc_emergency_water", "After adding bleach to disinfect emergency water, how long should it stand before drinking?", "At least 30 minutes.", ["Exactly 3 minutes.", "Drink immediately.", "At least 24 hours."], ["wait 30 minutes"], ["bleach", "30 minutes"], [1], True),
    spec("train_seen_probe", "train", "food_after_flood", "hard", "cdc_food_after_emergency", "Which flood-contact items should be thrown out because sanitizing is not effective?", "Wooden cutting boards, baby bottle nipples, and pacifiers.", ["Metal pans with intact surfaces.", "Factory-sealed bottled water outside the flood zone.", "Unopened food never touched by floodwater."], ["discard porous baby items and wooden boards"], ["wooden cutting boards", "baby bottle nipples", "pacifiers"], [], True),
    spec("train_seen_probe", "train", "food_after_flood", "medium", "cdc_food_after_emergency", "Why is smell not enough to decide if emergency food is safe?", "Unsafe food can look and smell normal.", ["Bad food always smells rotten.", "Safe food always smells like bleach.", "Smell is more reliable than temperature."], ["unsafe food can look and smell normal"], ["looks and smells normal", "Unsafe food"], [0, 2], True),
    spec("train_seen_probe", "train", "diabetes_emergency", "medium", "cdc_diabetes_emergencies", "How long should diabetes emergency supplies last according to the CDC emergency kit source?", "At least 1 to 2 weeks.", ["Only one meal.", "Exactly 24 hours.", "Until the next routine appointment only."], ["1 to 2 weeks of diabetes supplies"], ["1 to 2 weeks", "diabetes supplies"], [], True),
    spec("train_seen_probe", "train", "diabetes_emergency", "medium", "cdc_diabetes_emergencies", "What quick-carb amount does the diabetes emergency source name for low blood sugar?", "15 grams of quick carbs.", ["1 gram of quick carbs.", "100 grams of protein.", "No quick carbs are recommended."], ["15 grams quick carbs"], ["15 grams", "quick carbs"], [], True),
    spec("train_seen_probe", "train", "insulin_emergency", "hard", "cdc_insulin_emergency", "What is the key storage warning for insulin during an outage?", "Keep insulin cool, but do not freeze it.", ["Freeze insulin solid to preserve it.", "Leave insulin in direct sunlight.", "Heat insulin before use."], ["keep insulin cool but not frozen"], ["insulin cool", "freeze"], [0, 1, 2], True),
    spec("train_seen_probe", "train", "risk_communication", "medium", "cdc_cerc", "What are the six CERC principles emphasized in the CDC material?", "Be first, be right, be credible, express empathy, promote action, and show respect.", ["Be fast, be vague, hide gaps, and wait.", "Deny uncertainty and avoid action steps.", "Use fear, rumor, blame, and delay."], ["six CERC principles"], ["Be First", "Be Right", "Be Credible"], [], False),
    spec("train_seen_probe", "train", "cyclone_evacuation", "medium", "ndma_cyclone", "In cyclone preparedness, why is coastal evacuation planning important?", "Storm surge, wind, and flooding can make coastal areas unsafe and require organized movement to safer shelters.", ["Coastal residents should wait at the shoreline.", "Evacuation planning is only for earthquakes.", "Shelters should be chosen after roads flood."], ["cyclone coastal evacuation and shelters"], ["cyclone", "evacuation", "shelter"], [0, 2], True),
    # Dev calibration.
    spec("dev_calibration", "dev", "safe_room", "medium", "ready_safe_rooms_shelters", "What is FEMA 453 intended to guide?", "Design of shelters and safe rooms in buildings.", ["Food recall notices.", "COHgb measurement.", "Winter storm naming."], ["safe room shelter design guidance"], ["design shelters", "safe rooms"], [], False),
    spec("dev_calibration", "dev", "safe_room", "hard", "ready_safe_rooms_shelters", "Why does the safe-room document avoid defining a single explosive design threat?", "Threat distance and intervening structure are site-specific and hard to generalize.", ["Explosive threats never vary by site.", "All buildings use the same threat distance.", "The document is only about food safety."], ["site-specific threat distance and structure"], ["site-specific", "design level threat"], [], False),
    spec("dev_calibration", "dev", "protection_framework", "medium", "fema_protection_framework", "What mission does the National Protection Framework describe?", "How the whole community safeguards against terrorism, natural disasters, and other threats or hazards.", ["How families disinfect drinking water.", "How clinicians test COHgb.", "How freezers keep food cold."], ["whole community safeguards threats hazards"], ["whole community", "threats or hazards"], [], False),
    spec("dev_calibration", "dev", "protection_framework", "hard", "fema_protection_framework", "How does the Protection Framework describe resilience?", "The ability to prepare for, adapt to, withstand, and recover rapidly from disruptions.", ["The ability to ignore warnings.", "The ability to remove all uncertainty.", "The ability to avoid partnerships."], ["resilience definition"], ["Resilience is the ability", "recover rapidly"], [], False),
    spec("dev_calibration", "dev", "protection_framework", "hard", "fema_protection_framework", "Which three core capabilities span all five mission areas in the Protection Framework excerpt?", "Planning, Public Information and Warning, and Operational Coordination.", ["Water, Food, and Shelter.", "Diagnosis, Treatment, and Follow-up.", "Fire, Police, and Schools."], ["three cross-cutting core capabilities"], ["Planning", "Public Information and Warning", "Operational Coordination"], [], False),
    spec("dev_calibration", "dev", "protection_framework", "hard", "fema_protection_framework", "What lifeline functions are named in the protection critical tasks excerpt?", "Energy, communications, transportation, and water and wastewater management.", ["Sports, entertainment, tourism, and retail.", "Only schools and parks.", "Only hospitals and pharmacies."], ["critical lifeline functions"], ["energy", "communications", "transportation", "water"], [], False),
    spec("dev_calibration", "dev", "flood_messaging", "medium", "ready_floods_toolkit", "What does the flood messaging toolkit discourage changing?", "Facts or data in posts, to ensure accuracy.", ["Local dates and locations only.", "The platform name only.", "The font size only."], ["do not adjust facts or data"], ["facts or data", "accuracy"], [], False),
    spec("dev_calibration", "dev", "flood_messaging", "medium", "ready_floods_toolkit", "What should additions to flood messaging focus on?", "State-specific statistics, concerns, or risks.", ["Invented damage estimates.", "Unverified rescue claims.", "Personal rumors."], ["state-specific statistics concerns risks"], ["state-specific", "statistics"], [1, 2], True),
    spec("dev_calibration", "dev", "pfa_support", "medium", "va_pfa", "Which PFA behavior interferes with giving support?", "Acting like someone is weak or exaggerating.", ["Listening to the person's concerns.", "Asking what works for them.", "Offering appropriate accompaniment."], ["acting weak exaggerating interferes"], ["weak or exaggerating"], [], True),
    spec("dev_calibration", "dev", "pfa_children", "medium", "va_pfa", "For a child afraid to sleep alone after disaster, what is one recommended bedtime support?", "Spend extra quiet time together and use calming activities before bed.", ["Shame the child for being afraid.", "Forbid any discussion of fear.", "Tell the child dreams are real danger."], ["calming bedtime support"], ["extra quiet time", "calming activities"], [0, 1, 2], True),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(stem: str, choices: list[dict[str, str]]) -> str:
    payload = {"question_stem": stem, "choices": choices}
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_cpt() -> dict[str, list[dict[str, Any]]]:
    return {split: read_jsonl(CPT_DIR / f"cpt_{split}.jsonl") for split in ["train", "dev", "test"]}


def find_evidence(item: MCQSpec, cpt_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_score = -1
    for row in cpt_rows[item.source_split]:
        haystack = normalize(" ".join([row.get("document_id", ""), row.get("chunk_id", ""), row.get("title", ""), row.get("text", "")]))
        score = sum(1 for term in item.evidence_terms if term.lower() in haystack)
        source_bonus = 2 if item.source_family.lower().split("_")[0] in haystack else 0
        if score + source_bonus > best_score:
            best = row
            best_score = score + source_bonus
        if score == len(item.evidence_terms):
            return row
    missing = [term for term in item.evidence_terms if best and term.lower() not in normalize(best.get("text", ""))]
    raise RuntimeError(f"No exact evidence for {item.question_stem!r}; best={best and best.get('chunk_id')} missing={missing}")


def evidence_span(text: str, terms: tuple[str, ...], window: int = 560) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    lower = flat.lower()
    positions = [lower.find(term.lower()) for term in terms if lower.find(term.lower()) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - window // 4)
    end = min(len(flat), center + window)
    return flat[start:end]


def labeled_choices(item: MCQSpec, index: int) -> tuple[list[dict[str, str]], str, dict[str, str], list[str]]:
    labels = ["A", "B", "C", "D"]
    correct_label = labels[index % 4]
    choices_by_label: dict[str, str] = {}
    distractor_iter = iter(item.distractors)
    for label in labels:
        choices_by_label[label] = item.correct if label == correct_label else next(distractor_iter)
    choices = [{"label": label, "text": choices_by_label[label]} for label in labels]
    distractor_rationales = {
        label: ("unsafe misconception" if position in item.unsafe_distractor_indices else "plausible but contradicted or unsupported distractor")
        for position, label in enumerate([label for label in labels if label != correct_label])
    }
    unsafe_labels = [label for position, label in enumerate([label for label in labels if label != correct_label]) if position in item.unsafe_distractor_indices]
    return choices, correct_label, distractor_rationales, unsafe_labels


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cpt_rows = load_cpt()
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    source_cards: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(SPECS):
        evidence = find_evidence(item, cpt_rows)
        choices, correct_label, distractor_rationales, unsafe_labels = labeled_choices(item, index)
        example_id = f"beacon_mcq_knowledge_v1_{index:03d}"
        public_hash = content_hash(item.question_stem, choices)
        correct_text = next(choice["text"] for choice in choices if choice["label"] == correct_label)
        public_rows.append(
            {
                "example_id": example_id,
                "schema_version": SCHEMA_PUBLIC,
                "eval_bucket": item.eval_bucket,
                "hazard_bucket": item.hazard_bucket,
                "difficulty": item.difficulty,
                "cpt_exposure": item.cpt_exposure,
                "question_stem": item.question_stem,
                "choices": choices,
                "prompt_template_id": PROMPT_TEMPLATE_ID,
                "public_content_hash": public_hash,
            }
        )
        key_rows.append(
            {
                "example_id": example_id,
                "schema_version": SCHEMA_KEY,
                "correct_label": correct_label,
                "correct_choice_hash": sha256_text(correct_text),
                "source_family": item.source_family,
                "evidence_doc_ids": [evidence["document_id"]],
                "evidence_chunk_ids": [evidence["chunk_id"]],
                "source_urls": [evidence.get("url", "")],
                "required_facts": list(item.required_facts),
                "evidence_spans": [evidence_span(evidence["text"], item.evidence_terms)],
                "distractor_rationales": distractor_rationales,
                "unsafe_distractor_labels": unsafe_labels,
                "critical_safety_subset": item.critical_safety_subset,
                "review_status": "approved",
            }
        )
        source_cards[evidence["document_id"]] = {
            "document_id": evidence["document_id"],
            "source_id": evidence.get("source_id", ""),
            "organization": evidence.get("organization", ""),
            "title": evidence.get("title", ""),
            "url": evidence.get("url", ""),
            "language": evidence.get("language", ""),
            "hazards": evidence.get("hazards", []),
            "cpt_split": evidence.get("split", ""),
        }
    return public_rows, key_rows, sorted(source_cards.values(), key=lambda row: row["document_id"])


def validate(public_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    labels = {"A", "B", "C", "D"}
    if len(public_rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} public rows, got {len(public_rows)}")
    if len(key_rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} key rows, got {len(key_rows)}")
    public_ids = [row["example_id"] for row in public_rows]
    key_ids = [row["example_id"] for row in key_rows]
    if len(set(public_ids)) != len(public_ids):
        errors.append("duplicate public example_id")
    if set(public_ids) != set(key_ids):
        errors.append("public/key example_id mismatch")
    key_by_id = {row["example_id"]: row for row in key_rows}
    label_counts = Counter(row["correct_label"] for row in key_rows)
    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        errors.append(f"label counts mismatch: {dict(label_counts)}")
    stem_counts = Counter(row["question_stem"] for row in public_rows)
    duplicates = [stem for stem, count in stem_counts.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate question stems: {duplicates[:5]}")
    choice_signatures = Counter(json.dumps(row["choices"], sort_keys=True) for row in public_rows)
    if any(count > 1 for count in choice_signatures.values()):
        errors.append("duplicate full choice set")
    forbidden_public_fields = {
        "correct_label",
        "correct_choice_hash",
        "evidence_doc_ids",
        "evidence_chunk_ids",
        "source_urls",
        "required_facts",
        "evidence_spans",
        "distractor_rationales",
        "unsafe_distractor_labels",
        "critical_safety_subset",
        "review_status",
    }
    forbidden_phrases = {"all of the above", "none of the above"}
    for row in public_rows:
        overlap = forbidden_public_fields.intersection(row)
        if overlap:
            errors.append(f"{row['example_id']} public leakage fields: {sorted(overlap)}")
        choices = row.get("choices", [])
        if {choice.get("label") for choice in choices} != labels or len(choices) != 4:
            errors.append(f"{row['example_id']} choices must be A-D exactly once")
        if row["public_content_hash"] != content_hash(row["question_stem"], choices):
            errors.append(f"{row['example_id']} public_content_hash mismatch")
        if any(any(phrase in choice["text"].lower() for phrase in forbidden_phrases) for choice in choices):
            errors.append(f"{row['example_id']} forbidden choice phrase")
        key = key_by_id[row["example_id"]]
        correct_text = next(choice["text"] for choice in choices if choice["label"] == key["correct_label"])
        if sha256_text(correct_text) != key["correct_choice_hash"]:
            errors.append(f"{row['example_id']} correct_choice_hash mismatch")
        if key["correct_label"] in set(key.get("unsafe_distractor_labels", [])):
            errors.append(f"{row['example_id']} correct label marked unsafe")
    bucket_counts = Counter(row["eval_bucket"] for row in public_rows)
    expected_buckets = {"heldout_test": 60, "train_seen_probe": 10, "dev_calibration": 10}
    if dict(bucket_counts) != expected_buckets:
        errors.append(f"bucket counts mismatch: {dict(bucket_counts)}")
    critical_count = sum(1 for row in key_rows if row["critical_safety_subset"])
    if critical_count < 24:
        errors.append(f"expected at least 24 critical safety rows, got {critical_count}")
    return errors


def manifest(public_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]], source_cards: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    counts = {
        "total": len(public_rows),
        "by_eval_bucket": dict(Counter(row["eval_bucket"] for row in public_rows)),
        "by_cpt_exposure": dict(Counter(row["cpt_exposure"] for row in public_rows)),
        "by_hazard_bucket": dict(Counter(row["hazard_bucket"] for row in public_rows)),
        "by_difficulty": dict(Counter(row["difficulty"] for row in public_rows)),
        "by_correct_label": dict(Counter(row["correct_label"] for row in key_rows)),
        "critical_safety_subset": sum(1 for row in key_rows if row["critical_safety_subset"]),
        "source_documents": len(source_cards),
    }
    by_doc: dict[str, int] = defaultdict(int)
    for row in key_rows:
        by_doc[row["evidence_doc_ids"][0]] += 1
    return {
        "schema_version": "beacon-mcq-knowledge-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "counts": counts,
        "questions_public": "questions_public.jsonl",
        "answer_key_private": "answer_key_private.jsonl",
        "source_cards_private": "source_cards_private.jsonl",
        "split_policy": {
            "heldout_test": "60 MCQs sourced only from cpt_test.jsonl; headline metric.",
            "train_seen_probe": "10 MCQs sourced from cpt_train.jsonl; diagnostic recall probe.",
            "dev_calibration": "10 MCQs sourced from cpt_dev.jsonl; prompt/scoring sanity only.",
        },
        "scoring_policy": {
            "prompt": "Answer with exactly one letter: A, B, C, or D.",
            "invalid_outputs_score": 0,
            "headline_metric": "heldout_test_accuracy",
            "overall_80_accuracy": "mixed diagnostic only",
        },
        "source_document_question_counts": dict(sorted(by_doc.items())),
        "hashes": {
            "questions_public.jsonl": sha256_file(OUT_DIR / "questions_public.jsonl"),
            "answer_key_private.jsonl": sha256_file(OUT_DIR / "answer_key_private.jsonl"),
            "source_cards_private.jsonl": sha256_file(OUT_DIR / "source_cards_private.jsonl"),
        },
    }


def main() -> None:
    public_rows, key_rows, source_cards = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "questions_public.jsonl", public_rows)
    write_jsonl(OUT_DIR / "answer_key_private.jsonl", key_rows)
    write_jsonl(OUT_DIR / "source_cards_private.jsonl", source_cards)
    errors = validate(public_rows, key_rows)
    write_json(OUT_DIR / "eval_manifest.json", manifest(public_rows, key_rows, source_cards, errors))
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors))
    print(f"Wrote {len(public_rows)} MCQs to {OUT_DIR}")


if __name__ == "__main__":
    main()
