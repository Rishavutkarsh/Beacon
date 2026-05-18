# Beacon Source Research v1

This is a source-selection research pack for a future local grounding tool. It does not create grounding cards and should not be treated as a training approval.

## Recommendation

- Status: `valid`
- Candidate sources reviewed: 46
- Accepted for grounding research: 31
- Clean downloaded/extracted accepted documents: 31
- Coverage gaps: none

Proceed to compact card creation only after human review of `candidate_sources.jsonl`, especially the deferred India-specific gaps.

## Core Sources

- `cdc_power_outage` - What to Do to Protect Yourself During a Power Outage (Centers for Disease Control and Prevention); hazards: power_outage, carbon_monoxide, food, medicine, heat_cold
- `cdc_floodwater_safety` - Safety Guidelines: Floodwater (Centers for Disease Control and Prevention); hazards: floodwater, wounds, electrical, contamination
- `cdc_reenter_flooded_home` - Reentering Your Flooded Home Safely (Centers for Disease Control and Prevention); hazards: floodwater, structural, electrical, mold
- `epa_flood_cleanup_iaq` - Resources for Flood Cleanup and Indoor Air Quality (US Environmental Protection Agency); hazards: flood_cleanup, mold, indoor_air, shelter_hygiene
- `fda_food_water_floods` - Food and Water Safety During Power Outages and Floods (US Food and Drug Administration); hazards: food_safety, water_safety, floodwater, power_outage
- `cdc_co_clinical_disasters` - Clinical Guidance for Carbon Monoxide Poisoning Following Disasters and Severe Weather (Centers for Disease Control and Prevention); hazards: carbon_monoxide, generators, symptoms
- `cdc_diabetes_emergencies` - Diabetes Care During Emergencies (Centers for Disease Control and Prevention); hazards: diabetes, medicine_disruption, emergency_kit
- `cdc_emergency_water` - About Drinking Water Emergencies (Centers for Disease Control and Prevention); hazards: water_safety, disinfection, boil_water
- `cdc_food_after_emergency` - Keep Food Safe After a Disaster or Emergency (Centers for Disease Control and Prevention); hazards: food_safety, floodwater, power_outage
- `cdc_insulin_emergency` - Managing Insulin in an Emergency (Centers for Disease Control and Prevention); hazards: diabetes, insulin, medicine_disruption
- `epa_emergency_disinfection` - Emergency Disinfection of Drinking Water (US Environmental Protection Agency); hazards: water_safety, disinfection, chemical_contamination
- `fda_drugs_disaster` - Safe Drug Use After a Natural Disaster (US Food and Drug Administration); hazards: medicine_disruption, floodwater, drug_safety
- `ndma_cyclone` - NDMA Cyclone (National Disaster Management Authority, India); hazards: cyclone, preparedness
- `ready_floods` - Floods (Ready.gov / FEMA); hazards: flood, route_safety, preparedness
- `ready_landslides` - Landslides and Debris Flow (Ready.gov / FEMA); hazards: landslide, structural, route_safety
- `ready_power_outages` - Power Outages (Ready.gov / FEMA); hazards: power_outage, preparedness, carbon_monoxide
- `ready_wildfires` - Wildfires (Ready.gov / FEMA); hazards: fire, smoke, evacuation
- `ready_winter` - Winter Weather (Ready.gov / FEMA); hazards: cold_wave, winter_storm, power_outage
- `nws_turn_around` - Turn Around Don't Drown (US National Weather Service); hazards: flood, route_safety
- `ready_heat` - Extreme Heat (Ready.gov / FEMA); hazards: heatwave, vulnerable_people

## Supporting Sources

- `nws_heat` - Heat Safety (US National Weather Service); hazards: heatwave
- `unicef_wash_emergencies` - WASH in emergencies (UNICEF); hazards: wash, children, emergencies
- `who_diarrhoea` - Diarrhoea (World Health Organization); hazards: diarrhoea, ors, dehydration
- `who_wash_emergencies` - Environmental health in emergencies (World Health Organization); hazards: wash, water_safety, sanitation
- `ndma_heat_wave` - NDMA Heat Wave (National Disaster Management Authority, India); hazards: heatwave
- `who_risk_comm` - Risk communications (World Health Organization); hazards: risk_communication, misinformation
- `imd_weather_warnings` - IMD Weather Portal (India Meteorological Department); hazards: weather_alerts, live_fact_uncertainty
- `nws_flood_safety` - Flood Safety (US National Weather Service); hazards: flood, route_safety
- `nws_lightning` - Lightning Safety (US National Weather Service); hazards: lightning, storm
- `nws_winter` - Winter Weather Safety (US National Weather Service); hazards: cold_wave, winter_storm

## Coverage Matrix Summary

- `flood_route`: covered (9 accepted, 0 India)
- `water_wash`: covered (5 accepted, 0 India)
- `food_safety`: covered (5 accepted, 0 India)
- `power_co_electrical`: covered (8 accepted, 0 India)
- `medicine_diabetes`: covered (4 accepted, 0 India)
- `wounds_cleanup`: covered (3 accepted, 0 India)
- `shelter_vulnerable`: covered (4 accepted, 1 India)
- `cyclone_coastal`: covered (2 accepted, 2 India)
- `landslide_structural`: covered (2 accepted, 0 India)
- `heat_cold_lightning`: covered (6 accepted, 1 India)
- `misinformation_live_status`: covered (2 accepted, 1 India)

## Deferred Or Rejected Sources To Review

- `ndma_earthquake_guidance_candidate` - Important India gap; add once a stable downloadable page/PDF is identified and extracted cleanly.
- `sachet_dos_donts_metadata` - Useful India official source, but current portal extraction may expose app/runtime artifacts; keep as candidate until clean export is available.
- `mohfw_heat_public_health_candidate` - Potentially better India public-health authority for heat illness; needs specific stable document.
- `ifrc_public_awareness_candidate` - High-quality NGO candidate; prior automated fetch was blocked.
- `sphere_handbook_candidate` - Excellent standard for WASH/shelter, but do not store or distill until licensing and extraction are reviewed.
- `fssai_food_emergency_candidate` - Potential India-specific food-safety source; needs stable emergency/disaster page.
- `ndma_dos_donts` - sachet_static_capture_contains_embedded_browser_api_key
- `ndma_flood_guidelines_pdf` - HTTPError: 404 Client Error: Not Found for url: https://ndma.gov.in/images/guidelines/flood.pdf
- `ndma_floods` - HTTPError: 404 Client Error: Not Found for url: https://ndma.gov.in/Natural-Hazards/Floods
- `ndma_landslide` - HTTPError: 404 Client Error: Not Found for url: https://ndma.gov.in/Natural-Hazards/Landslide
- `ndma_heatwave_guidelines_pdf` - known_bad_pdf_extraction_too_short
- `sphere_handbook` - HTTPError: 403 Client Error: Forbidden for url: https://spherestandards.org/handbook/
- `ifrc_public_awareness` - HTTPError: 403 Client Error: Forbidden for url: https://www.ifrc.org/document/public-awareness-and-public-education-disaster-risk-reduction
- `usgs_landslide_signs` - ValueError: empty_extracted_text
- `nidm_ndma_heatwave_pdf` - known_bad_pdf_extraction_too_short

## Next Step

Use accepted core/supporting sources to create 20-30 compact grounding cards. Keep deferred/manual-review sources out of cards until extraction and licensing are resolved.
