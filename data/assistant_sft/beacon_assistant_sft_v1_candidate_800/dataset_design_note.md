# Beacon Assistant SFT v1 Draft Design

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
{
  "by_hazard": {
    "accessibility_elder_disabled_pregnancy_child_language": 34,
    "carbon_monoxide_fuel": 54,
    "crowd_shelter_overcrowding": 20,
    "dam_bund_flash_flood": 3,
    "dam_flash_flood_riverbank_coastal": 18,
    "diabetes_medication": 54,
    "electrical_wet_devices": 60,
    "flash_flood_building_shelter": 3,
    "flash_flood_dry_channel": 3,
    "flash_flood_storm_drain": 3,
    "food_flood_power": 75,
    "heatwave_cold_lightning_dust": 27,
    "infrastructure_power_telecom_road_transit": 10,
    "landslide_structural": 60,
    "misinformation_fake_alerts_helplines_rescue": 20,
    "post_disaster_contamination_infection": 27,
    "route_rescue_live_fact": 66,
    "shelter_hygiene": 54,
    "urban_fire_lpg_chemical": 32,
    "visual_uncertainty": 48,
    "wash_ors_water": 75,
    "wounds_first_aid": 54
  },
  "by_language": {
    "english": 452,
    "hinglish": 348
  },
  "by_risk": {
    "critical": 163,
    "high": 415,
    "medium": 222
  },
  "by_split": {
    "dev": 120,
    "final_eval": 120,
    "train": 560
  },
  "total": 800
}
```

## Readiness

This is a structurally valid candidate package for review. It is not yet recommended for SFT until row review is completed and any flagged rows are edited or removed.
