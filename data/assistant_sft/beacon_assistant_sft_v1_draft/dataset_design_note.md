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
    "carbon_monoxide": 1,
    "electrical_flood": 1,
    "food_water": 2,
    "landslide_structural": 1,
    "live_route_authority": 2,
    "medicine_diabetes": 1,
    "shelter_hygiene": 1,
    "visual_uncertainty": 1,
    "wash_ors": 1,
    "wounds_first_aid": 1
  },
  "by_language": {
    "english": 7,
    "hinglish": 5
  },
  "by_risk": {
    "critical": 4,
    "high": 4,
    "low": 1,
    "medium": 3
  },
  "by_split": {
    "dev": 2,
    "final_eval": 2,
    "train": 8
  },
  "total": 12
}
```

## Readiness

This is a structurally valid draft package and review seed. It is not yet recommended for SFT because the row count is intentionally small and review statuses are pending.
