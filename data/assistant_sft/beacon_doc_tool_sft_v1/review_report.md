# Beacon Official-Doc Tool SFT v1

This candidate lane teaches explicit document lookup. It is not training-approved.

- Status: `valid`
- Rows: 1200
- Tool-required rows: 960
- Training export allowed: `False`

## Validation

- No validation issues.

## Sample Rows

- `beacon_doc_tool_sft_v1_0000` tool_grounded -> docs: cdc_food_after_emergency, fda_food_water_floods
- `beacon_doc_tool_sft_v1_0001` tool_no_support -> docs: ready_heat, ndma_cyclone_guidelines_pdf, cdc_food_after_emergency
- `beacon_doc_tool_sft_v1_0002` query_rewrite_tool_grounded -> docs: ndma_heat_wave, ready_heat
- `beacon_doc_tool_sft_v1_0003` tool_grounded -> docs: ndma_heat_wave, ready_heat
- `beacon_doc_tool_sft_v1_0004` tool_grounded -> docs: cdc_food_after_emergency, fda_food_water_floods
- `beacon_doc_tool_sft_v1_0005` query_rewrite_tool_no_support -> docs: cdc_food_after_emergency, who_diarrhoea, ndma_cyclone_guidelines_pdf
- `beacon_doc_tool_sft_v1_0006` query_rewrite_tool_grounded -> docs: cdc_floodwater_safety, cdc_reenter_flooded_home
- `beacon_doc_tool_sft_v1_0007` tool_grounded -> docs: cdc_floodwater_safety, epa_emergency_disinfection
- `beacon_doc_tool_sft_v1_0008` tool_grounded -> docs: cdc_emergency_water, epa_emergency_disinfection, fda_food_water_floods
- `beacon_doc_tool_sft_v1_0009` tool_no_support -> docs: ndma_cyclone_guidelines_pdf, cdc_food_after_emergency, ready_floods
- `beacon_doc_tool_sft_v1_0010` query_rewrite_tool_grounded -> docs: nws_turn_around, ready_floods
- `beacon_doc_tool_sft_v1_0011` tool_grounded -> docs: ready_floods, ready_power_outages
- `beacon_doc_tool_sft_v1_0012` tool_grounded -> docs: epa_emergency_disinfection, cdc_emergency_water, fda_food_water_floods
- `beacon_doc_tool_sft_v1_0013` query_rewrite_tool_no_support -> docs: ndma_cyclone_guidelines_pdf, nws_lightning, cdc_food_after_emergency
- `beacon_doc_tool_sft_v1_0014` query_rewrite_tool_grounded -> docs: fda_food_water_floods, cdc_food_after_emergency, cdc_floodwater_safety
- `beacon_doc_tool_sft_v1_0015` tool_grounded -> docs: fda_food_water_floods, cdc_food_after_emergency, cdc_power_outage
- `beacon_doc_tool_sft_v1_0016` tool_grounded -> docs: cdc_diabetes_emergencies, cdc_insulin_emergency
- `beacon_doc_tool_sft_v1_0017` tool_no_support -> docs: cdc_food_after_emergency, cdc_power_outage, ndma_cyclone_guidelines_pdf
- `beacon_doc_tool_sft_v1_0018` query_rewrite_tool_grounded -> docs: cdc_emergency_water, epa_emergency_disinfection, fda_food_water_floods
- `beacon_doc_tool_sft_v1_0019` tool_grounded -> docs: cdc_emergency_water, epa_emergency_disinfection
