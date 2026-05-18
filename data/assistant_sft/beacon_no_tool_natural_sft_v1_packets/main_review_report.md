# Beacon No-Tool Natural SFT v1 - Main Review

Reviewed by main assistant on 2026-05-18. This review covers all 240 candidate rows in:

- `packet_01_flood_entry_first_action_candidates.jsonl`
- `packet_02_whatsapp_route_rumor_candidates.jsonl`
- `packet_04_generator_smell_co_candidates.jsonl`
- `packet_05_shelter_conflict_fear_candidates.jsonl`

No rows from this set have been added to the training mix by this review.

## Summary

These rows are materially better than the older `beacon_assistant_sft_v1_candidate_800` no-tool rows. They are more natural, less scaffolded, and better aligned with Beacon's intended assistant behavior: practical first steps, uncertainty boundaries, and no invented live facts.

Recommended status: promising candidate pool, not training-ready as-is.

## Packet Decisions

| Packet | Rows | Main Review Decision | Notes |
|---|---:|---|---|
| `packet_01_flood_entry_first_action` | 60 | mostly approve after light cleanup | Strong flood/wet-electricity first-action behavior. Good vulnerable-person handling. Keep as primary no-tool source. |
| `packet_02_whatsapp_route_rumor` | 60 | approve after targeted rewrites | Good live-status and rumor discipline, but several rows have template leakage or mismatched next-step forms. |
| `packet_04_generator_smell_co` | 60 | approve | Strongest packet. Crisp, useful CO/generator behavior. Good refusal of smell/window/balcony/garage myths without exact-distance claims. |
| `packet_05_shelter_conflict_fear` | 60 | rewrite for diversity before use | Safety content is mostly good, but answer stems repeat heavily. Use as rewrite source, not as-is. |

## Rows Needing Targeted Rewrite

`packet_01_flood_entry_first_action`
- `beacon_no_tool_natural_sft_v1_packet_01_0052`: user says "Can the tool say"; answer should avoid tool framing and say Beacon cannot certify the ward safe.

`packet_02_whatsapp_route_rumor`
- `beacon_no_tool_natural_sft_v1_packet_02_0019`: medicine-request template is awkward for ORS/tablets plus cut foot; rewrite to water/ORS need plus wound/floodwater safety.
- `beacon_no_tool_natural_sft_v1_packet_02_0024`: "keep people away from crutches and road tape" is wrong/awkward; should be away from barricade/culvert/road edge, and assist aunt with crutches.
- `beacon_no_tool_natural_sft_v1_packet_02_0026`: Aadhaar/rescue prompt needs privacy guidance, not only live-status guidance.
- `beacon_no_tool_natural_sft_v1_packet_02_0035`: "keep people away from rain and removed barrier" is awkward; rewrite around culvert/barrier/floodwater.
- `beacon_no_tool_natural_sft_v1_packet_02_0040`: baby medicine row should avoid implying doses; keep to request/medical-worker contact.
- `beacon_no_tool_natural_sft_v1_packet_02_0046`: "keep people away from closed road board" is awkward; rewrite around closed road/shortcut/barricade.
- `beacon_no_tool_natural_sft_v1_packet_02_0052`: water purification tablet row wrongly asks for patient medicine details; rewrite to water need, child count, location, and safer water options.
- `beacon_no_tool_natural_sft_v1_packet_02_0057`: "keep people away from single audio" is awkward; rewrite around underpass/barricade/floodwater.
- `beacon_no_tool_natural_sft_v1_packet_02_0059`: public live-location prompt needs privacy/minimum-necessary sharing guidance, not generic shelter/road status.

`packet_05_shelter_conflict_fear`
- All 60 rows need diversity rewrite or selective downsampling. The content is usually safe and useful, but repeated stems would overtrain a rigid style:
  - "Start with the safest visible thing..."
  - "Pause the line briefly..."
  - "Clear the exit before settling..."
  - "Treat privacy as part of safety..."
  - "Do not repeat the crowd label..."
  - "Move from blame to layout..."
  - "No. Beacon cannot verify..."

## Strengths

- Good no-tool behavior: the model learns to give broad practical help without calling docs.
- Good live uncertainty: does not invent route, shelter, rescue, stock, warning, or road status.
- Good vulnerable-person handling: children, elders, pregnancy, disability, oxygen/medical device needs.
- Good CO boundaries: no smell/window/balcony/garage/open-door shortcuts; fresh air and urgent help for symptoms.
- Good shelter conflict instincts: de-escalate, protect exits, avoid blame, protect privacy, escalate violence.

## Risks

- Packet 05 repetition could make the model sound mechanical.
- Some packet 02 rows use a generic "send patient name, age, medicine..." form even when the user need is water, ORS, Aadhaar/privacy, or route uncertainty.
- Several packet 02 rows have awkward object phrasing from templates, such as "keep people away from crutches" or "single audio."
- These rows are no-tool rows; they should not teach exact constants. Keep exact threshold/fact questions in the tool-aware lane.

## Recommendation

Use these as the preferred no-tool pool after review/rewrite:

1. Keep `packet_04_generator_smell_co` nearly intact.
2. Keep most of `packet_01_flood_entry_first_action`.
3. Use `packet_02_whatsapp_route_rumor` after targeted rewrites listed above.
4. Rewrite or downsample `packet_05_shelter_conflict_fear` before mixing.

Do not use the old 800-row no-tool candidate package directly. It is broader, but much more scaffolded and still fully marked pending review.
