# Beacon Doc-Tool Natural SFT v2 Candidate Review

Status: candidate assembled; not training-export ready.

## What Was Generated

- Source packet directory: `data/assistant_sft/beacon_doc_tool_sft_v2_natural_packets/`
- Assembled candidate directory: `data/assistant_sft/beacon_doc_tool_sft_v2_natural_candidate/`
- Total generated rows: 300
- Auto-approved for human review: 284
- Auto-rejected: 16

## Packet Results

| Packet | Family | Generated | Auto-approved | Auto-rejected | Notes |
|---|---|---:|---:|---:|---|
| 01 | flood entry / first action | 60 | 45 | 15 | Good coverage, but repeated answer stems remain. Needs top-up or rewrite before training. |
| 02 | WhatsApp route rumor | 60 | 60 | 0 | Revised successfully. Strong no-live-status behavior. |
| 03 | post-flood cleanup | 60 | 59 | 1 | Mostly strong. One row rejected for unsafe/safety-certainty wording. |
| 04 | generator smell / CO | 60 | 60 | 0 | Revised successfully. Fills the prior smell/no-smell CO gap. |
| 05 | shelter fear / conflict | 60 | 60 | 0 | Good coverage of emotional/conflict cases; split needs balancing. |

## Strict Gate Used

The auto gate rejects rows for:

- invalid JSON or missing required fields;
- missing search/read tool-call trace;
- unknown document or section IDs;
- missing `Evidence:` citation;
- unsupported live/current-status certainty;
- visible scaffold language from older rows;
- overused answer stems;
- overly long or too-short final answers.

Scripts:

- `scripts/review_beacon_doc_tool_natural_packets.py`
- `scripts/assemble_beacon_doc_tool_natural_packets.py`

## Human Review Notes

The best packets are 02, 04, and 05. They directly address the gaps the user named:

- unverified WhatsApp road/shelter/rescue rumors;
- generator smell/no-smell carbon monoxide trap;
- scared people arguing in shelters.

Packet 03 is also mostly usable and should only lose the one flagged row.

Packet 01 is safe but still a little stiff in several rows. I recommend generating or rewriting 16-20 replacement flood-entry rows before final SFT collation, rather than accepting the repeated-stem rows.

## Current Split Shape

- Train: 239
- Dev: 28
- Final eval: 17

This split is inherited from packets and should be reshuffled after final approval. Packet 05 currently has no final-eval rows.

## Recommendation

Do not train directly on this folder yet. Next step:

1. Create a small top-up/rewrite packet for flood-entry first-action rows.
2. Run the strict gate again.
3. Do a final row-by-row human approval pass.
4. Reshuffle into a cleaner train/dev/final split.
5. Only then mark `training_export_allowed=true`.
