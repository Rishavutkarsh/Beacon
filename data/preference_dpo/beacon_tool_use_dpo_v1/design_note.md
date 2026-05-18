# Beacon Tool-Use DPO V1

This package converts the approved Beacon tool-aware SFT rows into DPO-style
preference pairs.

## Purpose

The preference target is tool policy, not memorized disaster facts:

- Prefer tool calls for exact official constants, thresholds, source-sensitive
  rules, and live-status boundaries.
- Prefer abstention when retrieved evidence does not support a live/current
  claim.
- Prefer no-tool direct help for broad practical crisis guidance.
- Reject answers that skip required evidence, invent live facts, use the wrong
  tool path, or call tools when the user only needs general safe steps.

## Schema

Each row contains:

- `prompt`: system + user transcript.
- `chosen`: approved continuation from the reviewed SFT row.
- `rejected`: synthetic bad continuation for the same prompt.
- `prompt_messages`, `chosen_messages`, `rejected_messages`: structured message
  equivalents.
- `rejected_type`: why the rejected continuation should lose.
- `source_sft_row_id`: original approved SFT row.

Tool results are represented as synthetic user turns in the text transcript,
matching the SFT export convention.

## Counts

- Total pairs: 1390
- Train/dev/final_eval: 1126 / 132 / 132
- Tool-required pairs: 950
- No-tool pairs: 440

Rejected contrast types:

- `skipped_tool_wrong_or_unsupported_constant`
- `fabricated_live_or_missing_support`
- `skipped_required_tool`
- `wrong_tool_query_or_doc`
- `unnecessary_tool_use`

## Review Status

This is ready for review, not automatically training-approved. The chosen side
inherits reviewed SFT rows. The rejected side is generated contrast data and
should be sampled before any DPO run.

No DPO/GRPO/PPO run has been launched.
