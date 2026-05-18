# Beacon Tool-Use DPO Curated Review

## Summary

- Status: `curated_ready_for_sampling`
- Pair count: `2368`
- Splits: `{'train': 1898, 'dev': 229, 'final_eval': 241}`
- Validation errors: `0`
- Training launch: `not_launched`

## Pair Mix

- `final_grounding`: 657
- `no_tool_decision`: 214
- `no_tool_helpfulness`: 226
- `read_doc_decision`: 153
- `tool_decision`: 659
- `uncertainty_boundary`: 291
- `wrong_tool_contrast`: 168

## Reviewer-Driven Changes

- Converted no-support tool rows into direct uncertainty-boundary preference pairs.
- Collapsed no-tool rows to one preference pair per source row.
- Normalized chosen tool-call queries and capped top_k.
- Kept read-doc contrasts only when a wrong but returned document candidate exists.
- Deterministically shuffled all pairs so train files and review packets are not segmented by pair type.
- Regenerated chosen tool queries from user prompt and hazard only, avoiding expected-fact query contamination.
- Dropped reviewer-flagged ambiguous read-doc/final-grounding pairs.

## Review Notes

- No-support tool rows are now direct uncertainty-boundary preferences, so DPO does not reward tool calls whose only purpose is to discover missing evidence.
- No-tool source rows produce exactly one no-tool preference each, avoiding duplicate chosen answers.
- Tool-call continuations are assistant-only and parseable; tool results remain prompt context only.
- Chosen tool-call queries are compacted and `top_k` is capped at 6.
- Read-document negatives are only created from documents that the search tool actually returned.
- Packet balance summary: `{'min_pair_types_per_packet': 7, 'max_pair_types_per_packet': 7, 'packet_count': 16}`

## Independent Review

- Reviewer 1 verdict: approved as a DPO candidate after contaminated query anchors and ambiguous pairs were removed.
- Reviewer 2 verdict: approved as a DPO candidate; no remaining must-fix items.
- Shared caution: run DPO lightly and gate with tool-use evals because repeated rejected wording can still become a shortcut if over-optimized.

## Residual Risks

- This package should be sampled and spot-checked before any DPO launch; it is a preference candidate, not a training approval.
- The model still needs a tool-enabled eval gate because MCQ forced-choice is tool-free and will not measure call timing.
- DPO should be run lightly if used, because over-optimizing tool-call preference can make the assistant too eager to call tools.
