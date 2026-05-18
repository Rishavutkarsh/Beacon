# Beacon Tool-Use DPO V1 Reviewed Report

## Decision

Approved for sampling/reviewer spot-check, not yet approved for training launch.

I do not recommend the first full-trajectory DPO package as the training input,
because normal DPO trainers score the whole `chosen` / `rejected` continuation.
That would score synthetic tool-result turns as if the model wrote them.

This reviewed package fixes that by making every preference continuation an
assistant-only action:

- tool call vs unsupported direct answer
- correct read call vs wrong read call
- grounded final answer vs unsupported final answer
- no-tool practical answer vs unnecessary tool call

Tool-result turns can appear in the `prompt` for final-answer comparisons, but
never in `chosen` or `rejected`.

## Counts

- Source SFT rows reviewed: 1390
- DPO pairs created: 2985
- Train/dev/final_eval: 2406 / 289 / 290
- Review packets: 14

Pair types:

- `tool_decision`: 950
- `final_grounding`: 950
- `read_doc_decision`: 475
- `wrong_tool_contrast`: 170
- `no_tool_decision`: 440

Rejected types:

- `skipped_required_tool_or_fabricated_answer`: 950
- `unsupported_final_answer`: 950
- `wrong_read_doc_call`: 475
- `wrong_tool_query_or_live_lookup`: 170
- `unnecessary_tool_use`: 440

## Review Checks

Passed:

- `chosen` and `rejected` are never empty.
- `chosen` never equals `rejected`.
- `chosen` / `rejected` contain zero `<tool_result>` turns.
- no-tool chosen completions contain zero `<tool_call>` turns.
- tool-decision chosen completions all contain `<tool_call>`.
- every row has `review_decision=approved`.
- every row has packet assignment and review notes.

## Residual Risk

Rejected continuations are synthetic. They are intentionally bad but short and
patterned. Before a real DPO run, sample per packet and make sure the model does
not overlearn one exact bad phrase. If needed, diversify rejected final-answer
templates.

No DPO/PPO/GRPO run has been launched.
