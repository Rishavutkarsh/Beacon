# Beacon Tool-Use GRPO v1

Prompt-only RL dataset derived from the final reviewed Beacon tool/no-tool SFT dataset.

- Status: `valid`
- Rows: 1390
- Tool-required rows: 711
- No-tool rows: 679
- Training stage intent: `Full first-turn GRPO starts from the best tool-aware SFT adapter; later environment/tool-loop GRPO can train read/final-answer behavior.`
- Training export allowed: `True`

## Validation

- No validation issues.

## Reward Shape

- Deterministic rewards check first-turn tool/no-tool choice, search-call syntax, query quality, no extra prose in tool-call completions, no live-status hallucination, and no placeholder tool arguments.
- Semantic rewards are reserved for an optional strict-JSON Qwen judge outside the first memory-conservative GPU loop.
- Search quality can add at most +3 total reward; extra searches do not keep adding reward.
