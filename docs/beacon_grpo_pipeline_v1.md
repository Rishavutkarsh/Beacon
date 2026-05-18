# Beacon Tool-Use GRPO Pipeline v1

This launch starts from the best tool-aware SFT adapter and the prompt-only
GRPO dataset derived from the approved SFT rows. It is a full first-turn GRPO
run, not a full tool-loop GRPO run. A later run can switch the starting adapter
to the evaluated DPO adapter if that evaluation passes.

## Research Grounding

- GRPO was introduced in DeepSeekMath as a critic-free group-relative RL method
  for language-model reasoning: https://arxiv.org/abs/2402.03300
- TRL's `GRPOTrainer` supports custom reward functions that return one scalar
  per completion, multiple reward functions, and recent tool/environment-based
  agent training: https://huggingface.co/docs/trl/grpo_trainer

## Data

Dataset builder:

`scripts/create_beacon_tool_grpo_dataset.py`

Output:

`data/rl_grpo/beacon_tool_use_grpo_v1`

The rows are prompt-only. Prompts contain the Beacon tool contract and the user
turn, but not the gold tool trace or target answer. Gold queries, allowed docs,
expected facts, and target responses are metadata for deterministic rewards,
offline judging, and audit.

## Reward Shape

The first training loop is deterministic and memory conservative:

- Reward valid JSON tool-call syntax and allowed tool names.
- Reward `search_official_docs` before `read_official_doc`.
- Reward reading a doc ID allowed by the reviewed trace.
- Reward good search keywords, capped at +3 total.
- Reward supported exact facts in grounded rows.
- Reward abstention on no-support rows.
- Penalize live-status hallucination, unsupported exact claims, unsafe claims,
  malformed tool calls, wrong tools, and unnecessary tools on no-tool rows.

Qwen judging is intentionally optional and should first run as an offline scorer
that emits strict JSON verdicts. Do not load a large Qwen judge beside the policy
model on a single T4 until memory is proven safe.

## Kaggle Run

Notebook skeleton:

`kaggle_notebooks/beacon-tool-grpo-qwen-judge-v1/beacon_tool_grpo_qwen_judge_v1.py`

Current full first-turn defaults:

- all train rows
- `DEV_ROW_LIMIT = 64`
- `NUM_GENERATIONS = 2`
- `MAX_PROMPT_LENGTH = 512`
- `MAX_COMPLETION_LENGTH = 192`
- `MAX_STEPS = 150`
- deterministic rewards only
- no vLLM
- trainable PEFT adapter loaded from `beacon-tool-sft-best-dev-adapter`

## Deployment Gate

Before any Kaggle push, follow:

`docs/beacon_kaggle_deployment_gate.md`

Minimum required checks:

- local syntax check
- local dataset validation
- path/config sanity review
- memory-risk review
- independent code-review pass
