# Beacon GRPO Launch Review v1

Review date: 2026-05-18
Training export approval: `user_approved_2026-05-18_training_export`

Scope:

- `scripts/create_beacon_tool_grpo_dataset.py`
- `data/rl_grpo/beacon_tool_use_grpo_v1`
- `kaggle_notebooks/beacon-tool-grpo-qwen-judge-v1/beacon_tool_grpo_qwen_judge_v1.py`
- `kaggle_notebooks/beacon-tool-grpo-qwen-judge-v1/kernel-metadata.json`

## Research Check

- GRPO is appropriate for behavior shaped by sampled completions and scalar
  rewards, following DeepSeekMath's critic-free group-relative setup:
  https://arxiv.org/abs/2402.03300
- Current TRL `GRPOTrainer` supports callable reward functions, multiple
  generations per prompt, and tool/environment extensions:
  https://huggingface.co/docs/trl/grpo_trainer

## Local Gate

- Dataset builder ran successfully.
- Dataset validation status: `valid`.
- Rows: `1390`.
- Tool-required rows after policy override: `856`.
- No-tool rows after policy override: `534`.
- Live/current-status overrides: `98`.
- Prompt tool-result leakage: `0`.
- Training export approval recorded in the GRPO manifest.
- Syntax check passed:
  `python -m py_compile scripts\create_beacon_tool_grpo_dataset.py kaggle_notebooks\beacon-tool-grpo-qwen-judge-v1\beacon_tool_grpo_qwen_judge_v1.py`

## Review Findings

- Fixed blocker: the first smoke reward previously penalized completions that
  did not include `read_official_doc`, even though the notebook does not run a
  tool loop. The reward now treats this launch as a first-turn
  tool-decision/search smoke.
- Fixed blocker: `live_fact_uncertainty` rows inherited tool-required traces
  from SFT, which conflicted with the target policy of not using tools for
  live/current-status verification. The GRPO dataset now overrides those rows
  to no-tool while preserving the original source metadata.
- Non-blocking risk: this smoke does not execute document tools during rollout,
  so it primarily trains whether to use a tool and how to search. A later
  environment/tool-loop GRPO run is needed to train search-result-dependent read
  and final-answer behavior.
- Non-blocking risk: Kaggle runtime may need TRL argument adjustment if the
  installed TRL version differs from the current documented API. The script
  records package versions and writes `run_status.json` on failure.

## Memory Review

- One T4 target.
- `num_generations = 2`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 8`
- `max_prompt_length = 512`
- `max_completion_length = 192`
- `max_steps = 150`
- all train rows, dev capped at 64 rows
- 4-bit base model load, trainable SFT PEFT adapter, deterministic rewards only,
  no colocated Qwen judge, no vLLM.

Decision: launchable as full first-turn GRPO after dataset and notebook are
uploaded to Kaggle. It does not train read/final-answer behavior; that remains
for a later tool-loop/environment GRPO run.
