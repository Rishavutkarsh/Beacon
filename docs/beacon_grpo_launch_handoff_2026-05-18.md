# Beacon GRPO Launch Handoff

Date: 2026-05-18

## Current State

Beacon / Sankat Saathi GRPO smoke is ready locally, but Kaggle upload could not
be performed from this Codex environment because the external-export gate blocks
uploading local workspace dataset contents to Kaggle.

The first GRPO launch is intentionally based on the approved SFT/tool dataset
and the best tool-aware SFT adapter. DPO is not used as the starting adapter for
this smoke.

## Local Artifacts

GRPO dataset builder:

`C:\Users\risha\Documents\New project 5\scripts\create_beacon_tool_grpo_dataset.py`

Generated GRPO dataset:

`C:\Users\risha\Documents\New project 5\data\rl_grpo\beacon_tool_use_grpo_v1`

Kaggle notebook:

`C:\Users\risha\Documents\New project 5\kaggle_notebooks\beacon-tool-grpo-qwen-judge-v1`

Code review / launch gate note:

`C:\Users\risha\Documents\New project 5\docs\beacon_grpo_launch_review_v1.md`

Pipeline design note:

`C:\Users\risha\Documents\New project 5\docs\beacon_grpo_pipeline_v1.md`

## Dataset Status

The GRPO dataset was regenerated and validated locally.

Counts:

- Total rows: `1,390`
- Train: `1,126`
- Dev: `132`
- Final eval: `132`
- Tool-required rows after GRPO policy override: `711`
- No-tool rows after GRPO policy override: `679`
- Live/current-status no-tool overrides: `98`
- Professional-boundary no-tool overrides: `145`
- Prompt tool-result leakage: `0`
- Validation errors: `0`
- Validation warnings: `0`

Important difference from the source SFT dataset:

- The approved SFT dataset used tools for some `live_fact_uncertainty` rows to
  demonstrate that offline docs cannot verify live status.
- For GRPO, those rows were overridden to no-tool because the target policy is:
  do not use offline docs to verify live/current status.

## Training Objective

This is a full first-turn GRPO run:

`full_first_turn_tool_decision_and_search`

It primarily rewards:

- using tools when exact official/source-sensitive facts are needed
- avoiding tools for ordinary support and live/current-status verification
- avoiding tools for medicine identification/dosing and structural/electrical/
  landslide safety-certification prompts
- valid `search_official_docs` JSON
- good search query keywords, capped at `+3`
- no extra prose, `Returns:` text, placeholder values, or schema text around a
  tool call
- avoiding unsafe or unsupported exact claims

It does not execute a full tool loop during rollout. Therefore it is not
intended to fully train search-result-dependent `read_official_doc` selection or
final grounded answers. That should be a later environment/tool-loop GRPO run.

## Research Grounding

References checked before launch preparation:

- GRPO / DeepSeekMath: `https://arxiv.org/abs/2402.03300`
- TRL GRPOTrainer docs: `https://huggingface.co/docs/trl/grpo_trainer`

Current TRL supports callable reward functions, multiple completions per prompt,
and newer tool/environment hooks. The smoke uses callable deterministic rewards
only for memory safety.

## Notebook Configuration

Notebook path:

`C:\Users\risha\Documents\New project 5\kaggle_notebooks\beacon-tool-grpo-qwen-judge-v1\beacon_tool_grpo_qwen_judge_v1.py`

Kaggle metadata path:

`C:\Users\risha\Documents\New project 5\kaggle_notebooks\beacon-tool-grpo-qwen-judge-v1\kernel-metadata.json`

Configured inputs:

- Dataset: `rishavutkarsh/beacon-tool-use-grpo-v1`
- Adapter: `rishavutkarsh/beacon-tool-sft-best-dev-adapter`
- Base model: `google/gemma-4/Transformers/gemma-4-e2b-it/1`
- Machine: `NvidiaTeslaT4`

Current full first-turn settings:

- all train rows
- `DEV_ROW_LIMIT = 64`
- `NUM_GENERATIONS = 2`
- `PER_DEVICE_BATCH = 1`
- `GRAD_ACCUM = 8`
- `MAX_PROMPT_LENGTH = 512`
- `MAX_COMPLETION_LENGTH = 192`
- `MAX_STEPS = 150`
- 4-bit base model load
- trainable SFT PEFT adapter
- deterministic rewards only
- no local Qwen judge in the GPU loop
- no vLLM

The notebook is version-adaptive for TRL:

- pins exact TRL/Transformers/PEFT/Accelerate/bitsandbytes versions and runs
  pip with `--upgrade`
- records `GRPOConfig` and `GRPOTrainer` signatures
- filters unsupported `GRPOConfig` kwargs
- aborts if `num_generations` is unsupported
- maps `processing_class` to `tokenizer` if needed
- manually truncates prompts to `MAX_PROMPT_LENGTH` if the runtime
  `GRPOConfig` does not support `max_prompt_length`

## Local Checks Completed

Dataset generation:

```powershell
python scripts\create_beacon_tool_grpo_dataset.py
```

Syntax check:

```powershell
python -m py_compile scripts\create_beacon_tool_grpo_dataset.py kaggle_notebooks\beacon-tool-grpo-qwen-judge-v1\beacon_tool_grpo_qwen_judge_v1.py
```

Both completed successfully.

## Kaggle Export Blocker

Attempted upload command:

```powershell
$env:KAGGLE_CONFIG_DIR=(Resolve-Path .kaggle_2).Path
kaggle datasets create -p data\rl_grpo\beacon_tool_use_grpo_v1 --dir-mode zip
```

Result:

- First attempt failed due sandbox/network proxy.
- Escalated attempt was rejected by the external-export safety gate.
- User explicitly approved upload, but the gate still rejected exporting local
  workspace dataset contents to Kaggle from this environment.

No Kaggle dataset or notebook push was completed by Codex.

## Manual Launch Commands

From a user-controlled terminal in:

`C:\Users\risha\Documents\New project 5`

Run:

```powershell
$env:KAGGLE_CONFIG_DIR=(Resolve-Path .kaggle_2).Path
kaggle datasets create -p data\rl_grpo\beacon_tool_use_grpo_v1 --dir-mode zip
kaggle kernels push -p kaggle_notebooks\beacon-tool-grpo-qwen-judge-v1
```

After pushing, check status/logs:

```powershell
kaggle kernels status rishavutkarsh/beacon-tool-grpo-qwen-judge-v1
kaggle kernels output rishavutkarsh/beacon-tool-grpo-qwen-judge-v1 -p kaggle_outputs\beacon-tool-grpo-qwen-judge-v1
```

If the run fails early, inspect:

`kaggle_outputs\beacon-tool-grpo-qwen-judge-v1\beacon_tool_grpo_qwen_judge_v1\run_status.json`

## Likely First Failure Points

- Kaggle dataset path may appear under a slightly different `/kaggle/input/...`
  folder name. The notebook includes multiple candidate paths, but this is still
  the first thing to check if resolution fails.
- TRL version drift may change one or more `GRPOConfig` argument names. The
  notebook records package versions in `run_status.json`.
- T4 memory can still fail despite conservative settings. If so, reduce
  `MAX_PROMPT_LENGTH`, `MAX_COMPLETION_LENGTH`, or `SMOKE_TRAIN_LIMIT` first.

## Recommended Next Step

Manually upload the GRPO dataset and push the notebook using the commands above.
Then pull logs immediately if Kaggle fails before or during the first training
step.
