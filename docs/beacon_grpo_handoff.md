# Beacon Tool-Learning Handoff

## Goal

Beacon / Sankat Saathi is an offline crisis assistant. The current work is about teaching the model when and how to use an offline official-document tool, especially for exact safety facts, thresholds, rules, and source-sensitive guidance.

The core behavior we want:

1. Use the tool when the user asks for exact official facts, thresholds, durations, temperatures, quantities, or source-sensitive rules.
2. Do not use the tool for ordinary practical support, emotional support, broad common-sense crisis guidance, or live/current status verification.
3. When tools are used, answer only from returned evidence.
4. If evidence does not support the claim, abstain clearly and give safer generic next steps.

## Current Approved SFT Dataset

Final reviewed SFT dataset:

`C:\Users\risha\Documents\New project 5\data\assistant_sft\beacon_tool_plus_no_tool_sft_v1_final_reviewed`

Dataset summary:

| Field | Count |
| --- | ---: |
| Total rows | 1,390 |
| Train | 1,126 |
| Dev | 132 |
| Final eval | 132 |
| Tool-required rows | 950 |
| No-tool rows | 440 |
| Assistant tool calls | 1,900 |
| Tool results | 1,900 |
| Tool validation errors | 0 |

Important properties:

- Every tool-required row has an executable `search_official_docs` call followed by `read_official_doc`.
- Tool results were regenerated and validated against the current local tool index.
- Stored tool payloads exactly match the local tool outputs after repair.
- The user approved this dataset for training on May 18, 2026.

## Tool Runtime Contract

Runtime prompt:

`C:\Users\risha\Documents\New project 5\prompts\beacon_tool_system_prompt_v1.md`

Available tools:

### `search_official_docs`

Purpose: search the local index of approved official, NGO, and public-health disaster documents.

Arguments:

```json
{"query": "string", "hazard": "string|null", "organization": "string|null", "top_k": "integer"}
```

### `read_official_doc`

Purpose: read relevant sections from one document returned by `search_official_docs`.

Arguments:

```json
{"doc_id": "string", "section_or_page_query": "string", "top_k": "integer"}
```

Tool protocol:

1. First call must be `search_official_docs`.
2. Then call `read_official_doc` with a `doc_id` returned by search.
3. Then answer naturally using only returned evidence.
4. If evidence is weak, unrelated, or missing, say that the offline documents do not support the specific claim.

## Local Official Doc Index

Tool index:

`C:\Users\risha\Documents\New project 5\data\local_grounding\official_doc_tool_v1`

Key files:

- `official_doc_index.jsonl`
- `official_doc_section_index.jsonl`
- `official_doc_chunk_index.jsonl`
- `manifest.json`

Implementation:

`C:\Users\risha\Documents\New project 5\src\sankat_saathi_dataset\local_doc_tool.py`

The current retrieval uses BM25-style lexical matching over approved document and section indexes.

## Training So Far

We trained a tool-aware SFT layer using the final reviewed dataset.

Base/starting point:

`beacon-assistant-sft-v1-ckpt300-best-adapter`

Important result:

- SFT improved format exposure and demonstrated the tool protocol.
- But eval suggested the model still does not reliably choose tools for exact-constant cases.
- This raised the need for preference/RL-style training focused on tool-use decisions.

## DPO Attempt

DPO dataset:

`C:\Users\risha\Documents\New project 5\data\preference_dpo\beacon_tool_use_dpo_v1_curated`

Kaggle dataset:

`rishavutkarsh/beacon-tool-use-dpo-v1-curated`

Counts:

| Split | Rows |
| --- | ---: |
| Train | 1,898 |
| Dev | 229 |
| Final eval | 241 |
| Total preference pairs | 2,368 |

DPO pair types:

- `tool_decision`
- `read_doc_decision`
- `final_grounding`
- `no_tool_decision`
- `no_tool_helpfulness`
- `uncertainty_boundary`
- `wrong_tool_contrast`

Status:

- Several Kaggle TRL/Unsloth DPO attempts hit OOM or PEFT adapter/reference issues.
- We patched the next DPO notebook to use explicit train/reference adapters, shorter lengths, and precomputed ref logprobs, but the user stopped that run.
- DPO is still possible later, but the current strategy is shifting to GRPO because the target behavior is about the model's own tool-use decisions during generation.

## Why GRPO Now

The problem is not only "prefer answer A over answer B." We want the model to make a sequence of decisions:

1. Should I use a tool?
2. What search query should I write?
3. Did I choose a returned doc?
4. Did I read before answering?
5. Did I answer only from evidence?
6. Did I abstain when evidence does not support the claim?

GRPO is a better conceptual fit because it can reward or penalize the model's generated behavior directly.

## Proposed GRPO Reward Design

User-specified reward sketch:

| Behavior | Reward |
| --- | ---: |
| Right grounded answer | `+7` |
| Wrong answer | `-5` |
| Unsafe answer | `-8` |
| Wrong or failed tool call | Negative |
| Good keyword search | `+1` |
| Great keyword search | `+2` |
| Search reward cap | `+3` |

Important note:

Search reward should be capped at `+3`. After that, extra searching should not add points, but the model can still gain answer-quality points.

## Judge Plan

Use a hybrid reward function:

1. Deterministic checks for things we can verify exactly.
2. Qwen as an LLM judge only for semantic quality and support checks.

Deterministic checks:

- Valid `<tool_call>...</tool_call>` JSON.
- Tool names are only `search_official_docs` and `read_official_doc`.
- `read_official_doc` is not called before search.
- `read_official_doc.doc_id` came from search results.
- Required constants from `expected_facts` appear when needed.
- No unsupported exact claims.
- No live-status hallucination.
- No tool use in no-tool rows unless genuinely justified.
- No final answer before required evidence is read.

Qwen judge checks:

- Is the final answer correct for the user request?
- Is it grounded in the provided evidence?
- Does it add unsupported official facts?
- Is it unsafe?
- Is abstention appropriate when evidence is missing?

Qwen should emit strict JSON, not a raw scalar score. The reward code should map the JSON verdict into scalar rewards.

Example judge JSON shape:

```json
{
  "answer_correct": true,
  "grounded_in_evidence": true,
  "unsupported_extra": false,
  "unsafe": false,
  "abstention_correct": false,
  "notes": "Short reason."
}
```

## GRPO Dataset Plan

Create prompt-only rows from the approved SFT dataset.

Suggested output:

`C:\Users\risha\Documents\New project 5\data\rl_grpo\beacon_tool_use_grpo_v1`

Each row should include:

- `row_id`
- `split`
- `hazard`
- `row_family`
- `tool_required`
- `query_rewrite_required`
- `prompt`
- `prompt_messages`
- `user_prompt`
- `expected_facts`
- `allowed_doc_ids`
- `allowed_section_ids`
- `gold_tool_query`
- `target_response`
- `reward_rubric`

The prompt should normally contain only the system/tool contract and user turn. It should not include the gold tool trace or target answer.

## GRPO Notebook Plan

Suggested notebook directory:

`C:\Users\risha\Documents\New project 5\kaggle_notebooks\beacon-tool-grpo-qwen-judge-v1`

Suggested first run should be conservative:

- Start from the current best tool-aware SFT adapter or the previous best SFT adapter, depending on eval target.
- Use LoRA only.
- Use a small prompt subset first.
- Use short max prompt/completion lengths.
- Use low `num_generations`, likely 2 to 4.
- Avoid loading a large local Qwen judge on the same T4 as the policy model unless memory is proven safe.
- Prefer deterministic reward for the first smoke run, with Qwen judge as a separate offline scorer or lightweight optional stage.

Reason:

GRPO samples multiple completions per prompt, so it is more memory-heavy than SFT and can be more expensive than DPO.

## Deployment Gate

Standing user rule:

Before any Kaggle deployment, do a quick code review.

Deployment gate doc:

`C:\Users\risha\Documents\New project 5\docs\beacon_kaggle_deployment_gate.md`

Required before push:

1. Local syntax check.
2. Local path/config sanity check.
3. Memory-risk review.
4. At least one independent code-review pass.
5. Two reviewers for high-risk training changes.
6. Push only after review passes.
7. Pull logs immediately on failure and iterate faster.

## Next Concrete Work

1. Build `scripts/create_beacon_tool_grpo_dataset.py`.
2. Generate `data/rl_grpo/beacon_tool_use_grpo_v1`.
3. Add validation report and manifest.
4. Create a Kaggle GRPO notebook skeleton.
5. Implement deterministic reward functions first.
6. Add Qwen judge integration as optional/scored mode, not necessarily inside the first GPU training loop.
7. Run local validation.
8. Get reviewer pass before any Kaggle push.

## Key Principle

SFT teaches the model the tool syntax and examples of good behavior. GRPO should teach the decision policy: use the tool when exact official evidence is needed, avoid it when it is not needed, search well, read before answering, and abstain when the offline documents cannot support the claim.
