# Sankat Saathi E2B Short-GPU Evidence Sprint

## Summary
The native-template `100 train / 20 eval` Gemma 4 E2B-IT LoRA sprint completed successfully on Kaggle T4. The run is valid as an infrastructure and training-path proof, but it is not a promotable behavioral improvement.

The adapter trained and eval loss improved versus the earlier tiny smoke run, but deterministic generation on the frozen 20-prompt eval set was exactly identical to base Gemma 4 E2B-IT.

## Run Artifacts
- CPU preflight: `outputs/kaggle_gemma_e2b_preflight_v10/sankat_saathi_gemma4_e2b_preflight/preflight_report.json`
- SFT output: `outputs/kaggle_gemma_e2b_sft_v6/sankat_saathi_gemma4_e2b_it_lora_sft/`
- Behavioral eval: `outputs/kaggle_gemma_e2b_eval_v3/sankat_saathi_e2b_eval/`
- Side-by-side examples: `outputs/kaggle_gemma_e2b_eval_v3/side_by_side_template100.md`

## Preflight Result
- Dataset mounted on Kaggle CPU with `100` train rows and `20` eval rows.
- Gemma 4 model mount resolved as `google/gemma-4/Transformers/gemma-4-e2b-it/1`.
- Processor resolved as `Gemma4Processor`.
- Native chat serialization confirmed with `<|turn>` markers and `enable_thinking=False`.
- Token lengths were safe for `max_length=768`: max full example length was `512` tokens.
- Dataset errors: none.

## Training Result
- Kaggle accelerator: Tesla T4.
- Model type: `gemma4`.
- LoRA target modules: `["linear"]`.
- Effective LoRA rank: `2`.
- Trainable params: `1,376,256`.
- Training rows: `100`.
- Eval rows: `20`.
- Epochs: `3`.
- Train steps: `12`.
- Eval loss: `5.384`.
- Eval perplexity: `217.90`.

Compared with the earlier old-template tiny run, eval loss improved from `6.021` to `5.384`, so the corrected training path is functioning.

## Behavioral Eval Result
Frozen eval compared:
- base Gemma 4 E2B-IT
- old tiny adapter
- native-template `100/20` adapter

All three produced the same aggregate behavioral metrics:

| Model | Unsafe certainty | Fabricated live facts | Structured output | Escalation mean | Useful action | Critical safety pass |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.00 | 0.05 | 0.40 | 0.5875 | 0.90 | 0.154 |
| Old tiny adapter | 0.00 | 0.05 | 0.40 | 0.5875 | 0.90 | 0.154 |
| Native-template 100/20 adapter | 0.00 | 0.05 | 0.40 | 0.5875 | 0.90 | 0.154 |

The base and native-template adapter generations were exactly identical on all 20 prompts.

## Interpretation
This does not prove SFT cannot help. It shows this bounded LoRA setup did not shift deterministic generation enough to change argmax outputs on the frozen eval.

Likely causes:
- Effective rank `2` and only `12` optimizer steps are still very small.
- The approved dataset remains pattern-heavy and may not contain enough targeted contrast against base weaknesses.
- Deterministic decoding can hide small probability shifts if the top token does not change.
- Base Gemma already follows several structure and safety cues from the prompt, so the adapter needs stronger targeted signal to produce visible changes.

## Decision
Do not promote the native-template `100/20` adapter as a quality improvement.

Use it as evidence that:
- the Kaggle E2B path works,
- native-template training is wired correctly,
- CPU preflight prevents wasted GPU time,
- behavioral eval is now capable of catching non-improvement honestly.

## Next Move
Before another GPU run, improve data and eval rather than only increasing epochs:
- Expand approved data toward `300 train / 60 eval`.
- Add more examples that explicitly correct current base weaknesses: incomplete sections, missing uncertainty notes, fabricated live facts, weak Hinglish completion, diabetes/ORS nuance, and flood-food specificity.
- Add a small demo-critical eval subset with human preference labels, not only rule metrics.
- Consider a slightly stronger but still bounded run only after data improves: more steps, and rank higher than `2` only if a Kaggle preflight proves stable.

For the hackathon story, keep the claim disciplined: Sankat Saathi is an offline Gemma crisis companion with a working training/eval pipeline and transparent safety gates. Fine-tuning improvement is not yet established.
