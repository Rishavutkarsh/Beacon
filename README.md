# Sankat Saathi / Beacon Dataset And Training Pipeline

Offline crisis-companion dataset, CPT, SFT, and evaluation workspace for the
Kaggle Gemma 4 Good Hackathon.

## Current Beacon Status

Beacon now has a train-ready crisis CPT package, Kaggle T4 training/eval
kernels, a first CPT adapter evaluation, and the start of a separate assistant
SFT data track.

- CPT corpus package: `data/dapt_corpus/beacon_crisis_v1_cpt_kaggle/`
- CPT config used for the first run: `lr=1e-5`, `2 epochs`, `max_seq_length=1024`,
  packed raw-text CPT, QLoRA attention+MLP, `r=16`, `alpha=32`.
- CPT eval result: CPT improved source-QA knowledge over base, especially on
  electrical/flood, some CO, water-disinfection, and shelter-hygiene questions.
  It also produced more unsupported source-adjacent extras, so it is useful as
  an SFT initialization candidate, not a deployable assistant by itself.
- Source-QA comparison summary:
  `analysis/beacon_source_qa_eval_v1/source_qa_base_vs_cpt_comparison.json`
- Assistant SFT draft bundle:
  `data/assistant_sft/beacon_assistant_sft_v1_draft/`

Do not treat CPT as final behavior training. CPT is for domain adaptation;
assistant behavior should come from reviewed SFT/DPO/GRPO-style data later.

## Padhai Lens Track

This repo also contains `padhai_lens/`, a model-first NCERT curiosity tutor
pipeline for the same hackathon. It targets Digital Equity + Future of Education
with small Class 8 chapter packs and a Gemma E2B LoRA workflow.

Quick smoke:

```powershell
python scripts/padhai_generate_dataset.py --profile tiny
python scripts/padhai_validate_dataset.py padhai_lens/data/processed/tiny --write-report
python scripts/padhai_prepare_training_data.py padhai_lens/data/processed/tiny --allow-unapproved-smoke
python scripts/padhai_eval_report.py padhai_lens/data/processed/tiny/eval_heldout.jsonl
```

Full training target:

```powershell
python scripts/padhai_generate_dataset.py --profile full
python scripts/padhai_validate_dataset.py padhai_lens/data/processed/full --strict --write-report
```

See `padhai_lens/README.md` for the review/export/SFT/DPO gates.

This repo currently focuses on dataset creation, training readiness, Kaggle
execution scripts, and offline evaluation. It generates reviewable text and
image+text examples grounded in official disaster, WASH, food-safety, and
risk-communication guidance.

## Beacon CPT Workflow

Build and validate the CPT package:

```powershell
python scripts/prepare_beacon_cpt_package.py build
python scripts/prepare_beacon_cpt_package.py validate
```

Kaggle kernels used for CPT:

```powershell
kaggle kernels push -p kaggle\beacon_dapt_cpt_smoke --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle\beacon_dapt_cpt_train --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle\beacon_dapt_cpt_eval --accelerator NvidiaTeslaT4
```

Important: always use the explicit `NvidiaTeslaT4` accelerator override. The
scripts also check for a single visible T4, pinned Unsloth dependencies, dataset
hashes, QLoRA target scope, finite losses, and adapter save/load artifacts.

## Beacon Behavioral And Source-QA Eval

Two eval tracks are currently separated:

- Real-problem assistant behavior eval: realistic crisis user situations. This
  is a safety/usefulness side report for behavior, not the main CPT metric.
- Source-QA knowledge eval: closed-book factual questions grounded in the crisis
  corpus. This is the main CPT domain-knowledge check.

Source-QA artifacts are produced by:

```powershell
python scripts/build_beacon_source_qa_eval.py build
python scripts/build_beacon_source_qa_eval.py validate
```

Kaggle kernels for source-QA generation and judging:

```powershell
kaggle kernels push -p kaggle\beacon_source_qa_base_generation --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle\beacon_source_qa_cpt_generation --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle\beacon_source_qa_qwen_judge --accelerator NvidiaTeslaT4
```

Lesson learned: prefer separate judge runs per candidate, then aggregate locally.
The paired judge works, but it rejudges base unnecessarily.

## Beacon Assistant SFT Draft

The SFT track is intentionally separate from CPT. A small reviewed-draft scaffold
exists at:

```powershell
python scripts/build_beacon_assistant_sft.py
python scripts/validate_beacon_assistant_sft.py data/assistant_sft/beacon_assistant_sft_v1_draft
```

The current assistant SFT draft is not approved for training. Its manifest has
`training_export_allowed=false`; review, expansion, and safety/source checks are
required before any SFT launch.

## Beacon Ollama Local Demo

Beacon's local demo path is documented in `docs/beacon_ollama_local.md`.
The short version:

```powershell
python scripts\beacon_docs_cli.py list-docs --query "generator carbon monoxide placement" --json
python scripts\export_beacon_ollama.py --help
ollama create beacon-gemma4-current-best -f ollama\Modelfile
powershell -ExecutionPolicy Bypass -File scripts\beacon_demo_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\promote_beacon_ollama_export.ps1
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "Can I run a generator in my garage if the door is open?"
```

The current-best Q4_K_M GGUF has been exported on Kaggle and copied to
`ollama/beacon-gemma4-e2b-current-best-q4_k_m.gguf`. Ollama is the laptop-local
model runner; Beacon's controller owns offline official docs, retrieval, prompt
assembly, and citation policy. The active model pointer is
`config/beacon_current_model.json`, currently set to Beacon Tool DPO CPT
Fullprompt Ckpt50; update that and rerun
`kaggle/beacon_ollama_export_current_best` when a better adapter is selected
later.

## What It Builds

- `guidance_facts`: source-backed atomic safety rules.
- `sft_text`: crisis instruction examples with structured English + Hindi/Hinglish answers.
- `sft_vision`: image+text examples with explicit visual uncertainty behavior.
- `dpo_pairs`: chosen/rejected preference examples for safer crisis behavior.
- `eval`: held-out scenarios with rubric labels.
- `review_queue.csv`: human-readable rows for approval before training.

## Quick Start

```powershell
python scripts/generate_dataset.py --profile starter
python scripts/validate_dataset.py data/processed/starter
```

Generated files land in `data/processed/starter/`.

For the stronger gated dataset:

```powershell
python scripts/source_report.py
python scripts/image_manifest_report.py
python scripts/generate_dataset.py --profile hardened_text
python scripts/generate_dataset.py --profile hardened_vision
python scripts/generate_dataset.py --profile hardened_full
python scripts/dataset_report.py data/processed/hardened
python scripts/validate_dataset.py data/processed/hardened
```

Strict validation is the training gate and is expected to fail until review,
verified images, approved exports, and the second 3-reviewer receipt exist:

```powershell
python scripts/export_approved.py data/processed/hardened_text --mode text --task sft
python scripts/validate_dataset.py data/processed/hardened_text --strict --mode text --task sft --review-scope approved
python scripts/pre_training_gate.py data/processed/hardened_text --mode text
```

To review in smaller passes:

```powershell
python scripts/create_review_batch.py data/processed/hardened_text --mode text --limit 200
python scripts/apply_review_batch.py data/processed/hardened_text --batch data/processed/hardened_text/review_batch_text_200.csv --dry-run
```

## Kaggle CPU SFT Smoke

This is a tiny CPU-only training smoke test, not the final Gemma LoRA run.

```powershell
python scripts/prepare_cpu_sft_smoke.py --limit 64 --allow-unapproved-smoke
kaggle kernels push -p kaggle
```

Before pushing, upload or attach `kaggle/input/sankat-saathi-cpu-smoke/train.jsonl`
as the Kaggle dataset named in `SANKAT_DATA`, or edit `kernel-metadata.json` to
include the dataset source once created.

## Review Before Training

Open `data/processed/starter/review_queue.csv`, review the examples, and update
the `review_status` column to `approved`, `rejected`, or `edit_needed`.

Training scripts intentionally refuse to run unless a reviewed dataset is passed:

```powershell
python training/train_text_sft_lora.py --dataset-dir data/processed/starter --dry-run
python training/train_dpo_lora.py --dataset-dir data/processed/starter --dry-run
```

For real training, use `data/processed/hardened` only after:

1. Every row in `review_queue.csv` is reviewed.
2. Approved rows also have approved source checks and image-license checks.
3. `scripts/export_approved.py` has written approved-only JSONL files.
4. `scripts/validate_dataset.py --strict` passes.
5. `pre_training_review.json` records go decisions from the three reviewers:
   safety/source-grounding, multimodal/image, and training/eval readiness.

## Source Grounding

Seed rules are derived from official/public guidance:

- WHO household water treatment and safe storage after emergencies.
- WHO/WEDC WASH technical notes for emergencies.
- WHO flood food-safety guidance.
- WHO risk communication and community engagement.
- CDC floodwater safety.
- FDA food and water safety during power outages and floods.
- EPA emergency drinking-water disinfection.
- India NDMA flood/cyclone framing.

The generated dataset keeps source IDs on every example so review and writeup
materials can trace behavior back to public guidance.
