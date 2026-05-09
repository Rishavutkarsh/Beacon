# Sankat Saathi Dataset Pipeline

Offline crisis-companion dataset factory for the Kaggle Gemma 4 Good Hackathon.

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

This repo currently focuses on dataset creation and training readiness, not model
training. It generates reviewable text and image+text examples grounded in
official disaster, WASH, food-safety, and risk-communication guidance.

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
