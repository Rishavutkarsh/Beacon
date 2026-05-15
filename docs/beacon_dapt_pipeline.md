# Beacon DAPT Pipeline

This is the practical first-run CPT/DAPT data pipeline for Beacon. It collects official/reputable crisis text, packages it into document-level train/dev splits, validates the package, and stops before training.

## Files

- `scripts/download_beacon_dapt_sources.py`: bounded downloader/extractor for curated Beacon source lists.
- `scripts/package_beacon_dapt_corpus.py`: merge, dedupe, split, validate, and emit train-ready DAPT JSONL.
- `data/dapt_corpus/beacon_crisis_v1_download/`: downloaded raw/extracted source material and download manifest.
- `data/dapt_corpus/beacon_crisis_v1_train_ready/`: final train-ready package.

## Build

```powershell
python scripts\download_beacon_dapt_sources.py --source-list data\dapt_corpus\beacon_crisis_v1_download\source_list.jsonl --append
python scripts\download_beacon_dapt_sources.py --source-list data\dapt_corpus\beacon_crisis_v1_download\source_list_extra.jsonl --append
python scripts\download_beacon_dapt_sources.py --source-list data\dapt_corpus\beacon_crisis_v1_download\source_list_large2.jsonl --append
python scripts\download_beacon_dapt_sources.py --source-list data\dapt_corpus\beacon_crisis_v1_download\source_list_large3.jsonl --append
python scripts\download_beacon_dapt_sources.py --source-list data\dapt_corpus\beacon_crisis_v1_download\source_list_large4.jsonl --append
python scripts\download_beacon_dapt_sources.py --source-list data\dapt_corpus\beacon_crisis_v1_download\source_list_large5.jsonl --append
python scripts\package_beacon_dapt_corpus.py build
python scripts\package_beacon_dapt_corpus.py validate
```

## Current Package

- Output: `data/dapt_corpus/beacon_crisis_v1_train_ready/`
- Status: `dapt_ready=true`
- Estimated tokens: `2,131,792`
- Rows: `1,415`
- Train rows: `1,361`
- Dev rows: `54`
- Documents: `233`

## Training Hand-Off

Use:

- `dapt_train.jsonl` for CPT/DAPT training.
- `dapt_dev.jsonl` only for validation/perplexity-style monitoring.
- `training_config.json` as a draft, not a launch decision.

Training should not start until the DAPT variables are reviewed with the user.

## Kaggle CPT Package

The Kaggle-ready raw CPT package is:

```text
data/dapt_corpus/beacon_crisis_v1_cpt_kaggle/
  cpt_train.jsonl
  cpt_dev.jsonl
  cpt_test.jsonl
  cpt_training_config.json
  cpt_split_manifest.json
  cpt_package_manifest.json
  cpt_validation_report.json
  dataset-metadata.json
```

Current validation:

- `cpt_train`: `1361` rows, `2,052,878` estimated tokens
- `cpt_dev`: `23` rows, `45,744` estimated tokens
- `cpt_test`: `31` rows, `33,170` estimated tokens
- total: `2,131,792` estimated tokens
- validation: `pass`

Upload when explicitly approved:

```powershell
kaggle datasets create -p data\dapt_corpus\beacon_crisis_v1_cpt_kaggle
```

If the dataset already exists:

```powershell
kaggle datasets version -p data\dapt_corpus\beacon_crisis_v1_cpt_kaggle -m "Refresh Beacon crisis CPT v1"
```

## Kaggle Kernels

Training kernel:

```text
kaggle/beacon_dapt_cpt_train/
  beacon_cpt_train.py
  kernel-metadata.json
```

Eval kernel:

```text
kaggle/beacon_dapt_cpt_eval/
  beacon_cpt_eval.py
  kernel-metadata.json
```

Both kernels require Kaggle T4:

```powershell
kaggle kernels push -p kaggle\beacon_dapt_cpt_train --accelerator NvidiaTeslaT4
kaggle kernels push -p kaggle\beacon_dapt_cpt_eval --accelerator NvidiaTeslaT4
```

The train script follows the raw CPT helper:

- raw `text` only
- no chat template
- no response masking
- hash-gated dataset package
- Unsloth imported before TRL/PEFT trainer code
- single visible T4
- QLoRA attention + MLP targets under `language_model`
- baseline dev loss before training
- best checkpoint selected by dev loss
- test reserved for post-training evaluation
