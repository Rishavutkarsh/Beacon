# Beacon Ollama Local Demo Pipeline

Beacon keeps the app/controller separate from local model serving:

`user question -> Beacon controller -> offline official-doc tool -> Ollama model -> grounded answer with citations`

Ollama is only the laptop-local model runner. Beacon owns the official document index, retrieval, prompt assembly, and citation policy.
The Beacon controller sends its safety/tool instructions as a hidden Ollama `system` prompt before every user question. The user does not type or see that prompt in the demo.

## Current Artifacts

- Base model on Kaggle: `/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1`
- Best domain adapter: `kaggle/beacon_dapt_cpt_v2_checkpoints/checkpoint-300`
- Best behavior adapter for demo: `Beacon Tool DPO CPT Fullprompt Ckpt50`
- Kaggle dataset ref for behavior adapter: `rishavutkarsh/beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter`
- Offline doc index: `data/local_grounding/official_doc_tool_v1`
- Existing lower-level doc query script: `scripts/query_beacon_official_doc_tool.py`
- Current model pointer: `config/beacon_current_model.json`
- Kaggle export kernel: `kaggle/beacon_ollama_export_current_best`
- Ollama controller: `scripts/beacon_ollama_agent.py`
- Docs CLI: `scripts/beacon_docs_cli.py`
- Demo smoke script: `scripts/beacon_demo_smoke.ps1`

The current demo adapter is the tool-aware DPO checkpoint selected for shipping. It continues the Beacon CPT checkpoint-300 lineage and is exported against the Gemma 4 E2B base. SFT v2 final should not be used.

## Export GGUF On Kaggle

Run this in a Kaggle notebook/kernel with the base model and best adapter attached:

```bash
python -m pip install --no-cache-dir unsloth==2026.5.2 unsloth_zoo==2026.5.1 \
  transformers==5.5.0 peft==0.19.1 accelerate==1.13.0 bitsandbytes==0.49.2 sentencepiece

python scripts/export_beacon_ollama.py \
  --base-model /kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1 \
  --adapter /kaggle/input/beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter \
  --out-dir /kaggle/working/beacon_ollama_export \
  --quantization q4_k_m
```

Download the produced `.gguf`, place it beside `ollama/Modelfile`, and name it:

```text
ollama/beacon-gemma4-e2b-current-best-q4_k_m.gguf
```

If the filename differs, update the `FROM` line in `ollama/Modelfile`.

## Create And Run The Ollama Model

Install Ollama, then from this repo:

```powershell
ollama create beacon-gemma4-current-best -f ollama\Modelfile
ollama serve
```

On this Windows machine, Ollama was installed as a portable binary at:

```powershell
$ollama = "$env:LOCALAPPDATA\Programs\OllamaPortable\ollama.exe"
Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
& $ollama create beacon-gemma4-current-best -f ollama\Modelfile
```

In a second terminal:

```powershell
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "Can I run a generator in my garage if the door is open?"
```

The controller exposes the offline document tool to the model in two ways:

- For source-sensitive demo questions, `--force-docs` retrieves official document sections before the answer and passes them as `OFFLINE_DOC_EVIDENCE`.
- Without `--force-docs`, the hidden system prompt allows the model to request a structured tool call such as `<tool_call>{"name":"search_official_docs","arguments":{"query":"generator carbon monoxide"}}</tool_call>`, which the controller executes locally before asking the model for the final answer.

For a one-command pre-demo check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\beacon_demo_smoke.ps1
```

## Offline Doc Tool Smoke Tests

```powershell
python scripts\beacon_docs_cli.py list-docs --query "generator carbon monoxide placement" --json
python scripts\beacon_docs_cli.py read-doc --doc-id cdc_co_clinical_disasters --section-or-query "generator carbon monoxide placement" --json
```

## Demo Questions

```powershell
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "Can I run a generator in my garage if the door is open?"
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "How long is food safe in the fridge during a power outage?"
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "What should I do with insulin during a long power outage?"
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "Is floodwater safe to walk through if it looks shallow?"
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "How should I disinfect emergency drinking water?"
```

## Status And Blockers

Complete locally:

- Offline docs CLI over the existing Beacon official-doc index.
- Ollama controller with deterministic retrieval and a simple structured tool-call loop.
- Modelfile for the expected Q4_K_M GGUF.
- Kaggle/HF-style export script for merging the selected adapter and producing GGUF.
- Kaggle kernel package for exporting the current best adapter directly on Kaggle.
- Portable Ollama install and local import as `beacon-gemma4-current-best`.
- Grounded controller smoke test with the local Ollama model.

Kaggle run note:

- `rishavutkarsh/beacon-ollama-export-current-best` version 10 completed successfully on May 18, 2026.
- Export backend: direct fresh clone of `ggml-org/llama.cpp` after Unsloth merge.
- Local GGUF: `ollama/beacon-gemma4-e2b-current-best-q4_k_m.gguf`
- Size: `3427863744` bytes.
- SHA256: `a23397660c6231f679b32d214570689adf78ef13592db40c55df24686009fd44`

The successful path was:

1. Load Gemma 4 E2B IT with Unsloth.
2. Load Beacon Tool DPO CPT Fullprompt Ckpt50 adapter.
3. Explicitly `merge_and_unload()` the LoRA adapter before saving HF weights.
4. Uninstall Kaggle's incompatible `torchvision` before converter subprocesses.
5. Clone current `ggml-org/llama.cpp`, convert merged HF to F16 GGUF, then quantize with `llama-quantize Q4_K_M`.

Implementation notes:

- The controller sends Ollama API requests with top-level `"think": false`; otherwise this Gemma 4 E2B import can spend the response budget in Ollama's thinking channel and return an empty final response.
- Generated `.gguf` binaries are ignored by git. Keep the local file in `ollama/` for demos, but distribute it through Kaggle/HF artifacts rather than source control.
- If a better adapter is selected later, update `config/beacon_current_model.json`, rerun `scripts/export_beacon_ollama.py` on Kaggle/HF, replace the local GGUF, and rerun `ollama create`.
