# Beacon Hackathon Demo Runbook

## Demo Goal

Show Beacon as a local-first crisis guidance assistant:

`question -> hidden Beacon system prompt -> offline official-doc tool -> local Ollama Gemma model -> grounded answer with citations`

The point to say out loud: Ollama runs the local model, but Beacon owns the offline documents, retrieval, tool loop, prompt assembly, and citation policy.

## Current Shipping Model

- Human name: Beacon Tool DPO CPT Fullprompt Ckpt50
- Adapter: `rishavutkarsh/beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter`
- Local Ollama alias: `beacon-gemma4-current-best`
- Base: Gemma 4 E2B IT
- Lineage: Gemma 4 E2B IT -> Beacon CPT checkpoint-300 -> tool-aware full-prompt DPO checkpoint-50

## Pre-Recording Setup

```powershell
$ollama = "$env:LOCALAPPDATA\Programs\OllamaPortable\ollama.exe"
Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
& $ollama list
```

If a fresh DPO GGUF has just been downloaded into `ollama/beacon-gemma4-e2b-current-best-q4_k_m.gguf`, recreate the Ollama model:

```powershell
& $ollama create beacon-gemma4-current-best -f ollama\Modelfile
```

After the Kaggle export finishes, download/install/test the newest GGUF in one step:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\promote_beacon_ollama_export.ps1
```

## Tool Proof

Run this before the model answer to show offline docs are available locally:

```powershell
python scripts\beacon_docs_cli.py list-docs --query "generator carbon monoxide placement" --json
```

Expected top docs include CDC carbon monoxide guidance, Ready.gov/FEMA power outages, and CDC power outage guidance.

## Base Comparison

Base Gemma 4 E2B is available locally as `gemma4:e2b`. For a fair baseline, do not give it Beacon's system prompt, offline docs, or tool loop.

```powershell
& "$env:LOCALAPPDATA\Programs\OllamaPortable\ollama.exe" run gemma4:e2b --think=false
```

Or run one side-by-side comparison:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compare_base_vs_beacon.ps1
```

Narration:

> "Before the model answers, Beacon searches a local offline index of official guidance. This is not a web search; it is a local document tool controlled by the app."

## Main Demo Question

```powershell
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs --num-predict 260 --timeout-seconds 360 "Can I run a generator in my garage if the door is open?"
```

Expected content:

- Do not use a generator in a garage, even with doors or windows open.
- Keep generator at least 20 feet from windows, doors, or vents.
- Cite CDC power outage / CO guidance chunks.

Narration:

> "The user only asks a normal crisis question. Hidden from the user, Beacon passes the system policy and retrieved evidence to the local Gemma model through Ollama. The answer is grounded in the offline documents and includes citations."

## Backup Demo Questions

```powershell
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "How long is food safe in the fridge during a power outage?"
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "What should I do with insulin during a long power outage?"
python scripts\beacon_ollama_agent.py --model beacon-gemma4-current-best --force-docs "How should I disinfect emergency drinking water?"
```

## 3-Minute Video Shape

1. Problem: disaster advice should not rely only on model memory.
2. Architecture: Beacon controller, offline official docs, local Ollama Gemma model.
3. Tool proof: run `beacon_docs_cli.py`.
4. Main answer: run `beacon_ollama_agent.py` for generator/CO.
5. Closing: trained with Unsloth; local-first laptop proof now, mobile packaging later.

## Recording Checklist

1. Terminal at repo root.
2. Show `docs/beacon_demo_runbook.md` or briefly mention the architecture.
3. Run the tool proof command.
4. Run the main demo question.
5. Say the model was fine-tuned with Unsloth and exported through GGUF for local Ollama/llama.cpp serving.
6. Close with impact framing: official-document-grounded crisis guidance for low-connectivity disaster settings.

## What Not To Overclaim

- Do not say Ollama is the full app.
- Do not say this is phone-deployed today.
- Do not say the model independently reads docs; Beacon's controller provides the offline tool.
