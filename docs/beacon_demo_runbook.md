# Beacon Hackathon Demo Runbook

## Demo Goal

Show Beacon as a local-first crisis guidance assistant and contrast it with base Gemma:

`question -> hidden Beacon system prompt -> offline official-doc tool loop -> local Ollama Gemma model -> grounded answer with citations`

The point to say out loud: Base Gemma answers from model memory. Beacon uses the same local serving stack, but adds the offline documents, retrieval, tool loop, prompt assembly, and citation policy.

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

Run a side-by-side comparison. This prints the same user question once for base Gemma and once for Beacon with offline official-doc evidence:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compare_base_vs_beacon.ps1
```

For the strongest contrast, use the floodwater/baby-feeding-item myth prompt:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compare_base_vs_beacon.ps1 -Question "Floodwater touch hue baby bottle nipples/pacifiers ko bleach se sanitize karke use kar sakte hain?"
```

Expected contrast:

- Base may generalize from bleach/disinfection knowledge and say sanitizing is possible.
- Beacon should call/search/read offline official docs and say no: throw out baby bottle nipples and pacifiers touched by floodwater because sanitizing methods are not effective.
- Beacon should cite `cdc_food_after_emergency / cdc_food_after_emergency_chunk_0002`.

By default the comparison script uses preloaded offline evidence for demo stability. Say clearly that this is the same offline evidence path preloaded by the controller for a reliable recording.

To demonstrate the learned canonical tool-loop live instead, use `-ToolLoop`. This does not force a tool call; it lets the model decide, which is the behavior we trained:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compare_base_vs_beacon.ps1 -ToolLoop -Question "Floodwater touch hue baby bottle nipples/pacifiers ko bleach se sanitize karke use kar sakte hain?"
```

Narration:

> "The first answer is base Gemma without our tool contract. The second answer is Beacon: it has access to explicit offline hazard keywords and must use the local official-doc tool before answering exact safety claims. This is not a web search; it is a local document tool controlled by the app."

## Main Demo Question

```powershell
python scripts\beacon_ollama_tool_loop_agent.py --model beacon-gemma4-current-best --num-predict 260 --timeout-seconds 360 "Can I run a generator in my garage if the door is open?"
```

Expected content:

- Do not use a generator in a garage, even with doors or windows open.
- Keep generator at least 20 feet from windows, doors, or vents.
- Cite CDC power outage / CO guidance chunks.

Narration:

> "The user only asks a normal crisis question. Hidden from the user, Beacon passes the system policy and retrieved evidence to the local Gemma model through Ollama. The answer is grounded in the offline documents and includes citations."

## Backup Demo Questions

```powershell
python scripts\beacon_ollama_tool_loop_agent.py --model beacon-gemma4-current-best "How long is food safe in the fridge during a power outage?"
python scripts\beacon_ollama_tool_loop_agent.py --model beacon-gemma4-current-best "What should I do with insulin during a long power outage?"
python scripts\beacon_ollama_tool_loop_agent.py --model beacon-gemma4-current-best "How should I disinfect emergency drinking water?"
python scripts\beacon_ollama_tool_loop_agent.py --model beacon-gemma4-current-best "Is the bridge open now after flood?"
```

For the live-status question, expected behavior is a refusal to verify current status from offline docs, plus safer next steps.

## 3-Minute Video Shape

1. Problem: disaster advice should not rely only on model memory.
2. Architecture: Beacon controller, offline official docs, local Ollama Gemma model.
3. Tool proof: run `beacon_docs_cli.py`.
4. Main answer: run `compare_base_vs_beacon.ps1` for the floodwater baby-item myth or generator/CO.
5. Closing: trained with Unsloth; local-first laptop proof now, mobile packaging later.

## Recording Checklist

1. Terminal at repo root.
2. Show `docs/beacon_demo_runbook.md` or briefly mention the architecture.
3. Run the tool proof command.
4. Run the side-by-side comparison script.
5. Say the model was fine-tuned with Unsloth and exported through GGUF for local Ollama/llama.cpp serving.
6. Close with impact framing: official-document-grounded crisis guidance for low-connectivity disaster settings.

## What Not To Overclaim

- Do not say Ollama is the full app.
- Do not say this is phone-deployed today.
- Do not say the model independently reads docs; Beacon's controller provides the offline tool.
- The default side-by-side script uses controller-preloaded offline evidence for demo stability. Use `-ToolLoop` only when you want to show learned tool-call behavior live.
- Do not use `--force-docs` for normal demos. It is a debugging/stress-test flag for tool-required cases, not product behavior.
