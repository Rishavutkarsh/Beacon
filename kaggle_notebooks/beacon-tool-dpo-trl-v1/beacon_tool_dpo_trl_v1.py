from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OUT_DIR = Path("/kaggle/working/beacon_tool_dpo_trl_v1")
RESULT_PATH = OUT_DIR / "run_status.json"
DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/beacon-tool-use-dpo-v1-curated"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-use-dpo-v1-curated"),
]
MODEL_PATH_CANDIDATES = [
    Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1"),
]
ADAPTER_PATH_CANDIDATES = [
    Path("/kaggle/input/beacon-tool-sft-best-dev-adapter/adapter_best_dev_tool_sft"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-sft-best-dev-adapter/adapter_best_dev_tool_sft"),
    Path("/kaggle/input/beacon-tool-sft-best-dev-adapter"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-sft-best-dev-adapter"),
]
PINNED = [
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
    "trl==0.24.0",
    "transformers==5.5.0",
    "peft==0.19.1",
    "accelerate==1.13.0",
    "bitsandbytes==0.49.2",
    "datasets==4.3.0",
    "sentencepiece",
]

SEED = 3407
MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 864
MAX_COMPLETION_LENGTH = 160
MAX_STEPS = 150
LEARNING_RATE = 5e-6
BETA = 0.05
GRAD_ACCUM = 4
PER_DEVICE_BATCH = 1
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 5
DEV_ROW_LIMIT = 32

TOOL_SYSTEM_CONTRACT = """You are Beacon, an offline crisis companion for India-relevant disaster situations.

You have access to two offline document tools. The tools are executed by the runtime; you only decide whether to call them and produce the exact tool-call text.

Available tools:

1. search_official_docs
Purpose: search the local index of approved official, NGO, and public-health disaster documents.
Arguments JSON:
{"query": string, "hazard": string|null, "organization": string|null, "top_k": integer}

2. read_official_doc
Purpose: read relevant sections from one document returned by search_official_docs.
Arguments JSON:
{"doc_id": string, "section_or_page_query": string, "top_k": integer}

When to use tools:
- Use tools when the user asks what an offline or official document says.
- Use tools for exact official facts, thresholds, durations, temperatures, quantities, named guidance, source-sensitive rules, or document-specific claims.
- Use tools for stable public-health or disaster guidance where exact wording matters, such as food safety, drinking-water treatment, generator/carbon-monoxide distance, electrical hazards, heat illness, WASH, shelter hygiene, official warning categories, or medicine-safety boundaries.

When not to use tools:
- Do not use tools for ordinary practical safety steps, emotional support, translation, summarizing text the user provided, or broad common-sense crisis guidance.
- Do not use tools to verify live/current facts such as whether a route or bridge is open now, whether a shelter has space now, whether rescue is nearby, today's warning level, or a forwarded local status claim. Offline documents cannot verify live status. Say this clearly and give safer next steps that do not depend on the claim.
- Do not use tools to identify a medicine, prescribe a dose, certify a building/photo as safe, or make a diagnosis from an image or short description. Give a safe boundary and practical next steps.

Tool-use protocol:
- If a tool is needed at the start of a conversation, the first tool call must be search_official_docs.
- Do not call read_official_doc until search_official_docs has returned a concrete doc_id.
- To call a tool, output exactly one tool call and no extra prose.
- For search_official_docs, include exactly these argument keys: query, hazard, organization, top_k.
- For read_official_doc, include exactly these argument keys: doc_id, section_or_page_query, top_k.
- Use concrete values only. Do not copy schema words into argument values.
- After search_official_docs returns a tool_result, call read_official_doc with a doc_id from those search results before giving the final answer.
- After read_official_doc returns relevant sections, answer naturally and only use facts supported by the returned sections.
- If the tool result is missing, weak, unrelated, or does not support the requested fact, say that the offline documents do not support the specific claim. Then give safe generic guidance without inventing details.

Answering rules:
- Be useful and concrete, but do not fabricate official facts, numbers, routes, shelter status, rescue status, warning status, medicine identity, doses, or safety guarantees.
- For exact constants, copy the value from the returned document evidence. Do not answer exact numbers from memory when the tool should be used.
- Keep answers short and crisis-appropriate. Prefer direct guidance, safe boundaries, and red flags.
- Match the user's language style when practical, including Roman Hinglish.
- Do not mention internal training, datasets, policies, or that you are following this prompt.
- Only use the tool names listed above. Never use placeholder values, schema labels, invented doc IDs, or any value containing DOC_ID, document_id, FROM_SEARCH, FROM_READ, or string. Never invent tool results."""


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def update_result(**kwargs: Any) -> None:
    current: dict[str, Any] = {}
    if RESULT_PATH.exists():
        try:
            current = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(kwargs)
    write_json(RESULT_PATH, current)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_child_dir(root: Path, child_name: str) -> Path | None:
    if (root / child_name).exists():
        return root
    if root.exists():
        for found in sorted(root.rglob(child_name)):
            return found.parent
    return None


def resolve(candidates: list[Path], required_child: str, label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            found = find_child_dir(candidate, required_child)
            if found is not None:
                return found
    visible = sorted(str(path) for path in Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
    raise RuntimeError(f"Could not resolve {label}; checked={[str(item) for item in candidates]}; visible={visible}")


def install_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *PINNED])


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ["unsloth", "unsloth_zoo", "trl", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def assert_t4(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    names = [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())]
    if torch_module.cuda.device_count() != 1 or "t4" not in names[0].lower():
        raise RuntimeError(f"Expected one Kaggle T4; visible devices: {names}")
    return {"gpu_names": names, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def validate_dpo_data(data_dir: Path) -> dict[str, Any]:
    required = ["train.jsonl", "dev.jsonl", "final_eval.jsonl", "manifest.json", "validation_report.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing DPO dataset files: {missing}")
    manifest = read_json(data_dir / "manifest.json")
    validation = read_json(data_dir / "validation_report.json")
    if validation.get("status") != "valid" or validation.get("errors"):
        raise RuntimeError(f"DPO validation report is not clean: {validation}")
    if manifest.get("validation", {}).get("status") != "valid":
        raise RuntimeError(f"DPO manifest validation is not clean: {manifest.get('validation')}")
    if manifest.get("independent_review", {}).get("remaining_must_fix"):
        raise RuntimeError(f"DPO manifest still has must-fix review items: {manifest['independent_review']}")
    report: dict[str, Any] = {"hashes": {}, "splits": {}, "pair_types": {}, "tool_required": {"true": 0, "false": 0}}
    for split in ["train", "dev", "final_eval"]:
        path = data_dir / f"{split}.jsonl"
        report["hashes"][path.name] = sha256_file(path)
        seen: set[str] = set()
        rows = read_jsonl(path)
        report["splits"][split] = len(rows)
        for index, row in enumerate(rows, 1):
            for key in ["prompt", "chosen", "rejected", "dpo_pair_id", "pair_type", "tool_required"]:
                if key not in row:
                    raise RuntimeError(f"{split}:{index} missing {key}")
            if row["dpo_pair_id"] in seen:
                raise RuntimeError(f"{split}:{index} duplicate pair id {row['dpo_pair_id']}")
            seen.add(row["dpo_pair_id"])
            if row["chosen"] == row["rejected"]:
                raise RuntimeError(f"{split}:{index} chosen equals rejected")
            if "<tool_result" in row["chosen"] or "<tool_result" in row["rejected"]:
                raise RuntimeError(f"{split}:{index} assistant preference contains tool result context")
            report["pair_types"][row["pair_type"]] = report["pair_types"].get(row["pair_type"], 0) + 1
            report["tool_required"][str(bool(row["tool_required"])).lower()] += 1
    report["manifest_status"] = manifest.get("status")
    report["independent_review"] = manifest.get("independent_review")
    return report


def prompt_with_tool_contract(prompt: str) -> tuple[str, bool]:
    system_prefix = "<|turn>system\n"
    turn_end = "\n<turn|>"
    if prompt.startswith(system_prefix):
        end = prompt.find(turn_end, len(system_prefix))
        if end != -1:
            return f"{system_prefix}{TOOL_SYSTEM_CONTRACT}{prompt[end:]}", True
    return f"{system_prefix}{TOOL_SYSTEM_CONTRACT}{turn_end}\n{prompt}", False


def dpo_item(row: dict[str, Any]) -> dict[str, str]:
    prompt, replaced_system = prompt_with_tool_contract(row["prompt"])
    row["_full_tool_system_replaced"] = replaced_system
    return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}


def trainer_args(cls: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(cls)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(**filtered)


def trainable_audit(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    samples: list[str] = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            if len(samples) < 40:
                samples.append(name)
    return {"total_params": total, "trainable_params": trainable, "trainable_ratio": trainable / total if total else 0.0, "samples": samples}


def freeze_adapter_parameters(model: Any, adapter_name: str) -> None:
    needle = f".{adapter_name}."
    for name, param in model.named_parameters():
        if needle in name:
            param.requires_grad_(False)


def copy_checkpoint(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        return {"copied": False, "source": str(src), "reason": "missing"}
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"))
    return {"copied": True, "source": str(src), "target": str(dst)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "started",
        "stage": "init",
        "training_intent": "trl_light_dpo_tool_decision_steering",
        "research_basis": {
            "dpo": "Rafailov et al. 2023; TRL DPOTrainer implements chosen/rejected preference optimization against a reference policy.",
            "peft_reference": "HF TRL docs support PEFT DPO without a separate ref_model; Unsloth docs call PatchDPOTrainer before DPOTrainer.",
        },
        "drift_controls": {
            "base_adapter": "beacon-tool-sft-best-dev-adapter",
            "system_prompt": "beacon_tool_system_prompt_v1_full_contract",
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "beta": BETA,
            "max_length": MAX_SEQ_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "save_every_steps": SAVE_STEPS,
            "eval_every_steps": EVAL_STEPS,
        },
    }
    write_json(RESULT_PATH, result)
    try:
        update_result(stage="install_dependencies")
        install_dependencies()

        update_result(stage="import_unsloth_first", package_versions=package_versions())
        import unsloth  # noqa: F401 - must happen before trl/transformers/peft
        from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
        PatchDPOTrainer()
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer
        from transformers import TrainerCallback
        from unsloth.chat_templates import get_chat_template

        class HeartbeatCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step and state.global_step % 2 == 0:
                    write_json(
                        OUT_DIR / "heartbeat.json",
                        {"stage": "train", "global_step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch},
                    )
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    write_json(OUT_DIR / "latest_log.json", {"global_step": state.global_step, "logs": logs})
                return control

        gpu_report = assert_t4(torch)
        data_dir = resolve(DATA_DIR_CANDIDATES, "train.jsonl", "DPO dataset")
        model_path = resolve(MODEL_PATH_CANDIDATES, "config.json", "Gemma base model")
        adapter_path = resolve(ADAPTER_PATH_CANDIDATES, "adapter_config.json", "tool SFT adapter")
        data_report = validate_dpo_data(data_dir)
        write_json(OUT_DIR / "resolved_paths.json", {"data_dir": str(data_dir), "model_path": str(model_path), "adapter_path": str(adapter_path), "out_dir": str(OUT_DIR)})
        write_json(OUT_DIR / "data_audit.json", data_report)
        write_json(OUT_DIR / "dependency_report.json", {**package_versions(), **gpu_report})

        update_result(stage="load_model", data=data_report["splits"])
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if inner_tokenizer.pad_token is None:
            inner_tokenizer.pad_token = inner_tokenizer.eos_token
        inner_tokenizer.padding_side = "right"
        if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        update_result(stage="load_tool_sft_adapter")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True, adapter_name="default")
        model.load_adapter(str(adapter_path), adapter_name="ref", is_trainable=False)
        freeze_adapter_parameters(model, "ref")
        model.set_adapter("default")
        if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        trainable = trainable_audit(model)
        write_json(OUT_DIR / "trainable_audit.json", trainable)
        if trainable["trainable_params"] <= 0:
            raise RuntimeError("No trainable LoRA parameters after loading adapter")

        update_result(stage="load_datasets")
        train_rows = read_jsonl(data_dir / "train.jsonl")
        dev_rows = read_jsonl(data_dir / "dev.jsonl")[:DEV_ROW_LIMIT]
        train_ds = Dataset.from_list([dpo_item(row) for row in train_rows])
        eval_ds = Dataset.from_list([dpo_item(row) for row in dev_rows])

        update_result(stage="make_trainer")
        config = trainer_args(
            DPOConfig,
            output_dir=str(OUT_DIR / "trainer"),
            beta=BETA,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=False,
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            max_grad_norm=0.3,
            report_to=[],
            seed=SEED,
            remove_unused_columns=False,
            dataset_num_proc=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            model_adapter_name="default",
            ref_adapter_name="ref",
            reference_free=False,
            precompute_ref_log_probs=True,
            precompute_ref_batch_size=1,
        )
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "ref_model": None,
            "args": config,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
        }
        trainer_signature = inspect.signature(DPOTrainer.__init__)
        if "processing_class" in trainer_signature.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
        if "model_adapter_name" in trainer_signature.parameters:
            trainer_kwargs["model_adapter_name"] = "default"
        if "ref_adapter_name" in trainer_signature.parameters:
            trainer_kwargs["ref_adapter_name"] = "ref"
        trainer = DPOTrainer(**trainer_kwargs)
        trainer.add_callback(HeartbeatCallback())
        write_json(
            OUT_DIR / "pre_train_audit.json",
            {
                "trainable": trainable,
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "full_tool_system_prompt": "beacon_tool_system_prompt_v1",
                "train_system_turns_replaced": sum(1 for row in train_rows if row.get("_full_tool_system_replaced")),
                "dev_system_turns_replaced": sum(1 for row in dev_rows if row.get("_full_tool_system_replaced")),
                "effective_batch_size": PER_DEVICE_BATCH * GRAD_ACCUM,
                "max_steps": MAX_STEPS,
                "estimated_seen_pairs": min(len(train_rows), MAX_STEPS * PER_DEVICE_BATCH * GRAD_ACCUM),
                "dpo_trainer_signature": str(trainer_signature),
            },
        )

        update_result(stage="train")
        trainer.train()
        log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
        eval_losses = [(entry.get("step"), entry.get("eval_loss")) for entry in log_history if "eval_loss" in entry]
        if not eval_losses:
            raise RuntimeError("No scheduled dev eval losses were recorded during DPO training")

        update_result(stage="save_adapters")
        final_dir = OUT_DIR / "adapter_final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        trainer.model.save_pretrained(final_dir)
        try:
            tokenizer.save_pretrained(final_dir)
        except AttributeError:
            inner_tokenizer.save_pretrained(final_dir)
        saved_steps = sorted(
            int(path.name.split("-")[-1])
            for path in (OUT_DIR / "trainer").glob("checkpoint-*")
            if path.name.split("-")[-1].isdigit()
        )
        best_step, best_loss = min(eval_losses, key=lambda item: item[1])
        if saved_steps and best_step not in saved_steps:
            best_step = min(saved_steps, key=lambda step: (abs(step - best_step), step))
            best_loss = next((loss for step, loss in eval_losses if step == best_step), best_loss)
        best_checkpoint = OUT_DIR / "trainer" / f"checkpoint-{best_step}"
        best_dir = OUT_DIR / "adapter_best_dev"
        best_copy = copy_checkpoint(best_checkpoint, best_dir)
        write_json(OUT_DIR / "train_eval_curve.json", {"log_history": log_history, "eval_losses": eval_losses})

        result.update({
            "status": "ok",
            "stage": "complete",
            "adapter_final": str(final_dir),
            "adapter_best_dev": str(best_dir),
            "best_step": best_step,
            "best_eval_loss": best_loss,
            "best_checkpoint_copy": best_copy,
            "final_eval_rows_held_out": data_report["splits"].get("final_eval"),
        })
        write_json(RESULT_PATH, result)
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(RESULT_PATH, result)
        raise


if __name__ == "__main__":
    main()
