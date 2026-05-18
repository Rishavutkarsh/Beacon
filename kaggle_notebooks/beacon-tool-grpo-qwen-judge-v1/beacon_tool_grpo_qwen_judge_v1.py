from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections import Counter
import inspect
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OUT_DIR = Path("/kaggle/working/beacon_tool_grpo_qwen_judge_v1")
RESULT_PATH = OUT_DIR / "run_status.json"
DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/beacon-tool-use-grpo-v1"),
    Path("/kaggle/input/beacon-tool-use-grpo-v1/beacon_tool_use_grpo_v1"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-use-grpo-v1"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-use-grpo-v1/beacon_tool_use_grpo_v1"),
]
MODEL_PATH_CANDIDATES = [
    Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1"),
]
SFT_ADAPTER_CANDIDATES = [
    Path("/kaggle/input/beacon-tool-sft-best-dev-adapter"),
    Path("/kaggle/input/beacon-tool-sft-best-dev-adapter/adapter_best_dev_tool_sft"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-sft-best-dev-adapter"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-tool-sft-best-dev-adapter/adapter_best_dev_tool_sft"),
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
TRAIN_ROW_LIMIT = None
DEV_ROW_LIMIT = 64
MAX_SEQ_LENGTH = 768
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 192
MAX_STEPS = 150
NUM_GENERATIONS = 2
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-6
USE_OPTIONAL_QWEN_JUDGE = False
TRAINING_OBJECTIVE = "full_first_turn_tool_decision_and_search"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
EXACT_CLAIM_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:hours?|minutes?|days?|feet|degrees?|grams?|g|percent)|"
    r"40\s*degrees|20\s*feet|15\s*(?:grams?|g)|1\s*minute|3\s*minutes|30\s*minutes)\b",
    re.I,
)
LIVE_STATUS_RE = re.compile(r"\b(open now|safe now|current|right now|available now|rescue is near|verified now)\b", re.I)
ABSTAIN_RE = re.compile(r"\b(offline documents? (?:do|does) not support|cannot verify|not enough evidence|unverified)\b", re.I)


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
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")


def update_result(**kwargs: Any) -> None:
    current: dict[str, Any] = {}
    if RESULT_PATH.exists():
        try:
            current = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(kwargs)
    write_json(RESULT_PATH, current)


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


def resolve(candidates: list[Path], required_child: str, label: str, required: bool = True) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            found = find_child_dir(candidate, required_child)
            if found is not None:
                return found
    if required:
        visible = sorted(str(path) for path in Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
        raise RuntimeError(f"Could not resolve {label}; checked={[str(item) for item in candidates]}; visible={visible}")
    return None


def install_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", *PINNED])


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ["unsloth", "unsloth_zoo", "trl", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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


def validate_dataset(data_dir: Path) -> dict[str, Any]:
    required = ["train.jsonl", "dev.jsonl", "final_eval.jsonl", "manifest.json", "validation_report.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing GRPO dataset files: {missing}")
    manifest = read_json(data_dir / "manifest.json")
    validation = read_json(data_dir / "validation_report.json")
    if validation.get("status") != "valid" or validation.get("errors"):
        raise RuntimeError(f"GRPO validation report is not clean: {validation}")
    report: dict[str, Any] = {"splits": {}, "families": {}, "tool_required": Counter()}
    for split in ["train", "dev", "final_eval"]:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        report["splits"][split] = len(rows)
        for index, row in enumerate(rows, 1):
            for key in ["prompt", "tool_required", "expected_facts", "allowed_doc_ids", "target_response", "reward_rubric"]:
                if key not in row:
                    raise RuntimeError(f"{split}:{index} missing {key}")
            if "<tool_result" in row["prompt"]:
                raise RuntimeError(f"{split}:{index} prompt leaks tool result")
            report["families"][row.get("row_family", "")] = report["families"].get(row.get("row_family", ""), 0) + 1
            report["tool_required"][str(bool(row["tool_required"])).lower()] += 1
    report["tool_required"] = dict(report["tool_required"])
    report["manifest_status"] = manifest.get("status")
    report["training_stage_intent"] = manifest.get("training_stage_intent")
    return report


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], int]:
    calls: list[dict[str, Any]] = []
    errors = 0
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            errors += 1
            continue
        if not isinstance(payload, dict):
            errors += 1
            continue
        calls.append(payload)
    if "<tool_call>" in text and not calls:
        errors += 1
    return calls, errors


def extra_text_outside_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def has_placeholder_tool_arg(value: Any) -> bool:
    lowered = str(value).lower()
    return any(token in lowered for token in ["doc_id", "document_id", "from_search", "from_read", "string", "..."])


def token_set(text: str) -> set[str]:
    return {item for item in re.findall(r"[a-z0-9]+", text.lower()) if len(item) > 2}


def search_quality_reward(query: str, gold_query: str, expected_facts: list[str], user_prompt: str) -> float:
    if not query:
        return 0.0
    query_terms = token_set(query)
    gold_terms = token_set(gold_query) | token_set(" ".join(expected_facts)) | token_set(user_prompt)
    if not query_terms or not gold_terms:
        return 0.0
    overlap = len(query_terms & gold_terms) / max(1, min(len(query_terms), len(gold_terms)))
    reward = 0.0
    if overlap >= 0.25:
        reward += 1.0
    if overlap >= 0.45:
        reward += 1.0
    if any(str(fact).lower() in query.lower() for fact in expected_facts):
        reward += 1.0
    return min(3.0, reward)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def deterministic_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    tool_required_values = kwargs.get("tool_required", [])
    row_family_values = kwargs.get("row_family", [])
    expected_facts_values = kwargs.get("expected_facts", [])
    allowed_doc_ids_values = kwargs.get("allowed_doc_ids", [])
    gold_tool_query_values = kwargs.get("gold_tool_query", [])
    user_prompt_values = kwargs.get("user_prompt", [])
    for index, completion in enumerate(completions):
        text = completion_text(completion)
        calls, parse_errors = parse_tool_calls(text)
        tool_required = bool(tool_required_values[index])
        row_family = str(row_family_values[index])
        expected_facts = expected_facts_values[index] or []
        allowed_doc_ids = set(allowed_doc_ids_values[index] or [])
        gold_tool_query = str(gold_tool_query_values[index] or "")
        user_prompt = str(user_prompt_values[index] or "")
        score = 0.0

        if parse_errors:
            score -= 3.0 * parse_errors
        if any(call.get("name") not in {"search_official_docs", "read_official_doc"} for call in calls):
            score -= 3.0
        if calls:
            extra = extra_text_outside_tool_calls(text)
            if extra:
                score -= 3.0
            for call in calls:
                args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
                if any(has_placeholder_tool_arg(value) for value in args.values()):
                    score -= 4.0
        if LIVE_STATUS_RE.search(text) and not ABSTAIN_RE.search(text):
            score -= 5.0

        if not tool_required:
            score += 2.0 if not calls else -3.0
            if len(text.strip()) >= 40 and not calls:
                score += 1.0
            rewards.append(float(max(-8.0, min(8.0, score))))
            continue

        if not calls:
            exact_claims = EXACT_CLAIM_RE.findall(text)
            score -= 4.0
            if exact_claims:
                score -= 2.0
            rewards.append(float(max(-8.0, min(8.0, score))))
            continue

        names = [str(call.get("name")) for call in calls]
        if names[0] != "search_official_docs":
            score -= 3.0
        else:
            search_args = calls[0].get("arguments", {}) if isinstance(calls[0].get("arguments"), dict) else {}
            score += 1.0
            score += search_quality_reward(str(search_args.get("query", "")), gold_tool_query, expected_facts, user_prompt)
        if len(calls) > 1:
            score -= 1.0 * (len(calls) - 1)
        if "read_official_doc" in names:
            score -= 2.0

        lower_text = text.lower()
        if "no_support" in row_family:
            if calls and not ABSTAIN_RE.search(text):
                score += 0.5
            else:
                score += 4.0 if ABSTAIN_RE.search(text) else -3.0
        elif expected_facts:
            if not calls or any(name not in {"search_official_docs", "read_official_doc"} for name in names):
                present = sum(1 for fact in expected_facts if str(fact).lower() in lower_text)
                score += 4.0 * (present / max(1, len(expected_facts)))
        if re.search(r"\b(dose|diagnose|guarantee safe|definitely safe|verified open)\b", lower_text):
            score -= 8.0
        rewards.append(float(max(-8.0, min(10.0, score))))
    return rewards


def optional_qwen_judge_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    # Keep this out of the default T4 training loop. Run it offline or on a second
    # pass when memory is known safe. It should return strict JSON verdicts:
    # answer_correct, grounded_in_evidence, unsupported_extra, unsafe,
    # abstention_correct, notes.
    return [0.0 for _ in completions]


def build_grpo_config(GRPOConfig: Any) -> tuple[Any, dict[str, Any]]:
    requested = {
        "output_dir": str(OUT_DIR / "trainer"),
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_generations": NUM_GENERATIONS,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "logging_steps": 1,
        "save_steps": 50,
        "save_total_limit": 3,
        "report_to": "none",
        "bf16": False,
        "fp16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "use_vllm": False,
        "beta": 0.02,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": SEED,
    }
    signature = inspect.signature(GRPOConfig.__init__)
    accepted_keys = set(signature.parameters)
    accepted = {key: value for key, value in requested.items() if key in accepted_keys}
    dropped = {key: value for key, value in requested.items() if key not in accepted_keys}
    critical = ["num_generations"]
    missing_critical = [key for key in critical if key not in accepted]
    update_result(
        grpo_config_signature=str(signature),
        grpo_requested_config=requested,
        grpo_accepted_config=accepted,
        grpo_dropped_config=dropped,
    )
    if missing_critical:
        raise RuntimeError(f"GRPOConfig does not accept critical memory controls: {missing_critical}; refusing to run with unsafe defaults.")
    return GRPOConfig(**accepted), dropped


def build_grpo_trainer(GRPOTrainer: Any, trainer_kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(GRPOTrainer.__init__)
    accepted_keys = set(signature.parameters)
    if "processing_class" not in accepted_keys and "tokenizer" in accepted_keys and "processing_class" in trainer_kwargs:
        trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
    accepted = {key: value for key, value in trainer_kwargs.items() if key in accepted_keys}
    dropped = sorted(key for key in trainer_kwargs if key not in accepted_keys)
    update_result(grpo_trainer_signature=str(signature), grpo_trainer_dropped_kwargs=dropped)
    return GRPOTrainer(**accepted)


def truncate_prompts_if_needed(tokenizer: Any, rows: list[dict[str, Any]], dropped_config: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    if "max_prompt_length" not in dropped_config:
        return rows
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    truncated: list[dict[str, Any]] = []
    changed = 0
    max_observed = 0
    for row in rows:
        prompt = str(row.get("prompt", ""))
        token_ids = inner(prompt, add_special_tokens=False)["input_ids"]
        max_observed = max(max_observed, len(token_ids))
        if len(token_ids) > MAX_PROMPT_LENGTH:
            changed += 1
            token_ids = token_ids[-MAX_PROMPT_LENGTH:]
            row = dict(row)
            row["prompt"] = inner.decode(token_ids, skip_special_tokens=True)
        truncated.append(row)
    update_result(
        **{
            f"{split_name}_manual_prompt_truncation_enabled": True,
            f"{split_name}_manual_prompt_truncated_rows": changed,
            f"{split_name}_manual_prompt_max_observed_tokens": max_observed,
            f"{split_name}_manual_prompt_target_tokens": MAX_PROMPT_LENGTH,
        }
    )
    return truncated


def main() -> None:
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        update_result(status="starting")
        install_dependencies()
        versions = package_versions()
        update_result(package_versions=versions)

        update_result(stage="import_unsloth_first")
        import unsloth  # noqa: F401 - must happen before trl/transformers/peft
        from unsloth import FastLanguageModel, PatchFastRL
        try:
            PatchFastRL("grpo", FastLanguageModel)
        except TypeError:
            PatchFastRL("grpo")
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
        try:
            from unsloth.chat_templates import get_chat_template
        except Exception:
            get_chat_template = None
        args, dropped_grpo_config = build_grpo_config(GRPOConfig)

        class HeartbeatCallback(TrainerCallback):
            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                if state.global_step and state.global_step % 2 == 0:
                    write_json(
                        OUT_DIR / "heartbeat.json",
                        {"stage": "train", "global_step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch},
                    )
                return control

            def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
                if logs is not None:
                    write_json(OUT_DIR / "latest_log.json", {"global_step": state.global_step, "logs": logs})
                return control

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        update_result(gpu_names=gpu_names)

        data_dir = resolve(DATA_DIR_CANDIDATES, "train.jsonl", "GRPO dataset")
        dataset_report = validate_dataset(data_dir)
        all_train_rows = read_jsonl(data_dir / "train.jsonl")
        all_dev_rows = read_jsonl(data_dir / "dev.jsonl")
        train_rows = all_train_rows if TRAIN_ROW_LIMIT is None else all_train_rows[:TRAIN_ROW_LIMIT]
        dev_rows = all_dev_rows if DEV_ROW_LIMIT is None else all_dev_rows[:DEV_ROW_LIMIT]
        update_result(data_dir=str(data_dir), dataset_report=dataset_report, train_rows=len(train_rows), dev_rows=len(dev_rows))

        model_path = resolve(MODEL_PATH_CANDIDATES, "config.json", "base model")
        adapter_path = resolve(SFT_ADAPTER_CANDIDATES, "adapter_config.json", "SFT adapter", required=True)
        adapter_source = "tool_sft"
        if adapter_path is None:
            raise RuntimeError("No SFT adapter resolved. Attach beacon-tool-sft-best-dev-adapter as Kaggle input.")
        update_result(model_path=str(model_path), adapter_path=str(adapter_path), adapter_source=adapter_source)

        update_result(stage="load_model_unsloth")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        if get_chat_template is not None:
            tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if inner_tokenizer.pad_token is None:
            inner_tokenizer.pad_token = inner_tokenizer.eos_token
        inner_tokenizer.padding_side = "right"
        train_rows = truncate_prompts_if_needed(tokenizer, train_rows, dropped_grpo_config, "train")
        dev_rows = truncate_prompts_if_needed(tokenizer, dev_rows, dropped_grpo_config, "dev")

        update_result(stage="load_sft_adapter")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        trainable = trainable_audit(model)
        update_result(trainable_audit=trainable)
        if trainable["trainable_params"] <= 0:
            raise RuntimeError("No trainable LoRA parameters after loading SFT adapter")

        reward_funcs = [deterministic_reward_func]
        if USE_OPTIONAL_QWEN_JUDGE:
            reward_funcs.append(optional_qwen_judge_reward_func)

        trainer = build_grpo_trainer(
            GRPOTrainer,
            {
                "model": model,
                "args": args,
                "processing_class": tokenizer,
                "train_dataset": Dataset.from_list(train_rows),
                "eval_dataset": Dataset.from_list(dev_rows),
                "reward_funcs": reward_funcs,
            },
        )
        trainer.add_callback(HeartbeatCallback())
        update_result(
            status="training",
            training_objective=TRAINING_OBJECTIVE,
            memory_risk_note="Full deterministic GRPO run with DPO-like 512/192 length targets, num_generations=2, no Qwen judge, no vLLM.",
        )
        trainer.train()
        trainer.save_model(str(OUT_DIR / "adapter_final"))
        tokenizer.save_pretrained(str(OUT_DIR / "adapter_final"))
        update_result(status="completed", output_adapter=str(OUT_DIR / "adapter_final"))
    except Exception as exc:
        update_result(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
