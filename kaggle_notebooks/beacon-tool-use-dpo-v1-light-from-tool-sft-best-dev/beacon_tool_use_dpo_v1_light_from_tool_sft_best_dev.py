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
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUT_DIR = Path("/kaggle/working/beacon_tool_use_dpo_v1_light_from_tool_sft_best_dev")
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
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 3072
MAX_COMPLETION_LENGTH = 768
MAX_STEPS = 200
LEARNING_RATE = 5e-6
BETA = 0.05
GRAD_ACCUM = 4
PER_DEVICE_BATCH = 1
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 5


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


def ensure_optional_import_shims() -> None:
    """TRL 0.24 imports MergeKit callback helpers even when DPO never uses them."""
    shim_root = OUT_DIR / "vendor_shims"
    package_dir = shim_root / "mergekit"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "config.py").write_text(
        "class MergeConfiguration:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.args = args\n"
        "        self.kwargs = kwargs\n",
        encoding="utf-8",
    )
    (package_dir / "merge.py").write_text(
        "class MergeOptions:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.args = args\n"
        "        self.kwargs = kwargs\n\n"
        "def run_merge(*args, **kwargs):\n"
        "    raise RuntimeError('MergeKit shim: run_merge is unavailable in this DPO kernel')\n",
        encoding="utf-8",
    )
    (shim_root / "llm_blender.py").write_text(
        "class Blender:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.args = args\n"
        "        self.kwargs = kwargs\n\n"
        "    def loadranker(self, *args, **kwargs):\n"
        "        return self\n\n"
        "    def rank(self, *args, **kwargs):\n"
        "        raise RuntimeError('llm_blender shim: rank is unavailable in this DPO kernel')\n",
        encoding="utf-8",
    )
    (shim_root / "weave.py").write_text(
        "class EvaluationLogger:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.args = args\n"
        "        self.kwargs = kwargs\n\n"
        "    def log_prediction(self, *args, **kwargs):\n"
        "        return None\n\n"
        "    def finish(self, *args, **kwargs):\n"
        "        return None\n\n"
        "def init(*args, **kwargs):\n"
        "    return None\n\n"
        "def op(fn=None, *args, **kwargs):\n"
        "    if fn is None:\n"
        "        return lambda inner: inner\n"
        "    return fn\n\n"
        "def log(*args, **kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(shim_root))


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

    splits: dict[str, int] = {}
    hashes: dict[str, str] = {}
    pair_types: dict[str, int] = {}
    tool_required = {True: 0, False: 0}
    chosen_tool_calls = 0
    rejected_tool_calls = 0
    for split in ["train", "dev", "final_eval"]:
        path = data_dir / f"{split}.jsonl"
        hashes[path.name] = sha256_file(path)
        seen: set[str] = set()
        count = 0
        for index, row in enumerate(read_jsonl(path), 1):
            count += 1
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
            pair_types[row["pair_type"]] = pair_types.get(row["pair_type"], 0) + 1
            tool_required[bool(row["tool_required"])] += 1
            chosen_tool_calls += int("<tool_call>" in row["chosen"])
            rejected_tool_calls += int("<tool_call>" in row["rejected"])
        splits[split] = count
    return {
        "manifest_status": manifest.get("status"),
        "manifest_review": manifest.get("independent_review"),
        "splits": splits,
        "hashes": hashes,
        "pair_types": pair_types,
        "tool_required": {str(key): value for key, value in tool_required.items()},
        "chosen_tool_call_pairs": chosen_tool_calls,
        "rejected_tool_call_pairs": rejected_tool_calls,
    }


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
        "training_intent": "light_dpo_tool_decision_steering",
        "drift_controls": {
            "base_adapter": "beacon-tool-sft-best-dev-adapter",
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "beta": BETA,
            "save_every_steps": SAVE_STEPS,
            "eval_every_steps": EVAL_STEPS,
        },
    }
    write_json(RESULT_PATH, result)
    try:
        update_result(stage="install_dependencies")
        install_dependencies()

        update_result(stage="import_dependencies", package_versions=package_versions())
        ensure_optional_import_shims()
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        gpu_report = assert_t4(torch)
        data_dir = resolve(DATA_DIR_CANDIDATES, "train.jsonl", "DPO dataset")
        model_path = resolve(MODEL_PATH_CANDIDATES, "config.json", "Gemma base model")
        adapter_path = resolve(ADAPTER_PATH_CANDIDATES, "adapter_config.json", "tool SFT adapter")
        data_report = validate_dpo_data(data_dir)
        write_json(
            OUT_DIR / "resolved_paths.json",
            {"data_dir": str(data_dir), "model_path": str(model_path), "adapter_path": str(adapter_path), "out_dir": str(OUT_DIR)},
        )
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
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
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
        dev_rows = read_jsonl(data_dir / "dev.jsonl")
        train_ds = Dataset.from_list([{key: row[key] for key in ["prompt", "chosen", "rejected"]} for row in train_rows])
        eval_ds = Dataset.from_list([{key: row[key] for key in ["prompt", "chosen", "rejected"]} for row in dev_rows])

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
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=False,
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH,
            fp16=True,
            bf16=False,
            optim="adamw_8bit",
            max_grad_norm=0.3,
            report_to=[],
            seed=SEED,
            remove_unused_columns=False,
            dataset_num_proc=1,
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
        trainer = DPOTrainer(**trainer_kwargs)
        write_json(
            OUT_DIR / "pre_train_audit.json",
            {
                "trainable": trainable,
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
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
        best_step, best_loss = min(eval_losses, key=lambda item: item[1])
        best_checkpoint = OUT_DIR / "trainer" / f"checkpoint-{best_step}"
        best_dir = OUT_DIR / "adapter_best_dev"
        best_copy = copy_checkpoint(best_checkpoint, best_dir)
        write_json(OUT_DIR / "train_eval_curve.json", {"log_history": log_history, "eval_losses": eval_losses})

        result.update(
            {
                "status": "ok",
                "stage": "complete",
                "adapter_final": str(final_dir),
                "adapter_best_dev": str(best_dir),
                "best_step": best_step,
                "best_eval_loss": best_loss,
                "best_checkpoint_copy": best_copy,
                "final_eval_rows_held_out": data_report["splits"].get("final_eval"),
            }
        )
        write_json(RESULT_PATH, result)
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(RESULT_PATH, result)
        raise


if __name__ == "__main__":
    main()
