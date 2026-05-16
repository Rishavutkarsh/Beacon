from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = Path("/kaggle/working/beacon_dapt_cpt_smoke_v1")
RESULT_PATH = OUT_DIR / "smoke_result.json"
DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/beacon-crisis-v1-cpt"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-crisis-v1-cpt"),
]
MODEL_PATH_CANDIDATES = [
    Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1"),
]
EXPECTED_HASHES = {
    "cpt_train.jsonl": "f13ccb7a7e03b5de9b3ee553c8ac4975a8646accc86c04127edb00c2fc9bad8f",
    "cpt_dev.jsonl": "270d555448118bba4d00416aec9af330035a16451bf042861a08a7b6a68555f7",
    "cpt_test.jsonl": "7748a0c923339485c10849624f49bbfa860371cca445e394b084d84611a64249",
    "cpt_training_config.json": "f3da4bdfc88671fdf7aba905ce3a6c5b9e05062fec31e54aa2b1b7d2503919d1",
    "cpt_split_manifest.json": "94f9e1e7f52b77d9b10d0a20db4093661cfc76fc72d319df91736e4a44d7b1e4",
}
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
SEED = 17
SMOKE_TRAIN_ROWS = 16
SMOKE_DEV_ROWS = 8
SMOKE_MAX_STEPS = 10


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
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_result(payload: dict[str, Any]) -> None:
    write_json(RESULT_PATH, payload)


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


def resolve(candidates: list[Path], required_child: str, label: str) -> Path:
    for candidate in candidates:
        if (candidate / required_child).exists():
            return candidate
    raise RuntimeError(f"Could not resolve {label}; checked {[str(item) for item in candidates]}")


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


def assert_t4(torch_module) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    names = [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())]
    if torch_module.cuda.device_count() != 1 or "t4" not in names[0].lower():
        raise RuntimeError(f"CPT smoke requires a single Kaggle T4; visible devices: {names}")
    return {"gpu_names": names, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def validate_data(data_dir: Path) -> dict[str, Any]:
    observed = {name: sha256_file(data_dir / name) for name in EXPECTED_HASHES}
    mismatches = {name: {"expected": expected, "observed": observed[name]} for name, expected in EXPECTED_HASHES.items() if observed[name] != expected}
    if mismatches:
        raise RuntimeError(f"CPT smoke package hash gate failed: {mismatches}")
    manifest = read_json(data_dir / "cpt_package_manifest.json")
    internal = manifest.get("hashes") or {}
    internal_mismatches = {name: {"manifest": internal.get(name), "observed": observed[name]} for name in EXPECTED_HASHES if internal.get(name) != observed[name]}
    if internal_mismatches:
        raise RuntimeError(f"CPT smoke manifest hash mismatch: {internal_mismatches}")
    train = read_jsonl(data_dir / "cpt_train.jsonl")
    dev = read_jsonl(data_dir / "cpt_dev.jsonl")
    test = read_jsonl(data_dir / "cpt_test.jsonl")
    if any("messages" in row or "assistant_response" in row or "prompt" in row for row in train + dev + test):
        raise RuntimeError("CPT smoke package contains SFT/chat fields")
    return {"counts": {"train": len(train), "dev": len(dev), "test": len(test)}, "hashes": observed}


def language_targets(model) -> list[str]:
    suffixes = (
        ".q_proj", ".k_proj", ".v_proj", ".o_proj", ".gate_proj", ".up_proj", ".down_proj",
        ".q_proj.linear", ".k_proj.linear", ".v_proj.linear", ".o_proj.linear",
        ".gate_proj.linear", ".up_proj.linear", ".down_proj.linear",
    )
    candidates = [
        name for name, _module in model.named_modules()
        if "language_model" in name and "vision_tower" not in name and "audio_tower" not in name and name.endswith(suffixes)
    ]
    candidate_set = set(candidates)
    leaves = sorted(name for name in candidates if not any(other != name and other.startswith(name + ".") for other in candidate_set))
    if not leaves:
        raise RuntimeError("No language LoRA target leaves found")
    return leaves


def trainable_summary(model) -> dict[str, Any]:
    names: list[str] = []
    total = 0
    tower_names: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
            total += int(param.numel())
            if "vision_tower" in name or "audio_tower" in name:
                tower_names.append(name)
    if total == 0:
        raise RuntimeError("Smoke LoRA setup produced zero trainable parameters")
    if tower_names:
        raise RuntimeError(f"Smoke LoRA setup targeted tower parameters: {tower_names[:10]}")
    return {
        "trainable_param_count": total,
        "trainable_tensor_count": len(names),
        "tower_trainable_count": len(tower_names),
        "sample_trainable_names": names[:30],
    }


def sft_trainer(SFTTrainer, *, model, train_dataset, eval_dataset, tokenizer, args):
    signature = inspect.signature(SFTTrainer)
    kwargs = {"model": model, "train_dataset": train_dataset, "eval_dataset": eval_dataset, "args": args}
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError("Installed SFTTrainer accepts neither processing_class nor tokenizer")
    return SFTTrainer(**kwargs)


class CPTTrainerShim:
    @staticmethod
    def patch(SFTTrainer):
        class CPTTrainer(SFTTrainer):
            def _clip_grad_norm(self, model):
                restored = []
                for parameter in model.parameters():
                    grad = getattr(parameter, "grad", None)
                    if grad is not None and str(grad.dtype) == "torch.float16":
                        restored.append((parameter, grad.dtype))
                        grad.data = grad.data.float()
                try:
                    return super()._clip_grad_norm(model)
                finally:
                    for parameter, dtype in restored:
                        grad = getattr(parameter, "grad", None)
                        if grad is not None and grad.dtype != dtype:
                            grad.data = grad.data.to(dtype)
        return CPTTrainer


def assert_finite_metric(metrics: dict[str, Any], key: str, label: str) -> None:
    if key in metrics and metrics[key] is not None and not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Non-finite {label} {key}: {metrics[key]!r}")


def main() -> None:
    result: dict[str, Any] = {"status": "failure", "stage": "start"}
    write_result(result)
    try:
        data_dir = resolve(DATA_DIR_CANDIDATES, "cpt_train.jsonl", "CPT data")
        model_path = resolve(MODEL_PATH_CANDIDATES, "config.json", "Gemma 4 model")
        result.update({"data_dir": str(data_dir), "model_path": str(model_path)})
        result["stage"] = "install"
        install_dependencies()
        result["package_versions"] = package_versions()
        write_result(result)

        result["stage"] = "imports"
        import torch
        from datasets import Dataset, disable_caching
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer

        CPTTrainer = CPTTrainerShim.patch(SFTTrainer)
        disable_caching()
        random.seed(SEED)
        torch.manual_seed(SEED)
        result.update(assert_t4(torch))
        result["data_validation"] = validate_data(data_dir)
        config = read_json(data_dir / "cpt_training_config.json")
        write_result(result)

        result["stage"] = "load_model"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=int(config["max_seq_length"]),
            dtype=None,
            load_in_4bit=True,
        )
        if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if inner_tokenizer.pad_token is None:
            inner_tokenizer.pad_token = inner_tokenizer.eos_token
        inner_tokenizer.padding_side = "right"
        write_result(result)

        result["stage"] = "attach_lora"
        target_modules = language_targets(model)
        result["lora_target_audit"] = {"target_count": len(target_modules), "samples": target_modules[:40]}
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(config["lora_r"]),
            target_modules=target_modules,
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            random_state=SEED,
            use_gradient_checkpointing="unsloth",
        )
        result["trainable_summary"] = trainable_summary(model)
        write_json(OUT_DIR / "lora_target_audit.json", {"lora_target_audit": result["lora_target_audit"], "trainable_summary": result["trainable_summary"]})
        write_result(result)

        result["stage"] = "datasets"
        train_rows = [{"text": row["text"], "row_id": row["row_id"]} for row in read_jsonl(data_dir / "cpt_train.jsonl")[:SMOKE_TRAIN_ROWS]]
        dev_rows = [{"text": row["text"], "row_id": row["row_id"]} for row in read_jsonl(data_dir / "cpt_dev.jsonl")[:SMOKE_DEV_ROWS]]
        train_dataset = Dataset.from_list(train_rows)
        dev_dataset = Dataset.from_list(dev_rows)
        args = SFTConfig(
            output_dir=str(OUT_DIR / "trainer"),
            dataset_text_field="text",
            max_length=int(config["max_seq_length"]),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=SMOKE_MAX_STEPS,
            learning_rate=float(config["learning_rate"]),
            warmup_steps=1,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=5,
            save_strategy="no",
            prediction_loss_only=True,
            fp16=True,
            bf16=False,
            optim=str(config["optim"]),
            max_grad_norm=float(config["max_grad_norm"]),
            weight_decay=float(config["weight_decay"]),
            dataset_num_proc=1,
            packing=bool(config["packing"]),
            report_to=[],
            seed=SEED,
            remove_unused_columns=False,
        )
        trainer = sft_trainer(CPTTrainer, model=model, train_dataset=train_dataset, eval_dataset=dev_dataset, tokenizer=tokenizer, args=args)
        result["stage"] = "baseline_eval"
        baseline = dict(trainer.evaluate())
        assert_finite_metric(baseline, "eval_loss", "baseline")
        result["baseline_dev_metrics"] = baseline
        write_result(result)

        result["stage"] = "train"
        train_result = trainer.train()
        train_metrics = dict(train_result.metrics)
        assert_finite_metric(train_metrics, "train_loss", "smoke_train")
        result["train_metrics"] = train_metrics
        result["train_log_history"] = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
        write_result(result)

        result["stage"] = "post_eval_save"
        post = dict(trainer.evaluate())
        assert_finite_metric(post, "eval_loss", "post_smoke")
        result["post_dev_metrics"] = post
        curve_path = OUT_DIR / "train_eval_curve.json"
        write_json(curve_path, {"log_history": result["train_log_history"]})
        adapter_dir = OUT_DIR / "adapter_smoke"
        trainer.model.save_pretrained(str(adapter_dir))
        try:
            tokenizer.save_pretrained(str(OUT_DIR / "tokenizer"))
        except AttributeError:
            inner_tokenizer.save_pretrained(str(OUT_DIR / "tokenizer"))
        if not (adapter_dir / "adapter_config.json").exists():
            raise RuntimeError("Smoke adapter_config.json was not saved")
        result["adapter_smoke"] = str(adapter_dir)
        result["artifacts"] = {
            "smoke_result": str(RESULT_PATH),
            "lora_target_audit": str(OUT_DIR / "lora_target_audit.json"),
            "train_eval_curve": str(curve_path),
            "adapter_smoke": str(adapter_dir),
        }
        result["status"] = "success"
        result["stage"] = "complete"
        write_result(result)
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        result["status"] = "failure"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-30:]
        write_result(result)
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False))
        raise


if __name__ == "__main__":
    main()
