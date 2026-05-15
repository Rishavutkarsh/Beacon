from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = Path("/kaggle/working/beacon_dapt_cpt_eval_v1")
RESULT_PATH = OUT_DIR / "cpt_eval_result.json"
DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/beacon-crisis-v1-cpt"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-crisis-v1-cpt"),
]
MODEL_PATH_CANDIDATES = [
    Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1"),
]
ADAPTER_PATH = os.environ.get("BEACON_CPT_ADAPTER_PATH", "")
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
EXPECTED_HASHES = {
    "cpt_train.jsonl": "f13ccb7a7e03b5de9b3ee553c8ac4975a8646accc86c04127edb00c2fc9bad8f",
    "cpt_dev.jsonl": "270d555448118bba4d00416aec9af330035a16451bf042861a08a7b6a68555f7",
    "cpt_test.jsonl": "7748a0c923339485c10849624f49bbfa860371cca445e394b084d84611a64249",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(candidates: list[Path], child: str, label: str) -> Path:
    for candidate in candidates:
        if (candidate / child).exists():
            return candidate
    raise RuntimeError(f"Could not resolve {label}")


def install_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *PINNED])


def validate_t4(torch_module) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    names = [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())]
    if torch_module.cuda.device_count() != 1 or "t4" not in names[0].lower():
        raise RuntimeError(f"Expected single T4; visible={names}")
    return {"gpu_names": names, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def validate_hashes(data_dir: Path) -> dict[str, str]:
    observed = {name: sha256_file(data_dir / name) for name in EXPECTED_HASHES}
    mismatch = {name: {"expected": expected, "observed": observed[name]} for name, expected in EXPECTED_HASHES.items() if observed[name] != expected}
    if mismatch:
        raise RuntimeError(f"CPT eval package hash mismatch: {mismatch}")
    return observed


def sft_trainer(SFTTrainer, *, model, eval_dataset, tokenizer, args):
    signature = inspect.signature(SFTTrainer)
    kwargs = {"model": model, "train_dataset": eval_dataset, "eval_dataset": eval_dataset, "args": args}
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def eval_split(SFTConfig, SFTTrainer, model, tokenizer, rows: list[dict[str, Any]], split: str):
    from datasets import Dataset

    dataset = Dataset.from_list([{"text": row["text"], "row_id": row["row_id"]} for row in rows])
    args = SFTConfig(
        output_dir=str(OUT_DIR / f"eval_{split}"),
        dataset_text_field="text",
        max_length=2048,
        per_device_eval_batch_size=1,
        dataset_num_proc=1,
        packing=False,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = sft_trainer(SFTTrainer, model=model, eval_dataset=dataset, tokenizer=tokenizer, args=args)
    metrics = dict(trainer.evaluate())
    value = metrics.get("eval_loss")
    if value is not None and not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite {split} eval_loss")
    return metrics


def main() -> None:
    result: dict[str, Any] = {"status": "failure", "stage": "start"}
    write_json(RESULT_PATH, result)
    try:
        result["stage"] = "install"
        install_dependencies()
        result["package_versions"] = {name: metadata.version(name) for name in ["unsloth", "trl", "transformers", "peft", "datasets"]}
        write_json(RESULT_PATH, result)

        result["stage"] = "imports"
        import torch
        from unsloth import FastLanguageModel
        from peft import PeftModel
        from trl import SFTConfig, SFTTrainer

        result.update(validate_t4(torch))
        data_dir = resolve(DATA_DIR_CANDIDATES, "cpt_dev.jsonl", "CPT data")
        model_path = resolve(MODEL_PATH_CANDIDATES, "config.json", "base model")
        result["data_hashes"] = validate_hashes(data_dir)
        result["data_dir"] = str(data_dir)
        result["model_path"] = str(model_path)
        result["adapter_path"] = ADAPTER_PATH
        write_json(RESULT_PATH, result)

        result["stage"] = "load_model"
        model, tokenizer = FastLanguageModel.from_pretrained(str(model_path), max_seq_length=2048, dtype=None, load_in_4bit=True)
        if ADAPTER_PATH:
            adapter = Path(ADAPTER_PATH)
            if not (adapter / "adapter_config.json").exists():
                raise RuntimeError(f"Missing adapter_config.json at {adapter}")
            model = PeftModel.from_pretrained(model, str(adapter))
        FastLanguageModel.for_inference(model)
        dev_rows = read_jsonl(data_dir / "cpt_dev.jsonl")
        test_rows = read_jsonl(data_dir / "cpt_test.jsonl")
        result["stage"] = "eval"
        result["dev_metrics"] = eval_split(SFTConfig, SFTTrainer, model, tokenizer, dev_rows, "dev")
        result["test_metrics"] = eval_split(SFTConfig, SFTTrainer, model, tokenizer, test_rows, "test")
        result["status"] = "success"
        result["stage"] = "complete"
        write_json(RESULT_PATH, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        result["status"] = "failure"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-30:]
        write_json(RESULT_PATH, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        raise


if __name__ == "__main__":
    main()
