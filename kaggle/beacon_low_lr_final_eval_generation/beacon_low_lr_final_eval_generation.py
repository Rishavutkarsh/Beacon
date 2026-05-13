from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = Path("/kaggle/working/beacon_low_lr_final_eval_generation_run018")
RESULT_PATH = OUT_DIR / "generation_result.json"
SEED = 3407
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 192
BATCH_SIZE = 4
PINNED_PACKAGES = [
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
EXPECTED_FINAL_EVAL_COUNT = 93
EXPECTED_HASHES = {
    "final_eval.jsonl": "edd7f5badd492cc6a6324763dec7d13bdbc9f664f20720d6cbe845974e84c084",
    "dataset_freeze_manifest.json": "c9e30394c25cde7f8e12728efad285a6edd6cc365e30a039fa175b41c77abe24",
}
DATA_CANDIDATES = [
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-sft-run018-pruned"),
    Path("/kaggle/input/beacon-sft-run018-pruned"),
]
ADAPTER_CANDIDATES = [
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-low-lr1e5-best-adapters-run018"),
    Path("/kaggle/input/beacon-low-lr1e5-best-adapters-run018"),
]
MODEL_CANDIDATES = [
    Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1"),
]
CANDIDATES = [
    {"candidate_id": "base", "kind": "base", "checkpoint_step": None},
    {"candidate_id": "attention_only_best_dev", "kind": "adapter", "checkpoint_step": 224},
    {"candidate_id": "all_linear_best_dev", "kind": "adapter", "checkpoint_step": 224},
]


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def update_result(**items: Any) -> None:
    current: dict[str, Any] = {}
    if RESULT_PATH.exists():
        try:
            current = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(items)
    write_json(RESULT_PATH, current)


def install_dependencies() -> None:
    command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *PINNED_PACKAGES]
    result = subprocess.run(command, text=True, capture_output=True)
    write_json(
        OUT_DIR / "dependency_install.json",
        {"command": command, "returncode": result.returncode, "stdout_tail": result.stdout[-6000:], "stderr_tail": result.stderr[-6000:]},
    )
    if result.returncode != 0:
        raise RuntimeError("Pinned dependency install failed.")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def resolve_path(candidates: list[Path], required_file: str) -> Path:
    for candidate in candidates:
        if (candidate / required_file).exists():
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for found in input_root.rglob(required_file):
            parent = found.parent
            lower = str(parent).lower()
            if required_file == "final_eval.jsonl" and "beacon-sft-run018-pruned" in lower:
                return parent
            if required_file == "config.json" and "gemma-4" in lower and "e2b-it" in lower and "transformers" in lower:
                return parent
    raise RuntimeError(f"No candidate path contains {required_file}: {[str(item) for item in candidates]}")


def validate_gpu(torch_module: Any) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=0, got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    names = [torch_module.cuda.get_device_name(index) for index in range(torch_module.cuda.device_count())]
    if len(names) != 1 or "t4" not in names[0].lower():
        raise RuntimeError(f"This run requires one Kaggle T4; visible devices: {names}")
    return {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_names": names}


def validate_data(data_dir: Path) -> list[dict[str, Any]]:
    hashes = {name: sha256_file(data_dir / name) for name in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES:
        raise RuntimeError(f"Dataset hash gate failed: {hashes}")
    final_eval_rows = read_jsonl(data_dir / "final_eval.jsonl")
    if len(final_eval_rows) != EXPECTED_FINAL_EVAL_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_FINAL_EVAL_COUNT} final_eval rows, found {len(final_eval_rows)}")
    return final_eval_rows


def prepare_adapter_dir(adapter_root: Path, candidate_id: str) -> Path:
    direct = adapter_root / candidate_id
    if (direct / "adapter_model.safetensors").exists():
        return direct
    zip_path = adapter_root / f"{candidate_id}.zip"
    if not zip_path.exists():
        raise RuntimeError(f"Adapter for {candidate_id} not found in {adapter_root}")
    extract_dir = OUT_DIR / "extracted_adapters" / candidate_id
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    if (extract_dir / candidate_id / "adapter_model.safetensors").exists():
        return extract_dir / candidate_id
    if (extract_dir / "adapter_model.safetensors").exists():
        return extract_dir
    matches = list(extract_dir.rglob("adapter_model.safetensors"))
    if matches:
        return matches[0].parent
    raise RuntimeError(f"Could not locate adapter weights after extracting {zip_path}")


def resolve_adapter_root() -> Path:
    for candidate in ADAPTER_CANDIDATES:
        if (candidate / "attention_only_best_dev" / "adapter_model.safetensors").exists() or (candidate / "attention_only_best_dev.zip").exists():
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for found in input_root.rglob("adapter_model.safetensors"):
            parent = found.parent
            if parent.name == "attention_only_best_dev" and "beacon-low-lr1e5-best-adapters-run018" in str(parent).lower():
                return parent.parent
        for found in input_root.rglob("attention_only_best_dev.zip"):
            if "beacon-low-lr1e5-best-adapters-run018" in str(found).lower():
                return found.parent
    raise RuntimeError("Could not resolve Beacon candidate adapter dataset root.")


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    prompt_messages = [message for message in row["messages"] if message.get("role") != "assistant"]
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(text, list):
        text = "".join(text)
    return text.removeprefix("<bos>")


def load_model_and_tokenizer(FastLanguageModel: Any, get_chat_template: Any, model_path: Path, adapter_dir: Path | None = None) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if inner.pad_token is None:
        inner.pad_token = inner.eos_token
    inner.padding_side = "left"
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_candidate(FastLanguageModel: Any, get_chat_template: Any, torch: Any, model_path: Path, eval_rows: list[dict[str, Any]], candidate: dict[str, Any], adapter_root: Path) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    adapter_dir = None if candidate["kind"] == "base" else prepare_adapter_dir(adapter_root, candidate_id)
    model, tokenizer = load_model_and_tokenizer(FastLanguageModel, get_chat_template, model_path, adapter_dir)
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    predictions: list[dict[str, Any]] = []
    prediction_path = OUT_DIR / "predictions" / f"{candidate_id}.jsonl"
    if prediction_path.exists():
        prediction_path.unlink()
    for start in range(0, len(eval_rows), BATCH_SIZE):
        batch_rows = eval_rows[start:start + BATCH_SIZE]
        prompt_texts = [render_prompt(tokenizer, row) for row in batch_rows]
        encoded = inner(prompt_texts, return_tensors="pt", add_special_tokens=False, padding=True).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=inner.pad_token_id,
                eos_token_id=inner.eos_token_id,
            )
        batch_records: list[dict[str, Any]] = []
        prompt_width = encoded["input_ids"].shape[-1]
        for offset, row in enumerate(batch_rows):
            generated_ids = output[offset][prompt_width:]
            response = inner.decode(generated_ids, skip_special_tokens=True).strip()
            record = {
                "example_id": row["id"],
                "candidate_id": candidate_id,
                "candidate_kind": candidate["kind"],
                "checkpoint_step": candidate["checkpoint_step"],
                "split": "final_eval",
                "hazard_domain": row.get("hazard_domain"),
                "risk_level": row.get("risk_level"),
                "renderer_style": row.get("renderer_style"),
                "source_rule_ids": row.get("source_rule_ids", []),
                "target_behavior_tags": row.get("target_behavior_tags", []),
                "forbidden_behavior_tags": row.get("forbidden_behavior_tags", []),
                "prompt": row.get("prompt"),
                "target_response": row.get("target_response"),
                "model_response": response,
                "response_char_count": len(response),
                "response_token_count": int(generated_ids.numel()),
                "generation_config": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "batch_size": BATCH_SIZE},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            predictions.append(record)
            batch_records.append(record)
        append_jsonl(prediction_path, batch_records)
        update_result(stage="generate", current_candidate=candidate_id, generated_rows=len(predictions))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "candidate_id": candidate_id,
        "rows": len(predictions),
        "prediction_path": str(prediction_path),
        "avg_response_chars": sum(item["response_char_count"] for item in predictions) / max(1, len(predictions)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    update_result(status="running", stage="install_dependencies", started_at=datetime.now(timezone.utc).isoformat())
    install_dependencies()
    update_result(stage="imports")

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    update_result(stage="resolve_inputs")
    data_dir = resolve_path(DATA_CANDIDATES, "final_eval.jsonl")
    adapter_root = resolve_adapter_root()
    model_path = resolve_path(MODEL_CANDIDATES, "config.json")
    final_eval_rows = validate_data(data_dir)
    dependency_report = {
        "python": sys.version,
        "torch": package_version("torch"),
        "unsloth": package_version("unsloth"),
        "unsloth_zoo": package_version("unsloth_zoo"),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "accelerate": package_version("accelerate"),
        "bitsandbytes": package_version("bitsandbytes"),
    }
    gpu_report = validate_gpu(torch)
    write_json(OUT_DIR / "resolved_paths.json", {"data_dir": str(data_dir), "adapter_root": str(adapter_root), "model_path": str(model_path)})
    write_json(OUT_DIR / "dependency_report.json", {**dependency_report, **gpu_report})
    write_json(
        OUT_DIR / "eval_manifest.json",
        {
            "split": "final_eval",
            "row_count": len(final_eval_rows),
            "candidates": CANDIDATES,
            "final_eval_policy": "allowed_after_dev_winner_frozen_no_training_or_checkpoint_selection_feedback",
        },
    )

    summaries = []
    for candidate in CANDIDATES:
        update_result(stage="generate", current_candidate=candidate["candidate_id"], generated_rows=0)
        summaries.append(generate_candidate(FastLanguageModel, get_chat_template, torch, model_path, final_eval_rows, candidate, adapter_root))

    write_json(OUT_DIR / "prediction_summary.json", {"summaries": summaries})
    update_result(status="pass", stage="complete", finished_at=datetime.now(timezone.utc).isoformat(), summaries=summaries)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        update_result(
            status="fail",
            stage="error",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error_type=type(exc).__name__,
            message=str(exc),
            traceback_tail="\n".join(traceback.format_exc().splitlines()[-25:]),
        )
        raise
