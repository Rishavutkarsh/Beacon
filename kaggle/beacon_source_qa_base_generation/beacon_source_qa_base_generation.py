from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = Path("/kaggle/working/beacon_source_qa_base_generation_v1")
RESULT_PATH = OUT_DIR / "generation_result.json"
EXPECTED_ROWS = 60
EXPECTED_EVAL_HASH = "531ffdd2b4694642de0a25d36fb78e0e3dbceb4cbe1a275c5c5ec36d4b9c5e79"
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 180
PINNED = [
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
    "transformers==5.5.0",
    "peft==0.19.1",
    "accelerate==1.13.0",
    "bitsandbytes==0.49.2",
    "datasets==4.3.0",
    "sentencepiece",
]
DATA_CANDIDATES = [
    Path("/kaggle/input/beacon-source-qa-eval-v1"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-source-qa-eval-v1"),
]
MODEL_CANDIDATES = [
    Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1"),
    Path("/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1"),
]


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(candidates: list[Path], child: str, label: str) -> Path:
    for candidate in candidates:
        if (candidate / child).exists():
            return candidate
    visible = [str(path) for path in sorted(Path("/kaggle/input").iterdir())] if Path("/kaggle/input").exists() else []
    raise RuntimeError(f"Could not resolve {label}; visible_inputs={visible}")


def install_deps() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *PINNED])


def assert_t4(torch_module) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    names = [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())]
    if torch_module.cuda.device_count() != 1 or "t4" not in names[0].lower():
        raise RuntimeError(f"Expected single T4; visible={names}")
    return {"gpu_names": names, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["question"]},
    ]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(rendered).removeprefix("<bos>")


def generate_candidate(candidate_id: str, *, rows: list[dict[str, Any]], model_path: Path, adapter_path: Path | None) -> list[dict[str, Any]]:
    import gc
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(str(model_path), max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=True)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if inner.pad_token is None:
        inner.pad_token = inner.eos_token
    FastLanguageModel.for_inference(model)

    generations: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        prompt_text = render_prompt(tokenizer, item)
        inputs = inner(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=inner.eos_token_id,
            )
        continuation = output[0][inputs["input_ids"].shape[-1] :]
        answer = inner.decode(continuation, skip_special_tokens=True).strip()
        generations.append(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "example_id": item["example_id"],
                "candidate_id": candidate_id,
                "hazard_bucket": item["hazard_bucket"],
                "difficulty": item["difficulty"],
                "seen_in_cpt_train": item["seen_in_cpt_train"],
                "seen_in_cpt_dev": item["seen_in_cpt_dev"],
                "seen_in_cpt_test": item["seen_in_cpt_test"],
                "question_sha256": hashlib.sha256(item["question"].encode("utf-8")).hexdigest(),
                "response": answer,
                "response_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "generation_config": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
            }
        )
        print(f"[beacon-source-qa-generation] {candidate_id} {index}/{len(rows)} {item['example_id']}", flush=True)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generations


def main() -> None:
    result: dict[str, Any] = {"status": "failure", "stage": "start"}
    write_json(RESULT_PATH, result)
    try:
        result["stage"] = "install"
        install_deps()
        result["package_versions"] = {name: metadata.version(name) for name in ["unsloth", "transformers", "peft", "bitsandbytes"]}
        write_json(RESULT_PATH, result)

        result["stage"] = "imports"
        import torch

        result.update(assert_t4(torch))
        data_dir = resolve(DATA_CANDIDATES, "beacon_source_qa_eval_v1.jsonl", "source QA eval data")
        model_path = resolve(MODEL_CANDIDATES, "config.json", "Gemma base model")
        adapter_path = None
        eval_path = data_dir / "beacon_source_qa_eval_v1.jsonl"
        eval_hash = sha256_file(eval_path)
        if eval_hash != EXPECTED_EVAL_HASH:
            raise RuntimeError(f"Eval hash mismatch: expected={EXPECTED_EVAL_HASH} observed={eval_hash}")
        rows = read_jsonl(eval_path)
        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError(f"Expected {EXPECTED_ROWS} rows; got {len(rows)}")
        result.update({"data_dir": str(data_dir), "model_path": str(model_path), "adapter_path": "", "eval_hash": eval_hash, "row_count": len(rows)})
        write_json(RESULT_PATH, result)

        result["stage"] = "generate_base"
        all_generations = generate_candidate("base_gemma", rows=rows, model_path=model_path, adapter_path=None)
        write_jsonl(OUT_DIR / "base_source_qa_generations.jsonl", all_generations)

        result["status"] = "success"
        result["stage"] = "complete"
        result["outputs"] = {"base_source_qa_generations": str(OUT_DIR / "base_source_qa_generations.jsonl")}
        result["generation_hash"] = sha256_file(OUT_DIR / "base_source_qa_generations.jsonl")
        result["candidate_counts"] = {"base_gemma": EXPECTED_ROWS}
        write_json(RESULT_PATH, result)
    except Exception as exc:
        result["status"] = "failure"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-30:]
        write_json(RESULT_PATH, result)
        raise


if __name__ == "__main__":
    main()
