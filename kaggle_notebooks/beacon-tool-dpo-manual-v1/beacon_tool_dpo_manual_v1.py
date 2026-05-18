from __future__ import annotations

import hashlib
import json
import math
import os
import random
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

OUT_DIR = Path("/kaggle/working/beacon_tool_dpo_manual_v1")
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
    "transformers==5.5.0",
    "peft==0.19.1",
    "accelerate==1.13.0",
    "bitsandbytes==0.49.2",
    "sentencepiece",
]

SEED = 3407
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 3072
MAX_COMPLETION_LENGTH = 768
MAX_STEPS = 200
GRAD_ACCUM = 4
LEARNING_RATE = 5e-6
BETA = 0.05
EVAL_EVERY = 50
SAVE_EVERY = 50
DEV_EVAL_LIMIT = 128


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
    for name in ["unsloth", "unsloth_zoo", "transformers", "peft", "accelerate", "bitsandbytes"]:
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


def build_batch(tokenizer: Any, rows: list[dict[str, Any]], side: str, device: Any) -> dict[str, Any]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch_ids: list[list[int]] = []
    batch_labels: list[list[int]] = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False, truncation=True, max_length=MAX_PROMPT_LENGTH)["input_ids"]
        completion_ids = tokenizer(row[side], add_special_tokens=False, truncation=True, max_length=MAX_COMPLETION_LENGTH)["input_ids"]
        available = max(1, MAX_SEQ_LENGTH - len(prompt_ids))
        completion_ids = completion_ids[:available]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        batch_ids.append(input_ids)
        batch_labels.append(labels)
    max_len = max(len(ids) for ids in batch_ids)
    attention_mask: list[list[int]] = []
    for index in range(len(batch_ids)):
        pad_len = max_len - len(batch_ids[index])
        batch_ids[index] = batch_ids[index] + [pad_id] * pad_len
        batch_labels[index] = batch_labels[index] + [-100] * pad_len
        attention_mask.append([1] * (max_len - pad_len) + [0] * pad_len)
    import torch
    return {
        "input_ids": torch.tensor(batch_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(batch_labels, dtype=torch.long, device=device),
    }


def sequence_logps(model: Any, batch: dict[str, Any]) -> Any:
    import torch
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def compute_pair_logps(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: Any) -> tuple[Any, Any]:
    chosen = sequence_logps(model, build_batch(tokenizer, rows, "chosen", device))
    rejected = sequence_logps(model, build_batch(tokenizer, rows, "rejected", device))
    return chosen, rejected


def precompute_reference(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: Any, label: str) -> dict[str, dict[str, float]]:
    import torch
    model.eval()
    refs: dict[str, dict[str, float]] = {}
    path = OUT_DIR / f"reference_logps_{label}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for start in range(0, len(rows), 1):
                batch_rows = rows[start:start + 1]
                chosen, rejected = compute_pair_logps(model, tokenizer, batch_rows, device)
                for row, c_logp, r_logp in zip(batch_rows, chosen.detach().cpu().tolist(), rejected.detach().cpu().tolist()):
                    item = {"dpo_pair_id": row["dpo_pair_id"], "chosen": c_logp, "rejected": r_logp}
                    refs[row["dpo_pair_id"]] = {"chosen": c_logp, "rejected": r_logp}
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return refs


def dpo_loss(policy_chosen: Any, policy_rejected: Any, ref_chosen: Any, ref_rejected: Any) -> tuple[Any, dict[str, float]]:
    import torch
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = pi_logratios - ref_logratios
    losses = -torch.nn.functional.logsigmoid(BETA * logits)
    rewards_chosen = BETA * (policy_chosen - ref_chosen)
    rewards_rejected = BETA * (policy_rejected - ref_rejected)
    margin = rewards_chosen - rewards_rejected
    metrics = {
        "loss": float(losses.mean().detach().cpu()),
        "reward_margin": float(margin.mean().detach().cpu()),
        "reward_accuracy": float((margin > 0).float().mean().detach().cpu()),
    }
    return losses.mean(), metrics


def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], refs: dict[str, dict[str, float]], device: Any) -> dict[str, float]:
    import torch
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    margins: list[float] = []
    with torch.no_grad():
        for row in rows[:DEV_EVAL_LIMIT]:
            chosen, rejected = compute_pair_logps(model, tokenizer, [row], device)
            ref = refs[row["dpo_pair_id"]]
            ref_chosen = torch.tensor([ref["chosen"]], dtype=torch.float32, device=device)
            ref_rejected = torch.tensor([ref["rejected"]], dtype=torch.float32, device=device)
            loss, metrics = dpo_loss(chosen, rejected, ref_chosen, ref_rejected)
            losses.append(float(loss.detach().cpu()))
            accuracies.append(metrics["reward_accuracy"])
            margins.append(metrics["reward_margin"])
    return {
        "eval_loss": sum(losses) / len(losses),
        "eval_reward_accuracy": sum(accuracies) / len(accuracies),
        "eval_reward_margin": sum(margins) / len(margins),
        "eval_rows": len(losses),
    }


def save_adapter(model: Any, tokenizer: Any, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    try:
        tokenizer.save_pretrained(target)
    except AttributeError:
        pass


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "started",
        "stage": "init",
        "training_intent": "manual_light_dpo_tool_decision_steering",
        "research_basis": {
            "dpo": "Rafailov et al. 2023 DPO objective with frozen reference log-prob ratios",
            "lora": "Hu et al. 2021 LoRA parameter-efficient adapter update",
        },
        "drift_controls": {
            "base_adapter": "beacon-tool-sft-best-dev-adapter",
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "beta": BETA,
            "reference": "precomputed frozen log-probs from starting adapter before any update",
            "dev_eval_limit": DEV_EVAL_LIMIT,
        },
    }
    write_json(RESULT_PATH, result)
    try:
        update_result(stage="install_dependencies")
        install_dependencies()

        update_result(stage="import_dependencies", package_versions=package_versions())
        import torch
        from peft import PeftModel
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        random.seed(SEED)
        torch.manual_seed(SEED)
        gpu_report = assert_t4(torch)

        data_dir = resolve(DATA_DIR_CANDIDATES, "train.jsonl", "DPO dataset")
        model_path = resolve(MODEL_PATH_CANDIDATES, "config.json", "Gemma base model")
        adapter_path = resolve(ADAPTER_PATH_CANDIDATES, "adapter_config.json", "tool SFT adapter")
        data_report = validate_dpo_data(data_dir)
        write_json(OUT_DIR / "resolved_paths.json", {"data_dir": str(data_dir), "model_path": str(model_path), "adapter_path": str(adapter_path)})
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
            raise RuntimeError("No trainable adapter parameters")

        train_rows = read_jsonl(data_dir / "train.jsonl")
        dev_rows = read_jsonl(data_dir / "dev.jsonl")
        random.Random(SEED).shuffle(train_rows)
        device = next(model.parameters()).device

        update_result(stage="precompute_reference")
        ref_train = precompute_reference(model, inner_tokenizer, train_rows[: MAX_STEPS * GRAD_ACCUM], device, "train_used")
        ref_dev = precompute_reference(model, inner_tokenizer, dev_rows[:DEV_EVAL_LIMIT], device, "dev_eval")

        update_result(stage="train")
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=LEARNING_RATE)
        train_log: list[dict[str, Any]] = []
        best_eval_loss = float("inf")
        best_step = 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        used_rows = train_rows[: MAX_STEPS * GRAD_ACCUM]
        for step in range(1, MAX_STEPS + 1):
            micro_metrics: list[dict[str, float]] = []
            for micro in range(GRAD_ACCUM):
                row = used_rows[(step - 1) * GRAD_ACCUM + micro]
                chosen, rejected = compute_pair_logps(model, inner_tokenizer, [row], device)
                ref = ref_train[row["dpo_pair_id"]]
                ref_chosen = torch.tensor([ref["chosen"]], dtype=torch.float32, device=device)
                ref_rejected = torch.tensor([ref["rejected"]], dtype=torch.float32, device=device)
                loss, metrics = dpo_loss(chosen, rejected, ref_chosen, ref_rejected)
                (loss / GRAD_ACCUM).backward()
                micro_metrics.append(metrics)
            torch.nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 0.3)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 5 == 0:
                train_log.append({
                    "step": step,
                    "loss": sum(item["loss"] for item in micro_metrics) / len(micro_metrics),
                    "reward_accuracy": sum(item["reward_accuracy"] for item in micro_metrics) / len(micro_metrics),
                    "reward_margin": sum(item["reward_margin"] for item in micro_metrics) / len(micro_metrics),
                })
                write_json(OUT_DIR / "train_log.json", {"log": train_log})
            if step % EVAL_EVERY == 0:
                eval_metrics = evaluate(model, inner_tokenizer, dev_rows, ref_dev, device)
                train_log.append({"step": step, **eval_metrics})
                write_json(OUT_DIR / "train_log.json", {"log": train_log})
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    best_step = step
                    save_adapter(model, inner_tokenizer, OUT_DIR / "adapter_best_dev")
            if step % SAVE_EVERY == 0:
                save_adapter(model, inner_tokenizer, OUT_DIR / f"checkpoint-{step}")

        update_result(stage="save_final")
        save_adapter(model, inner_tokenizer, OUT_DIR / "adapter_final")
        result.update({
            "status": "ok",
            "stage": "complete",
            "adapter_final": str(OUT_DIR / "adapter_final"),
            "adapter_best_dev": str(OUT_DIR / "adapter_best_dev"),
            "best_step": best_step,
            "best_eval_loss": best_eval_loss,
            "train_steps": MAX_STEPS,
            "used_train_pairs": len(used_rows),
            "heldout_final_eval_pairs": data_report["splits"].get("final_eval"),
        })
        write_json(RESULT_PATH, result)
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(RESULT_PATH, result)
        raise


if __name__ == "__main__":
    main()
