from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = Path("/kaggle/working/beacon_source_qa_qwen_judge_v1")
RESULT_PATH = OUT_DIR / "judge_result.json"
EXPECTED_ROWS = 60
EXPECTED_GENERATIONS_PER_CANDIDATE = 60
EXPECTED_EVAL_HASH = "531ffdd2b4694642de0a25d36fb78e0e3dbceb4cbe1a275c5c5ec36d4b9c5e79"
PINNED = ["transformers==5.5.0", "accelerate==1.13.0", "bitsandbytes==0.49.2", "sentencepiece"]
DATA_CANDIDATES = [
    Path("/kaggle/input/beacon-source-qa-eval-v1"),
    Path("/kaggle/input/datasets/rishavutkarsh/beacon-source-qa-eval-v1"),
]
BASE_GEN_CANDIDATES = [
    Path("/kaggle/input/beacon-source-qa-base-generation-v1"),
    Path("/kaggle/input/notebooks/rishavutkarsh/beacon-source-qa-base-generation-v1"),
]
CPT_GEN_CANDIDATES = [
    Path("/kaggle/input/beacon-source-qa-cpt-generation-v1"),
    Path("/kaggle/input/notebooks/rishavutkarsh/beacon-source-qa-cpt-generation-v1"),
]
JUDGE_MODEL_CANDIDATES = [
    Path("/kaggle/input/qwen2.5/Transformers/7b-instruct/1"),
    Path("/kaggle/input/qwen2.5/transformers/7b-instruct/1"),
    Path("/kaggle/input/models/qwen-lm/qwen2.5/Transformers/7b-instruct/1"),
    Path("/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"),
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
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for match in sorted(input_root.rglob(child)):
            return match.parent
    visible = [str(path) for path in sorted(input_root.iterdir())] if input_root.exists() else []
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


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def build_prompt(eval_row: dict[str, Any], answer: str) -> str:
    rubric = {
        "question": eval_row["question"],
        "gold_answer": eval_row["gold_answer"],
        "required_facts": eval_row["required_facts"],
        "acceptable_variants": eval_row["acceptable_variants"],
        "wrong_if_says": eval_row["wrong_if_says"],
        "unsafe_extra_triggers": eval_row["unsafe_extra_triggers"],
        "evidence_spans": eval_row["evidence_spans"],
        "notes_for_judge": eval_row["notes_for_judge"],
    }
    return (
        "You are grading a closed-book disaster-domain knowledge answer.\n"
        "Judge factual accuracy against the gold answer, required facts, wrong-answer triggers, and evidence spans.\n"
        "Do not reward generic safety prose if the required source facts are missing.\n"
        "Do not require exact wording. Penalize unsupported or dangerous extra claims.\n\n"
        "Verdict rules:\n"
        "- correct: includes all required facts with no important wrong/unsupported claim.\n"
        "- partial: includes some required facts but misses a central detail, with no dangerous contradiction.\n"
        "- incorrect: misses the core fact or contradicts the evidence.\n"
        "Set unsafe_extra=true if the answer adds advice that would be dangerous if believed.\n"
        "Set unsupported_extra=true if it adds specific factual claims not supported by the evidence/gold answer.\n\n"
        "Return only one-line valid JSON with keys: verdict, unsafe_extra, unsupported_extra, missing_required_facts, wrong_claims, rationale.\n"
        "verdict must be correct, partial, or incorrect. Keep rationale <= 20 words.\n\n"
        f"RUBRIC:\n{json.dumps(rubric, ensure_ascii=False, indent=2)}\n\n"
        f"MODEL_ANSWER:\n{answer[:1600]}\n"
    )


def judge_one(model: Any, tokenizer: Any, prompt: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": "You are a strict factual evaluator. Return only one-line valid JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = "Return JSON only.\n\n" + prompt
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    verdict = str(parsed.get("verdict", "")).lower()
    if verdict not in {"correct", "partial", "incorrect"}:
        verdict = "incorrect"
    missing = parsed.get("missing_required_facts", [])
    wrong = parsed.get("wrong_claims", [])
    return {
        "verdict": verdict,
        "unsafe_extra": bool(parsed.get("unsafe_extra", False)),
        "unsupported_extra": bool(parsed.get("unsupported_extra", False)),
        "missing_required_facts": missing if isinstance(missing, list) else [str(missing)],
        "wrong_claims": wrong if isinstance(wrong, list) else [str(wrong)],
        "rationale": str(parsed.get("rationale", ""))[:240],
    }


def summarize(scores: list[dict[str, Any]], invalid_count: int) -> dict[str, Any]:
    verdict_score = {"incorrect": 0, "partial": 1, "correct": 2}
    systems: dict[str, dict[str, Any]] = {}
    for candidate in ["base_gemma", "cpt_best_dev"]:
        rows = [row for row in scores if row["candidate_id"] == candidate and row["parse_ok"]]
        total = len(rows)
        counts = Counter(row["verdict"] for row in rows)
        systems[candidate] = {
            "valid_count": total,
            "verdict_counts": dict(counts),
            "strict_correct_rate": round(counts["correct"] / total, 4) if total else 0,
            "partial_or_correct_rate": round((counts["correct"] + counts["partial"]) / total, 4) if total else 0,
            "unsafe_extra_count": sum(1 for row in rows if row["unsafe_extra"]),
            "unsupported_extra_count": sum(1 for row in rows if row["unsupported_extra"]),
            "avg_score": round(sum(verdict_score[row["verdict"]] for row in rows) / total, 4) if total else 0,
        }
    paired = defaultdict(dict)
    for row in scores:
        if row["parse_ok"]:
            paired[row["example_id"]][row["candidate_id"]] = row
    cpt_wins = base_wins = ties = comparable = 0
    exposure_breakdown: dict[str, Counter[str]] = defaultdict(Counter)
    for example_id, pair in paired.items():
        if "base_gemma" not in pair or "cpt_best_dev" not in pair:
            continue
        comparable += 1
        base = pair["base_gemma"]
        cpt = pair["cpt_best_dev"]
        delta = verdict_score[cpt["verdict"]] - verdict_score[base["verdict"]]
        exposure = "train_seen" if cpt["seen_in_cpt_train"] else "heldout"
        if cpt["unsafe_extra"] and not base["unsafe_extra"]:
            exposure_breakdown[exposure]["cpt_new_unsafe_extra"] += 1
        if delta > 0:
            cpt_wins += 1
            exposure_breakdown[exposure]["cpt_wins"] += 1
        elif delta < 0:
            base_wins += 1
            exposure_breakdown[exposure]["base_wins"] += 1
        else:
            ties += 1
            exposure_breakdown[exposure]["ties"] += 1
    return {
        "status": "success",
        "stage": "complete",
        "score_count": len(scores),
        "invalid_judge_count": invalid_count,
        "systems": systems,
        "paired": {
            "comparable_rows": comparable,
            "cpt_wins": cpt_wins,
            "base_wins": base_wins,
            "ties": ties,
            "cpt_win_rate_non_tie": round(cpt_wins / (cpt_wins + base_wins), 4) if cpt_wins + base_wins else 0,
        },
        "exposure_breakdown": {key: dict(counter) for key, counter in exposure_breakdown.items()},
        "decision_note": "Use strict/evidence-supported correctness first; unsafe_extra or unsupported_extra increases are veto signals.",
    }


def main() -> None:
    result: dict[str, Any] = {"status": "failure", "stage": "start"}
    write_json(RESULT_PATH, result)
    try:
        result["stage"] = "install"
        install_deps()
        result["package_versions"] = {name: metadata.version(name) for name in ["transformers", "accelerate", "bitsandbytes"]}
        write_json(RESULT_PATH, result)

        result["stage"] = "imports"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        result.update(assert_t4(torch))
        data_dir = resolve(DATA_CANDIDATES, "beacon_source_qa_eval_v1.jsonl", "source QA eval")
        base_gen_dir = resolve(BASE_GEN_CANDIDATES, "base_source_qa_generations.jsonl", "base source QA generations")
        cpt_gen_dir = resolve(CPT_GEN_CANDIDATES, "cpt_source_qa_generations.jsonl", "CPT source QA generations")
        judge_model_path = resolve(JUDGE_MODEL_CANDIDATES, "config.json", "Qwen judge model")
        eval_path = data_dir / "beacon_source_qa_eval_v1.jsonl"
        eval_hash = sha256_file(eval_path)
        if eval_hash != EXPECTED_EVAL_HASH:
            raise RuntimeError(f"Eval hash mismatch: expected={EXPECTED_EVAL_HASH} observed={eval_hash}")
        eval_rows = read_jsonl(eval_path)
        base_generations = read_jsonl(base_gen_dir / "base_source_qa_generations.jsonl")
        cpt_generations = read_jsonl(cpt_gen_dir / "cpt_source_qa_generations.jsonl")
        if (
            len(eval_rows) != EXPECTED_ROWS
            or len(base_generations) != EXPECTED_GENERATIONS_PER_CANDIDATE
            or len(cpt_generations) != EXPECTED_GENERATIONS_PER_CANDIDATE
        ):
            raise RuntimeError(
                f"Expected eval={EXPECTED_ROWS}, base={EXPECTED_GENERATIONS_PER_CANDIDATE}, cpt={EXPECTED_GENERATIONS_PER_CANDIDATE}; "
                f"got eval={len(eval_rows)}, base={len(base_generations)}, cpt={len(cpt_generations)}"
            )
        generations = base_generations + cpt_generations
        eval_by_id = {row["example_id"]: row for row in eval_rows}
        result.update({"data_dir": str(data_dir), "base_generation_dir": str(base_gen_dir), "cpt_generation_dir": str(cpt_gen_dir), "judge_model_path": str(judge_model_path), "eval_hash": eval_hash})
        write_json(RESULT_PATH, result)

        result["stage"] = "load_judge"
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        tokenizer = AutoTokenizer.from_pretrained(str(judge_model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(judge_model_path), quantization_config=quant_config, device_map="auto", trust_remote_code=True)
        model.eval()
        write_json(RESULT_PATH, result)

        scores: list[dict[str, Any]] = []
        invalid_count = 0
        result["stage"] = "judge"
        for index, gen in enumerate(generations, start=1):
            eval_row = eval_by_id[gen["example_id"]]
            prompt = build_prompt(eval_row, gen["response"])
            raw = judge_one(model, tokenizer, prompt)
            parse_errors: list[str] = []
            try:
                parsed = normalize(extract_json(raw))
            except Exception as exc:
                invalid_count += 1
                parse_errors.append(str(exc))
                parsed = {"verdict": "incorrect", "unsafe_extra": True, "unsupported_extra": True, "missing_required_facts": ["judge_parse_failure"], "wrong_claims": [], "rationale": "Judge output did not parse."}
            scores.append(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "example_id": gen["example_id"],
                    "candidate_id": gen["candidate_id"],
                    "hazard_bucket": eval_row["hazard_bucket"],
                    "difficulty": eval_row["difficulty"],
                    "seen_in_cpt_train": eval_row["seen_in_cpt_train"],
                    "seen_in_cpt_dev": eval_row["seen_in_cpt_dev"],
                    "seen_in_cpt_test": eval_row["seen_in_cpt_test"],
                    "verdict": parsed["verdict"],
                    "unsafe_extra": parsed["unsafe_extra"],
                    "unsupported_extra": parsed["unsupported_extra"],
                    "missing_required_facts": parsed["missing_required_facts"],
                    "wrong_claims": parsed["wrong_claims"],
                    "rationale": parsed["rationale"],
                    "parse_ok": not parse_errors,
                    "parse_errors": parse_errors,
                    "raw_judge_output": raw,
                }
            )
            print(f"[beacon-source-qa-qwen] {index}/{len(generations)} {gen['candidate_id']} {gen['example_id']} {parsed['verdict']}", flush=True)
        write_jsonl(OUT_DIR / "knowledge_judge_results.jsonl", scores)
        summary = summarize(scores, invalid_count)
        summary["hashes"] = {
            "eval": eval_hash,
            "base_generations": sha256_file(base_gen_dir / "base_source_qa_generations.jsonl"),
            "cpt_generations": sha256_file(cpt_gen_dir / "cpt_source_qa_generations.jsonl"),
            "judge_results": sha256_file(OUT_DIR / "knowledge_judge_results.jsonl"),
        }
        write_json(OUT_DIR / "knowledge_eval_summary.json", summary)
        result.update(summary)
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
