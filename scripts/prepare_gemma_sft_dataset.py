from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUT = "kaggle/input/sankat-saathi-text-sft-approved"
SYSTEM_PROMPT = (
    "You are Sankat Saathi, an offline crisis companion. Give conservative, practical, "
    "structured guidance. Never claim uncertain food, water, or medicine is definitely safe."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_to_text(answer: dict) -> str:
    sections = [
        ("risk_level", answer.get("risk_level", "")),
        ("immediate_action", answer.get("immediate_action", [])),
        ("resource_plan", answer.get("resource_plan", [])),
        ("unsafe_items", answer.get("unsafe_items", [])),
        ("missing_information", answer.get("missing_information", [])),
        ("escalation_signs", answer.get("escalation_signs", [])),
        ("what_not_to_do", answer.get("what_not_to_do", [])),
        ("hindi_hinglish", answer.get("hindi_hinglish", [])),
        ("uncertainty_note", answer.get("uncertainty_note", "")),
    ]
    lines: list[str] = []
    for key, value in sections:
        if isinstance(value, list):
            lines.append(f"{key}: " + " | ".join(str(item) for item in value))
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def clean_item(text: object) -> str:
    value = str(text).strip()
    value = re.sub(r"\s+", " ", value)
    unsafe_match = re.match(r"^Correct the unsafe assumption:\s*(.+?)(?:\.|$)", value, flags=re.IGNORECASE)
    if unsafe_match:
        assumption = unsafe_match.group(1).strip()
        value = re.sub(r"^Correct the unsafe assumption:\s*.+?(?:\.|$)", "", value, count=1, flags=re.IGNORECASE).strip()
        value = f"Do not assume that {assumption[0].lower() + assumption[1:]}. {value}".strip()
    else:
        value = re.sub(r"^Correct the unsafe assumption:\s*", "", value, flags=re.IGNORECASE)
    value = value.replace("This is not a safe basis for action.", "").strip()
    value = re.sub(r"\s+\.", ".", value)
    if len(value) > 180:
        cutoff = max(value.rfind(".", 0, 180), value.rfind(";", 0, 180), value.rfind(",", 0, 180))
        if cutoff < 80:
            cutoff = value.rfind(" ", 0, 180)
        value = value[:cutoff if cutoff > 80 else 180].strip() + "."
    return value.strip(" |")


def clean_list(values: object, limit: int) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    cleaned: list[str] = []
    for item in values:
        text = clean_item(item)
        if not text:
            continue
        if text.lower() in {existing.lower() for existing in cleaned}:
            continue
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def sentence_join(items: list[str]) -> str:
    sentences = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item[-1] not in ".!?":
            item += "."
        sentences.append(item)
    return " ".join(sentences)


def answer_to_clean_v2_text(answer: dict, language_mix: str | None = None) -> str:
    risk = clean_item(answer.get("risk_level", "high")).lower()
    immediate = clean_list(answer.get("immediate_action", []), 3)
    resources = clean_list(answer.get("resource_plan", []), 1)
    unsafe = clean_list(answer.get("unsafe_items", []), 1)
    missing = clean_list(answer.get("missing_information", []), 1)
    escalation = clean_list(answer.get("escalation_signs", []), 3)
    dont = clean_list(answer.get("what_not_to_do", []), 2)
    uncertainty = clean_item(answer.get("uncertainty_note", "Cannot determine full safety from the information given. Choose the lower-risk option and seek local help when reachable."))
    hinglish = clean_list(answer.get("hindi_hinglish", []), 2)

    lines = [
        f"risk_level: {risk}",
        f"immediate_action: {sentence_join(immediate)}",
        f"resource_plan: {sentence_join(resources)}",
        f"unsafe_items: {sentence_join(unsafe)}",
        f"missing_information: {sentence_join(missing)}",
        f"escalation_signs: {', '.join(escalation)}.",
        f"what_not_to_do: {sentence_join(dont)}",
    ]
    if language_mix in {"hinglish", "hindi", "bilingual"} and hinglish:
        lines.append(f"hindi_hinglish: {sentence_join(hinglish)}")
    else:
        lines.append("hindi_hinglish: Use simple local language if asked.")
    lines.append(f"uncertainty_note: {uncertainty}")
    return "\n".join(lines)


def answer_to_rendered_text(row: dict, render_style: str) -> str:
    if render_style == "legacy":
        return answer_to_text(row["assistant_response"])
    if render_style == "clean_v2":
        return answer_to_clean_v2_text(row["assistant_response"], row.get("language_mix"))
    raise ValueError(f"Unsupported render style: {render_style}")


def response_lint(records: list[dict]) -> dict:
    responses = [record["response"].replace("<turn|>", "") for record in records]
    lengths = sorted(len(response) for response in responses)
    unexpected_script = re.compile(r"[\u0E00-\u0E7F\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")
    markup = re.compile(r"</?\w+>")
    def count(predicate):
        return sum(1 for response in responses if predicate(response))
    report = {
        "count": len(responses),
        "length_chars": {
            "min": lengths[0] if lengths else 0,
            "p50": lengths[len(lengths) // 2] if lengths else 0,
            "p90": lengths[int((len(lengths) - 1) * 0.90)] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "over_700": sum(length > 700 for length in lengths),
            "over_1200": sum(length > 1200 for length in lengths),
        },
        "pipe_count": count(lambda response: "|" in response),
        "unsafe_assumption_phrase_count": count(lambda response: "Correct the unsafe assumption" in response),
        "markup_count": count(lambda response: bool(markup.search(response))),
        "unexpected_script_count": count(lambda response: bool(unexpected_script.search(response))),
    }
    errors = []
    if report["pipe_count"]:
        errors.append(f"pipe separators found in {report['pipe_count']} responses")
    if report["unsafe_assumption_phrase_count"] > max(3, len(responses) // 100):
        errors.append("too many 'Correct the unsafe assumption' phrases")
    if report["markup_count"]:
        errors.append(f"markup found in {report['markup_count']} responses")
    if report["unexpected_script_count"]:
        errors.append(f"unexpected script contamination found in {report['unexpected_script_count']} responses")
    if report["length_chars"]["over_1200"]:
        errors.append(f"{report['length_chars']['over_1200']} responses exceed 1200 chars")
    report["errors"] = errors
    return report


def gemma_prompt(user_prompt: str) -> str:
    return (
        f"<|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{user_prompt}<turn|>\n"
        "<|turn>model\n"
    )


def make_record(row: dict, render_style: str = "legacy") -> dict:
    prompt = gemma_prompt(row["user_prompt"])
    response_text = answer_to_rendered_text(row, render_style)
    response = response_text + "<turn|>"
    return {
        "id": row["example_id"],
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": row["user_prompt"],
            },
            {"role": "assistant", "content": response_text},
        ],
        "source_ids": row["source_ids"],
        "guidance_fact_ids": row["guidance_fact_ids"],
        "risk_tags": row["risk_tags"],
        "language_mix": row.get("language_mix"),
        "eval_rubric": row.get("eval_rubric", {}),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare approved text SFT data for a Kaggle Gemma LoRA run.")
    parser.add_argument("--dataset-dir", default="data/processed/hardened_text")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--render-style", choices=["legacy", "clean_v2"], default="legacy")
    parser.add_argument("--dataset-id", default="rishavutkarsh/sankat-saathi-text-sft-approved")
    parser.add_argument("--dataset-title", default="Sankat Saathi Text SFT Approved")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / "sft_text_approved.jsonl"
    eval_path = dataset_dir / "eval_text_approved.jsonl"
    gate_path = dataset_dir / "pre_training_review.json"
    if not train_path.exists() or not eval_path.exists():
        raise SystemExit("Approved text SFT exports are missing. Run export_approved.py --mode text --task sft first.")
    if not gate_path.exists():
        raise SystemExit("Missing pre_training_review.json. Run the text pre-training gate before packaging.")

    train_rows = read_jsonl(train_path)[: args.train_limit]
    eval_rows = read_jsonl(eval_path)[: args.eval_limit]
    if not train_rows:
        raise SystemExit("No approved training rows found.")
    if not eval_rows:
        raise SystemExit("No approved eval rows found.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_records = [make_record(row, args.render_style) for row in train_rows]
    eval_records = [make_record(row, args.render_style) for row in eval_rows]
    train_lint = response_lint(train_records)
    eval_lint = response_lint(eval_records)
    lint_report = {"render_style": args.render_style, "train": train_lint, "eval": eval_lint}
    (out_dir / "response_lint_report.json").write_text(json.dumps(lint_report, indent=2), encoding="utf-8")
    lint_errors = train_lint["errors"] + eval_lint["errors"]
    if args.render_style == "clean_v2" and lint_errors:
        raise SystemExit("clean_v2 response lint failed: " + "; ".join(lint_errors[:10]))
    write_jsonl(out_dir / "train.jsonl", train_records)
    write_jsonl(out_dir / "eval.jsonl", eval_records)

    training_config = {
        "max_length": args.max_length,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "eval_interval_steps": 25,
        "curve_eval_batches": 30,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "completion_only_loss": True,
        "render_style": args.render_style,
        "notes": "Prepared locally so Kaggle GPU time is used for model loading, tokenization, and training only.",
    }
    (out_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_dir": str(dataset_dir),
        "source_train_export": str(train_path),
        "source_eval_export": str(eval_path),
        "pre_training_review": json.loads(gate_path.read_text(encoding="utf-8")),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_ids": [row["example_id"] for row in train_rows],
        "eval_ids": [row["example_id"] for row in eval_rows],
        "render_style": args.render_style,
        "response_lint_report": lint_report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset_metadata = {
        "title": args.dataset_title,
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(train_rows)} train rows and {len(eval_rows)} eval rows to {out_dir}")


if __name__ == "__main__":
    main()
