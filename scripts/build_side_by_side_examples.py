from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_prompt(rows: list[dict]) -> dict[str, dict]:
    return {row["prompt_id"]: row for row in rows}


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human-readable base-vs-adapter examples.")
    parser.add_argument("--prompts", default="data/eval/sankat_saathi_e2b_smoke.jsonl")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    prompts = read_jsonl(Path(args.prompts))
    base = by_prompt(read_jsonl(Path(args.base)))
    adapter = by_prompt(read_jsonl(Path(args.adapter)))
    lines = ["# Sankat Saathi Base vs Adapter Examples", ""]
    count = 0
    for prompt in prompts:
        prompt_id = prompt["prompt_id"]
        if prompt_id not in base or prompt_id not in adapter:
            continue
        lines.extend(
            [
                f"## {prompt_id}: {prompt.get('category')}",
                "",
                f"**Prompt:** {prompt['user_prompt']}",
                "",
                "**Base Gemma 4 E2B-IT:**",
                "",
                truncate(base[prompt_id].get("response", ""), 1800),
                "",
                "**Adapter:**",
                "",
                truncate(adapter[prompt_id].get("response", ""), 1800),
                "",
            ]
        )
        count += 1
        if count >= args.limit:
            break
    if count == 0:
        raise SystemExit("no matching prompt ids found between base and adapter generations")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {count} examples to {args.out}")


if __name__ == "__main__":
    main()
