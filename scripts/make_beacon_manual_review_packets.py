from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1_manual_rewrite"
SOURCE_PACKAGE = ROOT / "data" / "assistant_sft" / "beacon_doc_tool_sft_v1"
OUT_DIR = WORK_DIR / "main_review_packets"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tool_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for message in row.get("messages", []):
        if message.get("role") == "tool" and message.get("name") == "read_official_doc":
            payload = json.loads(str(message.get("content", "{}")))
            sections.extend(payload.get("sections", []))
    return sections


def compact_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for section in tool_sections(row)[:3]:
        snippet = str(section.get("snippet", "")).replace("\n", " ")
        out.append(
            {
                "section_id": section.get("section_id"),
                "key_facts": section.get("key_facts", []),
                "snippet": snippet[:360],
            }
        )
    return out


def main() -> None:
    source_rows = {row["row_id"]: row for row in read_jsonl(SOURCE_PACKAGE / "all_rows.jsonl")}
    outputs: dict[str, dict[str, Any]] = {}
    for idx in range(1, 7):
        for row in read_jsonl(WORK_DIR / "shards" / f"shard_{idx:02d}_output.jsonl"):
            outputs[row["row_id"]] = row
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    packets = []
    for row_id in sorted(outputs):
        source = source_rows[row_id]
        output = outputs[row_id]
        packets.append(
            {
                "row_id": row_id,
                "split": source.get("split"),
                "row_family": source.get("row_family"),
                "case_family_id": source.get("case_family_id"),
                "hazard": source.get("hazard"),
                "user_prompt": source.get("user_prompt"),
                "expected_facts": source.get("expected_facts", []),
                "sections": compact_sections(source),
                "worker_decision": output.get("decision"),
                "worker_response": output.get("final_response", ""),
                "worker_notes": output.get("notes", ""),
            }
        )
    for start in range(0, len(packets), 24):
        packet = packets[start : start + 24]
        path = OUT_DIR / f"packet_{start // 24 + 1:02d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in packet), encoding="utf-8")
    manifest = {
        "packet_count": (len(packets) + 23) // 24,
        "row_count": len(packets),
        "rows_per_packet": 24,
        "review_instruction": "Main reviewer must read each packet and write decisions; scripts are packet formatting only.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
