import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def load_decisions(packet_dir):
    decisions = {}
    for path in packet_dir.glob("packet_*_auto_decisions.jsonl"):
        for row in load_jsonl(path):
            decisions[row["row_id"]] = row
    return decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/assistant_sft/beacon_no_tool_natural_sft_v1_candidate"))
    args = parser.parse_args()

    decisions = load_decisions(args.packet_dir)
    approved = []
    rejected = []
    for path in sorted(args.packet_dir.glob("packet_*_candidates.jsonl")):
        for row in load_jsonl(path):
            decision = decisions.get(row["row_id"], {"status": "reject", "errors": ["missing_auto_decision"]})
            row = dict(row)
            row["strict_review"] = {
                "auto_gate": decision,
                "main_review": {
                    "status": "auto_pass_pending_human_spotcheck" if decision["status"] == "needs_human_review" else "rejected_by_auto_gate",
                    "notes": "No-tool strict gate passed; still not export-ready until final human review." if decision["status"] == "needs_human_review" else "Rejected by no-tool strict gate.",
                },
            }
            if decision["status"] == "needs_human_review":
                approved.append(row)
            else:
                rejected.append(row)

    rows_by_split = defaultdict(list)
    for row in approved:
        rows_by_split[row.get("split", "train")].append(row)

    save_jsonl(args.out_dir / "all_auto_approved_rows.jsonl", approved)
    save_jsonl(args.out_dir / "rejected_rows.jsonl", rejected)
    save_jsonl(args.out_dir / "train.jsonl", rows_by_split["train"])
    save_jsonl(args.out_dir / "dev.jsonl", rows_by_split["dev"])
    save_jsonl(args.out_dir / "final_eval.jsonl", rows_by_split["final_eval"])

    manifest = {
        "schema_version": "beacon-no-tool-natural-sft-v1-candidate",
        "status": "auto_assembled_needs_human_final_approval",
        "source_packet_dir": str(args.packet_dir),
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "by_packet": dict(Counter(r.get("packet_id") for r in approved)),
        "by_split": dict(Counter(r.get("split") for r in approved)),
        "by_hazard": dict(Counter(r.get("hazard") for r in approved)),
        "training_export_allowed": False,
        "notes": [
            "No tool calls are allowed in this dataset.",
            "Rows are for natural behavior SFT: answer directly or refuse live verification without retrieval.",
            "Final human review should still reject awkward, repetitive, or under-specified rows.",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
