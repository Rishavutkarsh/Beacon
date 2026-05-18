import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("data/assistant_sft")
TOOL_SOURCE = ROOT / "beacon_doc_tool_sft_v1_manual_candidate" / "all_approved_rows.jsonl"
TOOL_DECISION_DIR = ROOT / "beacon_doc_tool_sft_v1_manual_rewrite" / "main_review_packets"
TOOL_SEED_SOURCE = ROOT / "beacon_doc_tool_sft_v1_manual_rewrite" / "approved_seed_rows.jsonl"
NO_TOOL_SOURCE = ROOT / "beacon_no_tool_natural_sft_v1_candidate" / "all_auto_approved_rows.jsonl"
OUT = ROOT / "beacon_tool_plus_no_tool_sft_v1_final_reviewed"

SYSTEM_NO_TOOL = (
    "You are Beacon, an offline crisis companion for India-relevant disaster situations. "
    "Give conservative, practical guidance. State uncertainty clearly, do not invent live facts, "
    "and give safer next steps before escalation."
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_tool_decisions():
    decisions = {}
    for idx in range(1, 49):
        path = TOOL_DECISION_DIR / f"packet_{idx:02d}_decisions.jsonl"
        for row in read_jsonl(path):
            if row["decision"] != "approved":
                continue
            decisions[row["row_id"]] = row
    return decisions


def replace_last_assistant(messages, final_response):
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "assistant":
            out[i]["content"] = final_response
            return out
    raise ValueError("row has no assistant message")


def reviewed_tool_rows():
    source_by_id = {row["row_id"]: row for row in read_jsonl(TOOL_SOURCE)}
    decisions = load_tool_decisions()
    rows = []
    missing = []
    for row_id, decision in decisions.items():
        src = source_by_id.get(row_id)
        if not src:
            missing.append(row_id)
            continue
        row = dict(src)
        row["messages"] = replace_last_assistant(src["messages"], decision["final_response"])
        row["target_response"] = decision["final_response"]
        row["review_status"] = "approved_by_main_review"
        row["training_ready"] = True
        row["training_export_allowed"] = False
        row["main_review"] = {
            "decision": "approved",
            "notes": decision.get("main_review_notes", ""),
            "source": "main_review_packets",
        }
        row["mix_source"] = "tool_aware_main_reviewed"
        rows.append(row)
    rows.sort(key=lambda r: r["row_id"])
    seed_rows = []
    included_ids = {row["row_id"] for row in rows}
    for src in read_jsonl(TOOL_SEED_SOURCE):
        if src["row_id"] in included_ids:
            continue
        row = dict(src)
        row["review_status"] = "approved_by_main_review"
        row["training_ready"] = True
        row["training_export_allowed"] = False
        row["main_review"] = {
            "decision": "approved",
            "notes": "Previously approved seed no-support tool row; included in final reviewed set.",
            "source": "approved_seed_rows",
        }
        row["mix_source"] = "tool_aware_seed_approved"
        seed_rows.append(row)
    return rows + seed_rows, missing


def no_tool_selection_ids():
    ids = []
    ids += [f"beacon_no_tool_natural_sft_v1_packet_01_{i:04d}" for i in range(1, 61)]
    ids += [f"packet_03_post_flood_cleanup_no_tool_{i:04d}" for i in range(1, 61)]
    ids += [f"beacon_no_tool_natural_sft_v1_packet_04_generator_smell_co_{i:04d}" for i in range(1, 41)]
    ids += [f"beacon_no_tool_natural_sft_v1_packet_02_{i:04d}" for i in range(0, 24)]
    ids += ["beacon_no_tool_natural_sft_v1_packet_02_0025"]
    ids += [f"beacon_no_tool_natural_sft_v1_packet_05_{i:04d}" for i in range(1, 16)]
    return ids


NO_TOOL_RESPONSE_OVERRIDES = {
    "beacon_no_tool_natural_sft_v1_packet_02_0019": (
        "I cannot verify that ORS or tablets are available at the school right now. Do not cross floodwater or damaged roads just for a forwarded stock claim, especially with a cut on your foot. Cover the cut if you can, keep it out of floodwater, and send one short request to PHC/ASHA/panchayat with your location, people count, water/ORS need, the foot cut, mobility limits, and callback number."
    ),
    "beacon_no_tool_natural_sft_v1_packet_01_0052": (
        "I cannot certify that ward is safe for newborns now. Choose a dry, clean room away from wet wiring, damp walls, and floodwater. Keep newborn supplies, formula, medicines, and records dry. Ask facility leadership, maintenance, or local health officials to inspect and clean the ward before placing newborns there."
    ),
}


def normalize_no_tool_row(src):
    response = NO_TOOL_RESPONSE_OVERRIDES.get(src["row_id"], src["target_response"])
    row = {
        "schema_version": "beacon-tool-plus-no-tool-sft-v1-reviewed-mix",
        "row_id": src["row_id"],
        "source_row_id": src["row_id"],
        "packet_id": src.get("packet_id"),
        "scenario_family": src.get("scenario_family"),
        "split": src["split"],
        "hazard": src["hazard"],
        "row_family": "no_tool_needed",
        "language": src.get("language"),
        "risk_level": src.get("risk_level"),
        "tool_required": False,
        "training_ready": True,
        "training_export_allowed": False,
        "review_status": "approved_by_main_review",
        "main_review": {
            "decision": "approved",
            "notes": "Selected from no-tool natural candidate after row-by-row review; edited where needed.",
            "source": "beacon_no_tool_natural_sft_v1_candidate",
        },
        "mix_source": "no_tool_natural_main_reviewed",
        "user_prompt": src["user_prompt"],
        "target_response": response,
        "messages": [
            {"role": "system", "content": SYSTEM_NO_TOOL},
            {"role": "user", "content": src["user_prompt"]},
            {"role": "assistant", "content": response},
        ],
        "must_include": src.get("must_include", []),
        "must_avoid": src.get("must_avoid", []),
    }
    return row


def reviewed_no_tool_rows():
    source_by_id = {row["row_id"]: row for row in read_jsonl(NO_TOOL_SOURCE)}
    rows = []
    missing = []
    for row_id in no_tool_selection_ids():
        src = source_by_id.get(row_id)
        if not src:
            missing.append(row_id)
            continue
        rows.append(normalize_no_tool_row(src))
    return rows, missing


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tool_rows, missing_tool = reviewed_tool_rows()
    no_tool_rows, missing_no_tool = reviewed_no_tool_rows()
    rows = tool_rows + no_tool_rows

    split_rows = {"train": [], "dev": [], "final_eval": []}
    for row in rows:
        split_rows[row["split"]].append(row)

    write_jsonl(OUT / "all_rows.jsonl", rows)
    for split, split_data in split_rows.items():
        write_jsonl(OUT / f"{split}.jsonl", split_data)

    row_family_counts = Counter(row.get("row_family") for row in rows)
    counts = {
        "total": len(rows),
        "tool_source_rows_including_seed": len(tool_rows),
        "tool_call_or_evidence_rows": len(rows) - row_family_counts.get("no_tool_needed", 0),
        "no_tool_total_rows": row_family_counts.get("no_tool_needed", 0),
        "no_tool_rows_added_from_natural_candidate": len(no_tool_rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_mix_source": dict(Counter(row["mix_source"] for row in rows)),
        "by_row_family": dict(row_family_counts),
        "by_hazard": dict(Counter(row.get("hazard") for row in rows)),
    }
    manifest = {
        "schema_version": "beacon-tool-plus-no-tool-sft-v1-reviewed-mix",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "main_reviewed_candidate_not_training_launched",
        "training_export_allowed": False,
        "base_recommendation": "train on SFT v1 ckpt300 best, keep CPT ckpt300 as knowledge reference baseline",
        "counts": counts,
        "sources": {
            "tool_rows": str(TOOL_SOURCE),
            "tool_seed_rows": str(TOOL_SEED_SOURCE),
            "tool_decisions": str(TOOL_DECISION_DIR),
            "no_tool_rows": str(NO_TOOL_SOURCE),
        },
        "missing_tool_rows_from_full_trace_source": missing_tool,
        "missing_no_tool_rows_from_candidate_source": missing_no_tool,
        "selection_policy": [
            "Use all full-trace tool-aware rows that had main-review decisions and were present in the full-trace candidate source.",
            "Include the 48 previously approved seed no-support tool rows.",
            "Add 200 reviewed no-tool rows: all packet 01, all packet 03, first 40 packet 04 CO rows, 25 packet 02 rumor/live-status rows, and first 15 packet 05 shelter-conflict rows.",
            "Downsample CO and shelter-conflict to avoid overtraining one behavior.",
            "Keep exact thresholds and official facts in the tool-aware lane; no-tool rows give broad practical behavior only.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    report = [
        "# Beacon Tool + No-Tool SFT v1 Reviewed Mix",
        "",
        "This package combines main-reviewed tool-aware rows with a reviewed slice of the new no-tool natural SFT candidate rows.",
        "",
        f"- Total rows: {counts['total']}",
        f"- Tool-call/evidence rows: {counts['tool_call_or_evidence_rows']}",
        f"- No-tool rows total: {counts['no_tool_total_rows']}",
        f"- No-tool rows added from natural candidate: {counts['no_tool_rows_added_from_natural_candidate']}",
        f"- Splits: {counts['by_split']}",
        "",
        "No training has been launched. `training_export_allowed` remains false for final inspection.",
        "",
        "Main no-tool edits applied:",
        "- Rewrote the ORS/tablet stock row with a cut foot so it no longer uses a medicine-dose request template.",
        "- Normalized the clinic newborn ward prompt to avoid teaching 'tool' language in a no-tool row.",
        "",
        "Known residual risk:",
        "- The full-trace tool-aware source was missing 10 row IDs from the 1152 reviewed packet decisions, so those 10 are not included here.",
        "- Packet 05 shelter-conflict rows are intentionally downsampled to 15 because the packet still has repeated patterns.",
    ]
    (OUT / "review_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
