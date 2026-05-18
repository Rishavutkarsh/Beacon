from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sankat_saathi_dataset.local_doc_tool import (
    build_indexes,
    build_tool_sft_package,
    load_doc_index,
    load_section_index,
    read_official_doc,
    search_official_docs,
    validate_tool_sft_rows,
)


def out_dir(name: str) -> Path:
    path = Path("test_runs") / "official_doc_tool" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_official_doc_tool_builds_doc_and_page_indexes() -> None:
    target = out_dir("indexes")
    result = build_indexes(target)
    assert result.errors == []
    assert result.manifest["doc_count"] >= 24
    assert result.manifest["section_count"] >= result.manifest["doc_count"]
    docs = read_jsonl(target / "official_doc_index.jsonl")
    chunks = read_jsonl(target / "official_doc_chunk_index.jsonl")
    pages = read_jsonl(target / "official_doc_page_index.jsonl")
    assert any(row["jurisdiction"] == "india" for row in docs)
    assert result.manifest["india_fallback_waivers"]
    assert result.manifest["page_index_status"] == "not_available_source_chunks_only"
    assert all(row["training_export_allowed"] is False for row in docs)
    assert all(row["training_export_allowed"] is False for row in chunks)
    assert all(row["official_tier"] and row["selection_role"] and row["selection_rationale"] for row in docs)
    assert all(row["canonical_url"] and row["can_retrieve"] is True for row in docs)
    assert all(row["chunk_index_id"] and row["chunk_id"] and row["heading_path"] for row in chunks)
    assert pages == []
    assert not any("â" in row["text"] or "Â" in row["text"] for row in chunks)


def test_official_doc_tool_retrieves_exact_food_constant() -> None:
    target = out_dir("retrieval")
    build_indexes(target)
    docs = load_doc_index(target)
    sections = load_section_index(target)
    doc_hits = search_official_docs("fridge power outage 40 degrees 4 hours", docs, hazard="food_safety", top_k=4)
    assert doc_hits
    section_hits = read_official_doc(doc_hits[0].doc_id, "fridge power outage 40 degrees 4 hours", sections, top_k=5)
    evidence = " ".join([hit.snippet for hit in section_hits]).lower()
    assert "4 hours" in evidence
    assert "40 degrees" in evidence


def test_doc_tool_sft_has_explicit_tool_traces() -> None:
    index_target = out_dir("sft_indexes")
    package_target = out_dir("sft_package")
    build_indexes(index_target)
    result = build_tool_sft_package(package_target, index_target)
    assert result.errors == []
    assert result.warnings == []
    assert result.manifest["row_count"] == 1200
    assert result.manifest["tool_required_count"] == 960
    assert result.manifest["query_rewrite_count"] >= 420
    assert result.manifest["unique_user_prompt_count"] >= 1140
    assert result.manifest["unique_target_response_count"] >= 1080
    assert result.manifest["by_split"] == {"train": 960, "dev": 120, "final_eval": 120}
    rows = read_jsonl(package_target / "all_rows.jsonl")
    tool_rows = [row for row in rows if row["tool_required"]]
    no_tool_rows = [row for row in rows if not row["tool_required"]]
    assert tool_rows and no_tool_rows
    assert any(row["tool_required"] for row in rows if row["split"] == "dev")
    assert any(row["tool_required"] for row in rows if row["split"] == "final_eval")
    assert any(row["row_family"] in {"tool_grounded", "query_rewrite_tool_grounded"} for row in rows if row["split"] == "dev")
    assert any(row["row_family"] in {"tool_grounded", "query_rewrite_tool_grounded"} for row in rows if row["split"] == "final_eval")
    assert any(row["row_family"] in {"tool_no_support", "query_rewrite_tool_no_support"} for row in rows if row["split"] == "dev")
    assert any(row["row_family"] in {"tool_no_support", "query_rewrite_tool_no_support"} for row in rows if row["split"] == "final_eval")
    assert any(row["query_rewrite_required"] for row in rows if row["split"] == "dev")
    assert any(row["query_rewrite_required"] for row in rows if row["split"] == "final_eval")
    train = [row for row in rows if row["split"] == "train"]
    heldout = [row for row in rows if row["split"] != "train"]
    train_signatures = {
        (row["row_family"], " ".join(row["user_prompt"].lower().split()), row["tool_query"])
        for row in train
    }
    train_targets = {row["target_response"] for row in train}
    assert not any(
        (row["row_family"], " ".join(row["user_prompt"].lower().split()), row["tool_query"]) in train_signatures
        for row in heldout
    )
    assert not any(row["target_response"] in train_targets for row in heldout)
    for row in tool_rows:
        serialized_read = [
            json.loads(msg["content"])
            for msg in row["messages"]
            if msg.get("role") == "tool" and msg.get("name") == "read_official_doc"
        ][0]
        serialized_section_ids = {section["section_id"] for section in serialized_read["sections"]}
        assert set(row["section_ids"]).issubset(serialized_section_ids)
    rewrite_rows = [row for row in rows if row["query_rewrite_required"]]
    assert rewrite_rows
    for row in rewrite_rows[:20]:
        search_call = next(msg for msg in row["messages"] if msg["role"] == "assistant" and "search_official_docs" in msg["content"])
        assert row["user_prompt"].lower() not in search_call["content"].lower()
    assert all(any(msg.get("role") == "tool" and msg.get("name") == "search_official_docs" for msg in row["messages"]) for row in tool_rows)
    assert all(any(msg.get("role") == "tool" and msg.get("name") == "read_official_doc" for msg in row["messages"]) for row in tool_rows)
    assert all(not any(msg.get("role") == "tool" for msg in row["messages"]) for row in no_tool_rows)
    assert validate_tool_sft_rows(rows, index_target).errors == []


def test_doc_tool_scripts_allow_blocked() -> None:
    index_target = out_dir("script_indexes")
    package_target = out_dir("script_package")
    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_beacon_official_doc_tool.py",
            "--out-dir",
            str(index_target),
            "--allow-blocked",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    sft = subprocess.run(
        [
            sys.executable,
            "scripts/build_beacon_doc_tool_sft.py",
            "--index-dir",
            str(index_target),
            "--out-dir",
            str(package_target),
            "--target-rows",
            "1200",
            "--allow-blocked",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert sft.returncode == 0, sft.stdout + sft.stderr
