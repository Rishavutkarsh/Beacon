from __future__ import annotations

from scripts import beacon_ollama_tool_loop_agent as agent


class FakeTools:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.read_calls: list[dict[str, object]] = []

    def search(self, query: str, hazard: str | None = None, organization: str | None = None, top_k: int = 4):
        self.search_calls.append(
            {"query": query, "hazard": hazard, "organization": organization, "top_k": top_k}
        )
        return {
            "documents": [
                {
                    "doc_id": "cdc_food_after_emergency",
                    "title": "Keep Food Safe After a Disaster or Emergency",
                    "organization": "CDC",
                    "hazards": ["food_safety", "floodwater"],
                }
            ]
        }

    def read(self, doc_id: str, section_or_page_query: str, top_k: int = 5):
        self.read_calls.append(
            {"doc_id": doc_id, "section_or_page_query": section_or_page_query, "top_k": top_k}
        )
        return {
            "sections": [
                {
                    "doc_id": doc_id,
                    "section_id": "cdc_food_after_emergency_chunk_0002",
                    "title": "Keep Food Safe After a Disaster or Emergency",
                    "text": "Throw out wooden cutting boards, baby bottle nipples, and pacifiers if they have come into contact with floodwaters.",
                    "key_facts": [],
                }
            ]
        }


def test_parse_tool_call_accepts_canonical_json() -> None:
    text = '<tool_call>{"name":"search_official_docs","arguments":{"query":"food_safety floodwater","hazard":"food_safety","organization":null,"top_k":4}}</tool_call>'

    call = agent.parse_tool_call(text)

    assert call is not None
    assert call["name"] == "search_official_docs"
    assert call["arguments"]["hazard"] == "food_safety"


def test_parse_tool_call_rejects_prose_or_multiple_calls() -> None:
    call = '<tool_call>{"name":"search_official_docs","arguments":{"query":"food","hazard":"food_safety","organization":null,"top_k":4}}</tool_call>'

    assert agent.parse_tool_call("Here is the call:\n" + call) is None
    assert agent.parse_tool_call(call + "\n" + call) is None


def test_run_tool_call_passes_canonical_search_args() -> None:
    tools = FakeTools()
    state: dict[str, object] = {}
    call = {
        "name": "search_official_docs",
        "arguments": {
            "query": "carbon_monoxide generators outdoors 20 feet",
            "hazard": "carbon_monoxide",
            "organization": None,
            "top_k": 4,
        },
    }

    result = agent.run_tool_call(tools, call, state)  # type: ignore[arg-type]

    assert result["documents"][0]["doc_id"] == "cdc_food_after_emergency"
    assert tools.search_calls == [
        {
            "query": "carbon_monoxide generators outdoors 20 feet",
            "hazard": "carbon_monoxide",
            "organization": None,
            "top_k": 4,
        }
    ]
    assert state["search_doc_ids"] == {"cdc_food_after_emergency"}


def test_read_accepts_section_or_page_query_and_requires_search_doc_id() -> None:
    tools = FakeTools()
    state: dict[str, object] = {"search_doc_ids": {"cdc_food_after_emergency"}}
    call = {
        "name": "read_official_doc",
        "arguments": {
            "doc_id": "cdc_food_after_emergency",
            "section_or_page_query": "baby bottle nipples pacifiers floodwater",
            "top_k": 5,
        },
    }

    error = agent.validate_tool_call(call, state)
    result = agent.run_tool_call(tools, call, state)  # type: ignore[arg-type]

    assert error is None
    assert result["sections"][0]["section_id"] == "cdc_food_after_emergency_chunk_0002"
    assert tools.read_calls == [
        {
            "doc_id": "cdc_food_after_emergency",
            "section_or_page_query": "baby bottle nipples pacifiers floodwater",
            "top_k": 5,
        }
    ]


def test_read_rejects_legacy_section_or_query_in_strict_mode() -> None:
    call = {
        "name": "read_official_doc",
        "arguments": {
            "doc_id": "cdc_food_after_emergency",
            "section_or_query": "baby bottle nipples pacifiers floodwater",
            "top_k": 5,
        },
    }

    error = agent.validate_tool_call(call, {"search_doc_ids": {"cdc_food_after_emergency"}})

    assert error == "read_official_doc requires exactly doc_id, section_or_page_query, and top_k."


def test_read_before_search_is_rejected() -> None:
    call = {
        "name": "read_official_doc",
        "arguments": {"doc_id": "cdc_food_after_emergency", "section_or_page_query": "food", "top_k": 5},
    }

    error = agent.validate_tool_call(call, {"search_doc_ids": set()})

    assert error == "Call search_official_docs before read_official_doc."


def test_placeholder_doc_id_is_rejected() -> None:
    call = {
        "name": "read_official_doc",
        "arguments": {"doc_id": "DOC_ID", "section_or_page_query": "food", "top_k": 5},
    }

    error = agent.validate_tool_call(call, {"search_doc_ids": {"DOC_ID"}})

    assert error is not None
    assert "placeholder" in error


def test_tool_loop_allows_search_then_read_then_final_answer() -> None:
    tools = FakeTools()
    outputs = iter(
        [
            '<tool_call>{"name":"search_official_docs","arguments":{"query":"food_safety floodwater baby bottle nipples pacifiers","hazard":"food_safety","organization":null,"top_k":4}}</tool_call>',
            '<tool_call>{"name":"read_official_doc","arguments":{"doc_id":"cdc_food_after_emergency","section_or_page_query":"baby bottle nipples pacifiers floodwater","top_k":5}}</tool_call>',
            "No. Throw them out.\n\nCitations:\n- cdc_food_after_emergency / cdc_food_after_emergency_chunk_0002",
        ]
    )

    answer = agent.run_tool_loop(
        "Floodwater touched baby bottle nipples; can I bleach sanitize them?",
        tools,  # type: ignore[arg-type]
        lambda _prompt: next(outputs),
        max_tool_calls=3,
    )

    assert answer.startswith("No. Throw them out.")
    assert len(tools.search_calls) == 1
    assert len(tools.read_calls) == 1


def test_tool_loop_recovers_from_tool_call_with_extra_prose() -> None:
    tools = FakeTools()
    outputs = iter(
        [
            'Here: <tool_call>{"name":"search_official_docs","arguments":{"query":"food","hazard":"food_safety","organization":null,"top_k":4}}</tool_call>',
            '<tool_call>{"name":"search_official_docs","arguments":{"query":"food_safety floodwater","hazard":"food_safety","organization":null,"top_k":4}}</tool_call>',
            "Final answer after correction.",
        ]
    )

    answer = agent.run_tool_loop(
        "What do docs say about floodwater food?",
        tools,  # type: ignore[arg-type]
        lambda _prompt: next(outputs),
        max_tool_calls=3,
    )

    assert answer == "Final answer after correction."
    assert len(tools.search_calls) == 1


def test_second_search_after_documents_is_rejected() -> None:
    call = {
        "name": "search_official_docs",
        "arguments": {"query": "water_safety boil", "hazard": "water_safety", "organization": None, "top_k": 4},
    }

    error = agent.validate_tool_call(call, {"search_doc_ids": {"cdc_emergency_water"}})

    assert error == "After search_official_docs returns documents, call read_official_doc before searching again."


def test_force_docs_changes_initial_prompt_to_require_search() -> None:
    prompt = agent.initial_tool_loop_prompt("How long is fridge food safe?", force_docs=True)

    assert "Offline official documents are required" in prompt
    assert "search_official_docs" in prompt


def test_force_docs_rejects_direct_answer_before_tool_use() -> None:
    tools = FakeTools()
    outputs = iter(
        [
            "Direct answer without docs.",
            '<tool_call>{"name":"search_official_docs","arguments":{"query":"food_safety floodwater","hazard":"food_safety","organization":null,"top_k":4}}</tool_call>',
            "Final answer after tool use.",
        ]
    )

    answer = agent.run_tool_loop(
        "What do official docs say about floodwater food?",
        tools,  # type: ignore[arg-type]
        lambda _prompt: next(outputs),
        max_tool_calls=3,
        force_docs=True,
    )

    assert answer == "Final answer after tool use."
    assert len(tools.search_calls) == 1


def test_live_status_regex_catches_current_route_questions() -> None:
    question = "Is the bridge open now and safe now after flood?"

    assert agent.LIVE_STATUS_RE.search(question)
    assert "cannot verify live/current status" in agent.live_status_boundary_response(question)
