"""Tests for the deep research graph (planner, fan-out, brief) — offline."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from langgraph_deep_research.brief import build_brief
from langgraph_deep_research.graph import build_deep_research_graph
from langgraph_deep_research.planner import planner_node


def test_planner_node_splits_topic(monkeypatch):
    """planner_node returns sub_queries from the model output."""

    class FakeModel:
        def invoke(self, prompt):
            return AIMessage(content="Query 1\nQuery 2\nQuery 3")

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )
    state = {
        "messages": [("user", "research quantum computing")],
        "sub_queries": [],
        "evidence": [],
        "brief": "",
    }
    result = planner_node(state)
    assert result["sub_queries"] == ["Query 1", "Query 2", "Query 3"]
    assert "Sub-queries" in result["messages"][0].content


def test_build_brief_with_evidence():
    evidence = [
        {"url": "https://a.com", "title": "A", "snippet": "aaa"},
        {"url": "https://b.com", "title": "B", "snippet": "bbb"},
    ]
    brief = build_brief(evidence)
    assert "[1]" in brief
    assert "[2]" in brief
    assert "https://a.com" in brief


def test_build_brief_empty():
    brief = build_brief([])
    assert "No sources found" in brief


def test_deep_research_graph_runs(monkeypatch):
    """The deep research graph runs planner, fan-out, search, and brief."""

    class FakeModel:
        def invoke(self, prompt):
            return AIMessage(content="q1\nq2")

    class FakeClient:
        def search(self, query, max_results=5, search_depth="basic"):
            return {
                "results": [
                    {
                        "title": f"Result for {query}",
                        "url": f"https://example.com/{query}",
                        "content": f"snippet {query}",
                    }
                ]
            }

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )
    monkeypatch.setattr(
        "langgraph_deep_research.tools._client", lambda: FakeClient()
    )

    g = build_deep_research_graph()
    result = g.invoke({"messages": [("user", "topic")]})
    assert "brief" in result
    assert len(result["evidence"]) == 2
    assert "q1" in result["brief"] or "Result for q1" in result["brief"]
