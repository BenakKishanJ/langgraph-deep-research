"""Tests for the deep research graph (planner, fan-out, reflection, brief) — offline."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from langgraph_deep_research.brief import build_brief
from langgraph_deep_research.graph import build_deep_research_graph
from langgraph_deep_research.planner import planner_node
from langgraph_deep_research.reflection import reflection_node, should_continue_reflection


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


def test_reflection_node_suggests_followups(monkeypatch):
    """reflection_node returns follow_up_queries from the model output."""

    class FakeModel:
        def invoke(self, prompt):
            return AIMessage(content="Follow-up 1\nFollow-up 2")

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )
    state = {
        "messages": [],
        "evidence": [{"url": "https://a", "title": "A", "snippet": "aaa"}],
        "round": 0,
        "follow_up_queries": [],
        "sub_queries": [],
        "brief": "",
    }
    result = reflection_node(state)
    assert result["follow_up_queries"] == ["Follow-up 1", "Follow-up 2"]
    assert result["round"] == 1
    assert "Reflection round 1" in result["messages"][0].content


def test_reflection_node_no_followups(monkeypatch):
    """reflection_node returns empty list when the model gives no suggestions."""

    class FakeModel:
        def invoke(self, prompt):
            return AIMessage(content="")

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )
    state = {
        "messages": [],
        "evidence": [{"url": "https://a", "title": "A", "snippet": "aaa"}],
        "round": 0,
        "follow_up_queries": ["old"],
        "sub_queries": [],
        "brief": "",
    }
    result = reflection_node(state)
    assert result["follow_up_queries"] == []
    assert result["round"] == 1


def test_should_continue_reflection():
    assert should_continue_reflection({"follow_up_queries": ["q"], "round": 0}) == "more"
    assert should_continue_reflection({"follow_up_queries": ["q"], "round": 2}) == "end"
    assert should_continue_reflection({"follow_up_queries": [], "round": 0}) == "end"


def test_deep_research_graph_runs(monkeypatch):
    """The deep research graph runs planner, fan-out, search, reflection, and brief."""

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
    # Start at round 2 so the reflection step stops immediately.
    result = g.invoke({"messages": [("user", "topic")], "round": 2})
    assert "brief" in result
    assert len(result["evidence"]) == 2
    assert "q1" in result["brief"] or "Result for q1" in result["brief"]


def test_deep_research_graph_reflection_loop(monkeypatch):
    """The graph loops back for follow-up searches within the rounds budget."""

    class FakeModel:
        def __init__(self):
            self.count = 0

        def invoke(self, prompt):
            self.count += 1
            if self.count == 1:
                return AIMessage(content="q1\nq2")
            elif self.count == 2:
                return AIMessage(content="followup")
            else:
                return AIMessage(content="")

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
    # Two initial queries + one follow-up query = three searches.
    assert len(result["evidence"]) == 3
    assert "brief" in result
