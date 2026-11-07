"""Tests for the single-node graph — uses a fake model, no network."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from langgraph_deep_research.graph import build_graph


def test_answer_node_with_fake_model(monkeypatch):
    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content="fake answer")

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )
    g = build_graph()
    result = g.invoke({"messages": [("user", "hi")]})
    assert result["messages"][-1].content == "fake answer"
