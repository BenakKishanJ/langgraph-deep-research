"""Tests for the agent graph with a tool loop — uses a fake model, no network."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from langgraph_deep_research.graph import build_graph


def test_agent_loop_runs_tool_then_answers(monkeypatch):
    """The graph runs the ask/tool loop until the model gives a final answer."""
    calls = {"model": 0}

    class FakeModel:
        def invoke(self, messages):
            calls["model"] += 1
            if calls["model"] == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": "test query"},
                            "id": "call_1",
                        }
                    ],
                )
            return AIMessage(content="final answer")

        def bind_tools(self, tools):
            return self

    class FakeClient:
        def search(self, query, max_results=5, search_depth="basic"):
            return {
                "results": [
                    {
                        "title": "Fake Result",
                        "url": "https://example.com",
                        "content": "fake content",
                    }
                ]
            }

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )
    monkeypatch.setattr(
        "langgraph_deep_research.tools._client", lambda: FakeClient()
    )

    g = build_graph()
    result = g.invoke({"messages": [("user", "ask something")]})

    messages = result["messages"]
    assert messages[-1].content == "final answer"
    assert calls["model"] == 2
    assert len(messages) == 4
    assert messages[1].tool_calls
    assert "fake content" in messages[2].content


def test_agent_answers_without_tool_when_no_tool_call(monkeypatch):
    """When the model gives a direct answer, the graph ends immediately."""

    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content="direct answer")

        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )

    g = build_graph()
    result = g.invoke({"messages": [("user", "hi")]})
    assert result["messages"][-1].content == "direct answer"


def test_ui_api_ask_runs_graph(monkeypatch):
    """ui_api.ask invokes the graph and returns the last message content."""

    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content="answer from graph")

        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(
        "langgraph_deep_research.graph._default_model", lambda: FakeModel()
    )

    from langgraph_deep_research.ui_api import ask

    assert ask("hello") == "answer from graph"
