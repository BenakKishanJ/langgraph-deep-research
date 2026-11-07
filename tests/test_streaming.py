"""Tests for the streaming progress endpoint — offline."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from langgraph_deep_research.graph import build_deep_research_graph
from langgraph_deep_research.ui.server import app


def test_ask_stream_endpoint_sends_events(monkeypatch):
    """The streaming endpoint yields events for each graph node."""

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

    from fastapi.testclient import TestClient

    client = TestClient(app)
    with client.stream("POST", "/api/ask-stream", json={"question": "hi"}) as response:
        events = []
        for line in response.iter_lines():
            if line:
                events.append(line)
        # Check that we received events for planner, search, reflection, brief.
        assert any("planner" in e for e in events)
        assert any("search" in e for e in events)
        assert any("reflection" in e for e in events)
        assert any("brief" in e for e in events)
