"""Offline end-to-end tests for the deep research graph.

Uses a fake chat model and a fake Tavily client so no network calls are
made.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_deep_research import graph as graph_module
from langgraph_deep_research import tools
from langgraph_deep_research.graph import build_deep_research_graph


class FakeModel:
    """A chat model that returns canned responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt, **kwargs):
        self.calls.append(str(prompt))
        content = self.responses.pop(0) if self.responses else ""
        return AIMessage(content=content)


class FakeTavilyClient:
    """A Tavily client that returns canned results keyed by query."""

    def __init__(self, results_by_query: dict[str, list[dict[str, str]]]):
        self.results_by_query = results_by_query

    def search(self, query, max_results=5, search_depth="basic"):
        return {"results": self.results_by_query.get(query, [])}


def _install_fakes(monkeypatch, model_responses, results_by_query):
    fake_model = FakeModel(model_responses)
    monkeypatch.setattr(graph_module, "_default_model", lambda: fake_model)
    fake_client = FakeTavilyClient(results_by_query)
    monkeypatch.setattr(tools, "_client", lambda: fake_client)


def _initial_state(question: str) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "sub_queries": [],
        "follow_up_queries": [],
        "round": 0,
        "evidence": [],
        "brief": "",
    }


def test_deep_research_single_round(monkeypatch):
    results_by_query = {
        "capital of France": [
            {
                "url": "https://example.com/france",
                "title": "France",
                "snippet": "Paris is the capital of France.",
            }
        ],
        "population of France": [
            {
                "url": "https://example.com/pop",
                "title": "Population",
                "snippet": "France has about 67 million people.",
            }
        ],
    }
    _install_fakes(
        monkeypatch,
        [
            "capital of France\npopulation of France\n",
            "",
        ],
        results_by_query,
    )
    graph = build_deep_research_graph()
    result = graph.invoke(_initial_state("Research France"))
    assert result["brief"]
    assert "# Research Report" in result["brief"]
    assert "## Bibliography" in result["brief"]
    assert "France" in result["brief"]
    assert "https://example.com/france" in result["brief"]


def test_deep_research_reflection_loop(monkeypatch):
    results_by_query = {
        "capital of France": [
            {
                "url": "https://example.com/france",
                "title": "France",
                "snippet": "Paris is the capital of France.",
            }
        ],
        "french history": [
            {
                "url": "https://example.com/history",
                "title": "History",
                "snippet": "France has a long history.",
            }
        ],
    }
    _install_fakes(
        monkeypatch,
        [
            "capital of France\n",
            "french history\n",
            "",
        ],
        results_by_query,
    )
    graph = build_deep_research_graph()
    result = graph.invoke(_initial_state("Research France"))
    assert result["round"] == 2
    assert "France" in result["brief"]
    assert "https://example.com/history" in result["brief"]
