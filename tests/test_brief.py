"""Offline tests for brief.build_brief - no network, no LLM."""
from __future__ import annotations

from langgraph_deep_research.brief import build_brief


def test_build_brief_has_findings_and_bibliography():
    evidence = [
        {"url": "https://example.com/one", "title": "Source One", "snippet": "First snippet."},
        {"url": "https://example.com/two", "title": "Source Two", "snippet": "Second snippet."},
    ]
    report = build_brief(evidence)
    assert "# Research Report" in report
    assert "## Findings" in report
    assert "## Bibliography" in report
    assert "[1]" in report
    assert "[2]" in report
    assert "Source One" in report
    assert "https://example.com/one" in report


def test_build_brief_empty():
    assert build_brief([]) == "# Research Report\n\nNo sources found."
