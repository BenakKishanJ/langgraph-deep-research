"""Tests for config.load_settings — offline, no network."""
from __future__ import annotations

from langgraph_deep_research.config import load_settings


def test_defaults_when_no_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    s = load_settings()
    assert s.openrouter_api_key == ""
    assert s.tavily_api_key == ""
    assert s.model == "nvidia/nemotron-3.5-lightning:free"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    s = load_settings()
    assert s.openrouter_api_key == "env-key"
