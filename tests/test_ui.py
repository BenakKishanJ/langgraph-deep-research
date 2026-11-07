"""Tests for the FastAPI UI — offline TestClient."""
from __future__ import annotations

from fastapi.testclient import TestClient

from langgraph_deep_research.ui.server import app


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(
        "langgraph_deep_research.ui.server.ask", lambda q: f"answer for {q}"
    )
    client = TestClient(app)
    resp = client.post("/api/ask", json={"question": "hello"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "answer for hello"}


def test_root_serves_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Deep Research" in resp.text
