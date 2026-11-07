"""Adapter the UI calls; invokes the graph."""
from __future__ import annotations

from .graph import graph


def ask(question: str) -> str:
    """Run the graph on a single user message and return the assistant reply."""
    result = graph.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content
