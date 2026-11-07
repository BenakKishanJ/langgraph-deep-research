"""Adapter the UI calls; invokes the deep research graph."""
from __future__ import annotations

from .graph import build_deep_research_graph


def ask(question: str) -> str:
    """Run the deep research graph and return the cited markdown brief."""
    result = build_deep_research_graph().invoke({"messages": [("user", question)]})
    return result["brief"]
