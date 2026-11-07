"""Adapter the UI calls; invokes the deep research graph."""
from __future__ import annotations

import json

from .graph import build_deep_research_graph


def ask(question: str) -> str:
    """Run the deep research graph and return the cited markdown brief."""
    result = build_deep_research_graph().invoke({"messages": [("user", question)]})
    return result["brief"]


def ask_stream(question: str):
    """Yield progress events as Server-Sent Events while running the graph."""
    for chunk in build_deep_research_graph().stream(
        {"messages": [("user", question)]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            yield f"event: {node}\ndata: {json.dumps(update)}\n\n"
