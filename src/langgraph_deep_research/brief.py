"""Build a markdown brief with numbered citations from evidence."""
from __future__ import annotations


def build_brief(evidence: list[dict[str, str]]) -> str:
    """Return a markdown brief with numbered citations."""
    if not evidence:
        return "# Research Brief\n\nNo sources found."
    lines = []
    for i, item in enumerate(evidence, start=1):
        lines.append(
            f"[{i}] {item.get('title', 'Untitled')} ({item.get('url', '')})"
        )
    return "\n".join(lines)
