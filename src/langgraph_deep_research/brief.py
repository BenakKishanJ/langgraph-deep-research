"""Build a structured markdown report with a numbered bibliography."""
from __future__ import annotations


def build_brief(evidence: list[dict[str, str]]) -> str:
    """Return a structured markdown report with Findings and Bibliography.

    Each finding cites its source as [N] and the numbered Bibliography
    lists every source with its title and URL. Empty evidence yields a
    short "no sources" report.
    """
    if not evidence:
        return "# Research Report\n\nNo sources found."
    lines = ["# Research Report", "", "## Findings", ""]
    for i, item in enumerate(evidence, start=1):
        snippet = item.get("snippet", "")
        lines.append(f"- {snippet} [{i}]")
        lines.append("")
    lines.append("## Bibliography")
    lines.append("")
    for i, item in enumerate(evidence, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        lines.append(f"[{i}] {title} - {url}")
    return "\n".join(lines)
