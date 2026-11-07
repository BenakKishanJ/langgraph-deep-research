"""LangChain tools that wrap external services for the research agent."""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from tavily import TavilyClient

from .config import load_settings


def _client() -> TavilyClient:
    """Build a Tavily client from current settings."""
    return TavilyClient(api_key=load_settings().tavily_api_key)


@tool
def web_search(query: Annotated[str, "The search query to run."]) -> str:
    """Search the web for recent information about a topic.

    Returns formatted results with titles, URLs, and content snippets.
    """
    client = _client()
    response = client.search(query, max_results=5, search_depth="basic")
    results = response.get("results", [])
    if not results:
        return "No results found."
    blocks = []
    for item in results:
        blocks.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
        )
    return "\n\n".join(blocks)


def web_search_results(query: str) -> list[dict[str, str]]:
    """Return structured search results: list of {url, title, snippet}."""
    client = _client()
    response = client.search(query, max_results=5, search_depth="basic")
    results = response.get("results", [])
    return [
        {
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
        }
        for item in results
    ]
