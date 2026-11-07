"""Agent graph with an ask/tool loop backed by Tavily search."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from .config import load_settings
from .tools import web_search


class State(TypedDict):
    # LangGraph channels without a reducer are last-write-wins: a plain
    # list would be replaced on every node return, so the ask/tool loop
    # would lose the user question and tool results. operator.add makes the
    # messages channel accumulate across nodes.
    messages: Annotated[list, operator.add]


class DeepResearchState(TypedDict):
    """State for the deep research graph: planner, fan-out, search, reflection, brief."""

    messages: Annotated[list, operator.add]
    sub_queries: list[str]
    follow_up_queries: list[str]
    round: int
    evidence: Annotated[list[dict[str, str]], operator.add]
    brief: str


def _default_model() -> ChatOpenAI:
    s = load_settings()
    return ChatOpenAI(
        model=s.model,
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
    )


def agent_node(state: State) -> State:
    """Call the model with tools bound; append the assistant reply."""
    model = _default_model().bind_tools([web_search])
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    """Route to the tool node when the model asks for a tool call."""
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "end"


def build_graph():
    tool_node = ToolNode([web_search])
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    g.add_edge("tools", "agent")
    return g.compile()


# --- Deep research graph nodes ---


def search_node(state: dict) -> dict:
    """Run a single web search and return evidence entries."""
    from .tools import web_search_results

    results = web_search_results(state["query"])
    return {"evidence": results}


def brief_node(state: DeepResearchState) -> dict:
    """Produce a markdown brief with numbered citations from evidence."""
    from .brief import build_brief

    brief = build_brief(state["evidence"])
    return {"brief": brief, "messages": [AIMessage(content=brief)]}


def fan_out_node(state: DeepResearchState) -> dict:
    """Join node between planning and searching.

    Returns a dict (nodes return dicts); the actual fan-out happens in the
    conditional edge registered on this node, whose router returns one
    ``Send`` per sub-query. Send may NOT be returned by a node itself.

    Returns an empty dict: this body runs BEFORE the router, so clearing
    ``follow_up_queries`` here would wipe the queries the reflection node
    just produced and make the loop re-search the original sub-queries.
    Keeping follow-ups intact lets the router pick them per round.
    """
    return {}


def fan_out_router(state: DeepResearchState) -> list:
    """Return one ``Send`` per query: use follow-ups if present, else sub-queries."""
    queries = state.get("follow_up_queries") or state.get("sub_queries", [])
    return [Send("search", {"query": q}) for q in queries]


def build_deep_research_graph():
    """Compile planner → fan-out → search → reflection → (loop or brief)."""
    from .planner import planner_node
    from .reflection import reflection_node, should_continue_reflection

    g = StateGraph(DeepResearchState)
    g.add_node("planner", planner_node)
    g.add_node("fan_out", fan_out_node)
    g.add_node("search", search_node)
    g.add_node("reflection", reflection_node)
    g.add_node("brief", brief_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "fan_out")
    # no path map: the router returns Send objects, so LangGraph starts one
    # search invocation per Send and then joins the parallel branches.
    g.add_conditional_edges("fan_out", fan_out_router)
    # search results flow into reflection, which may loop back to fan_out
    # or end at the brief.
    g.add_edge("search", "reflection")
    g.add_conditional_edges(
        "reflection",
        should_continue_reflection,
        {"more": "fan_out", "end": "brief"},
    )
    return g.compile()


graph = build_graph()
