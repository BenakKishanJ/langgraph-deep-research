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
    """State for the deep research graph: planner, fan-out, search, brief."""

    messages: Annotated[list, operator.add]
    sub_queries: list[str]
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
    """
    return {}


def fan_out_router(state: DeepResearchState) -> list:
    """Return one ``Send`` per sub-query: a search run starts for each one."""
    return [Send("search", {"query": q}) for q in state["sub_queries"]]


def build_deep_research_graph():
    """Compile the planner → fan-out → search → brief graph."""
    from .planner import planner_node

    g = StateGraph(DeepResearchState)
    g.add_node("planner", planner_node)
    g.add_node("fan_out", fan_out_node)
    g.add_node("search", search_node)
    g.add_node("brief", brief_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "fan_out")
    # no path map: the router returns Send objects, so LangGraph starts one
    # search invocation per Send and then joins the parallel branches.
    g.add_conditional_edges("fan_out", fan_out_router)
    # every search invocation ends at the brief, so the brief runs exactly
    # once, after all parallel searches have finished.
    g.add_edge("search", "brief")
    return g.compile()


graph = build_graph()
