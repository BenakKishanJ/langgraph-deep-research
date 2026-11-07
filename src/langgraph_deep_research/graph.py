"""Agent graph with an ask/tool loop backed by Tavily search."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

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


graph = build_graph()
