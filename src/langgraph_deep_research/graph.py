"""Single-node StateGraph that answers a user question."""
from __future__ import annotations

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from .config import load_settings


class State(TypedDict):
    messages: list


def _default_model() -> ChatOpenAI:
    s = load_settings()
    return ChatOpenAI(
        model=s.model,
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
    )


def answer_node(state: State) -> State:
    model = _default_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    g = StateGraph(State)
    g.add_node("answer", answer_node)
    g.set_entry_point("answer")
    g.set_finish_point("answer")
    return g.compile()


graph = build_graph()
