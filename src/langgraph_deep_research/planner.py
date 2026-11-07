"""Planner node that breaks a topic into sub-queries."""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from . import graph as graph_module
from .graph import DeepResearchState

_PLANNER_SYSTEM = (
    "You are a research planner. Break the user's topic into 3-5 sub-queries "
    "that cover the topic comprehensively. Return each sub-query on a new "
    "line, with no numbering or extra text."
)
_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _PLANNER_SYSTEM),
    ("human", "{topic}"),
])


def planner_node(state: DeepResearchState) -> dict:
    """Generate sub-queries from the user's message."""
    user_message = state["messages"][0]
    topic = user_message.content if hasattr(user_message, "content") else str(user_message)
    model = graph_module._default_model()
    prompt = _PLANNER_PROMPT.format(topic=topic)
    response = model.invoke(prompt)
    lines = [line.strip() for line in response.content.splitlines() if line.strip()]
    return {"sub_queries": lines, "messages": [AIMessage(content=f"Sub-queries: {lines}")]}
