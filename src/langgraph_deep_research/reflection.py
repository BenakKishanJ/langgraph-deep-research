"""Reflection node that critiques evidence and suggests follow-up queries."""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from . import graph as graph_module
from .graph import DeepResearchState

_REFLECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research referee. Review the following evidence and decide whether more research is needed. If yes, suggest up to 3 follow-up queries, each on a new line. If no more research is needed, respond with an empty line."),
    ("human", "{evidence}"),
])


def reflection_node(state: DeepResearchState) -> dict:
    """Produce follow-up queries from the current evidence."""
    evidence = state.get("evidence", [])
    evidence_text = "\n".join(
        f"- {item.get('title', '')}: {item.get('snippet', '')}" for item in evidence
    )
    model = graph_module._default_model()
    prompt = _REFLECT_PROMPT.format(evidence=evidence_text)
    response = model.invoke(prompt)
    lines = [line.strip() for line in response.content.splitlines() if line.strip()]
    round_number = state.get("round", 0) + 1
    return {
        "follow_up_queries": lines,
        "round": round_number,
        "messages": [AIMessage(content=f"Reflection round {round_number}: {lines}")],
    }


def should_continue_reflection(state: DeepResearchState) -> str:
    """Route to fan_out if there are follow-up queries and budget remains."""
    max_rounds = 2
    if state.get("follow_up_queries") and state.get("round", 0) < max_rounds:
        return "more"
    return "end"
