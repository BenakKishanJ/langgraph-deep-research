# Architecture

The project is a deep research agent built with LangGraph. It uses an
agent node that calls an OpenRouter chat model with a Tavily search tool
bound to it, a ToolNode that runs the tool, and a conditional edge that
loops back to the agent while the model keeps issuing tool calls. The UI
is a FastAPI app that exposes `/api/ask` and serves a minimal HTML page.

## State

`State` is a `TypedDict` with a single `messages` channel (list of
`BaseMessage`). The channel uses the `operator.add` accumulator reducer:
without it, LangGraph channels are last-write-wins and each node return
would wipe the previous messages, breaking the ask/tool loop. Messages
accumulate across the loop: the agent appends its reply (which may carry
tool calls), the tool node appends the tool results, and the agent appends
the final answer.

## Graph

- `agent_node` — calls the model with the user messages and the
  `web_search` tool bound to it; appends the assistant reply (which may
  contain tool calls).
- `tools` — a `ToolNode` that runs `web_search` for every tool call the
  model issued and appends the results as `ToolMessage`s.
- `should_continue` — a conditional edge from `agent` that routes back to
  `tools` while the last message has tool calls, or ends the run when the
  model gives a final answer.

The graph is compiled in `graph.py` and exposed as `graph`.

## Tools

`tools.py` wraps Tavily search as a LangChain `@tool` called
`web_search`. It builds a `TavilyClient` from `TAVILY_API_KEY` (via
`config.load_settings()`), runs a basic search, and returns formatted
title/URL/content blocks. A structured helper `web_search_results` returns
a list of `{url, title, snippet}` dicts for the deep research graph.

## Configuration

`config.load_settings()` reads `OPENROUTER_API_KEY`, `TAVILY_API_KEY`,
`OPENROUTER_BASE_URL`, and `DEFAULT_MODEL` from environment variables (with
sane defaults). A `.env` file in the project root is loaded via `python-dotenv`;
environment variables take precedence.

## UI

`ui/server.py` runs a FastAPI app on `0.0.0.0:5102`. The console script
`deep-research-ui` starts it.

## Testing

Tests use `pytest` with a fake model and `TestClient`; no network calls.

## Deep Research Graph

The deep research graph extends the agent with a planner node that breaks
a topic into sub-queries, a fan-out that runs a search node for each
sub-query, and a brief node that produces a cited markdown report.

### State

`DeepResearchState` adds three channels to `State`:

- `sub_queries` — a list of strings produced by the planner.
- `evidence` — an `Annotated[list[dict[str, str]], operator.add]` channel
  that accumulates `{url, title, snippet}` entries from every search node.
  The `operator.add` reducer appends each search's results so the brief
  node sees all sources.
- `brief` — the final markdown string.

### Graph

- `planner_node` — calls the model with a planning prompt; returns the
  sub-queries and an assistant message.
- `fan_out_node` — returns a `Send` for each sub-query, invoking
  `search_node` with `{"query": q}`.
- `search_node` — runs `web_search_results` and returns evidence entries.
- `brief_node` — calls `build_brief` from `brief.py` and writes the brief
  and a final assistant message.

The graph is compiled by `build_deep_research_graph()` (exposed as
`deep_research_graph` when used).
