# Architecture

The project is a deep research agent built with LangGraph. It uses a single
StateGraph node that calls an OpenRouter chat model. The UI is a FastAPI app
that exposes `/api/ask` and serves a minimal HTML page.

## State

`State` is a `TypedDict` with a single `messages` channel (list of
`BaseMessage`). The reducer is the default last-write-wins.

## Graph

- `answer_node` — calls the model with the user messages and appends the
  assistant reply.

The graph is compiled in `graph.py` and exposed as `graph`.

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
