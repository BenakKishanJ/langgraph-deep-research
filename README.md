# langgraph-deep-research

A beginner course that builds a deep research agent with LangGraph, Tavily
search, and OpenRouter free models.

## Project setup: run with uv

```sh
uv sync                      # create .venv and install dependencies
uv add <package>             # named by a part; resolved versions only
uv run pytest                # run the tests (no network: fake LLM)
```

The UI is prebuilt — you will not touch it. You build the agent graph it
talks to: a planner that splits your topic into sub-queries, parallel
Tavily searches, a bounded reflection loop, and a cited markdown report.

## Run the web UI

```sh
uv run deep-research-ui
```

Open <http://localhost:5102> and ask a question. The UI posts to `/api/ask`,
which runs the deep research graph and returns a cited markdown report.

## Run the tests

```sh
uv run pytest
```

All tests are offline: they use a fake chat model and a fake Tavily client,
so no network calls are made.

## Manual test

Set `OPENROUTER_API_KEY` and `TAVILY_API_KEY` in the environment (or in a
`.env` file), then run:

```sh
uv run deep-research-ui
```

Open <http://localhost:5102>, ask a question, and the UI returns a cited
markdown report. Use `/api/ask-stream` for live progress events.

## Follow along

```sh
git clone https://github.com/BenakKishanJ/langgraph-deep-research.git
git checkout part-N                                  # any published part
git diff part-(N-1) part-N                           # what part N adds
```
