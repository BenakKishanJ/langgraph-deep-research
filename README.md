# langgraph-deep-research

A beginner course that builds a deep research agent with LangGraph, Tavily
search, and OpenRouter free models.

## Project setup: run with uv

sh
uv sync                      # create .venv and install dependencies
uv add <package>             # named by a part; resolved versions only
uv run pytest                # run the tests (no network: fake LLM)


## Run the web UI

sh
uv run deep-research-ui


Open <http://localhost:5102> and ask a question. The UI posts to `/api/ask`,
which runs the LangGraph agent and returns the answer.

## Follow along

sh
git clone https://github.com/BenakKishanJ/langgraph-deep-research.git
git checkout part-N                                  # any published part
git diff part-(N-1) part-N                           # what part N adds

