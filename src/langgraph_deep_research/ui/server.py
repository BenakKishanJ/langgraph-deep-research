"""FastAPI web UI for the deep research agent."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .ui_api import ask

app = FastAPI()

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Deep Research UI</title>
</head>
<body>
    <h1>Deep Research</h1>
    <form id="ask-form">
        <input id="question" type="text" placeholder="Ask a question…" />
        <button type="submit">Ask</button>
    </form>
    <div id="answer"></div>
    <script>
        document.getElementById("ask-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const question = document.getElementById("question").value;
            const resp = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const data = await resp.json();
            document.getElementById("answer").textContent = data.answer;
        });
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return HTML_PAGE


@app.post("/api/ask")
async def ask_endpoint(request: Request) -> dict:
    body = await request.json()
    question = body.get("question", "")
    answer = ask(question)
    return {"answer": answer}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5102)


if __name__ == "__main__":
    main()
