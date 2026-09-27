"""Transformer server — FastAPI chat + tools API.

Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import Agent                          # noqa: E402
from agent.tools import DEFAULT_TOOLS, SearchProvider  # noqa: E402
from transformer.chat import load as load_model       # noqa: E402

app = FastAPI(title="Transformer API", version="0.1.0",
              description="1M-parameter prototype LLM with an agent loop and tools")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                    allow_headers=["*"])

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "transformer_1m_best.pt"
model = tok = None
if CKPT.exists():
    model, tok = load_model(str(CKPT))

agent = Agent(model=model, tok=tok, search_provider=SearchProvider())


class ChatIn(BaseModel):
    message: str
    history: str = ""


class ChatOut(BaseModel):
    reply: str
    tools_used: list[dict]


@app.get("/")
def root():
    return {"service": "transformer", "model": "proto-1m",
            "params": 1_032_704 if model else 0,
            "model_loaded": model is not None}


@app.get("/tools")
def tools():
    return [{"name": t.name, "description": t.description, "args": t.args}
            for t in DEFAULT_TOOLS.values()]


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    turn = agent.respond(body.message, body.history)
    return ChatOut(reply=turn.final, tools_used=turn.tool_calls)


@app.get("/health")
def health():
    return {"status": "ok"}
