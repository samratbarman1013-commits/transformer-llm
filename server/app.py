"""Transformer API server — private-model inference over HTTP.

The model weights never leave this server: clients (web app / Android APK)
authenticate with an API key and call POST /api/chat. Chat history stays on
the client device; only the current prompt+context is sent for inference.

Setup:
    Put the private model files in server-data/ (or set TRANSFORMER_MODEL_DIR):
        server-data/model_best.pt     (from the CI chain)
        server-data/tokenizer.json    (the 32k BPE tokenizer)
    or keep the 1M prototype at checkpoints/transformer_1m_best.pt + vocab.json.

Run:
    TRANSFORMER_API_KEY=<secret> uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

API_KEY = os.environ.get("TRANSFORMER_API_KEY", "").strip()
MODEL_DIR = Path(os.environ.get("TRANSFORMER_MODEL_DIR", ROOT / "server-data"))
MAX_NEW_CAP = 600

app = FastAPI(title="Transformer API", version="0.2.0",
              description="Private-model chat API. The model stays on the server.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                    allow_headers=["*"])

# ---------- model loading (BPE ladder preferred, char prototype as fallback) ----
model = tok = eos_id = None
model_info = {"loaded": False, "kind": None, "config": None, "params": 0}


def _load_models() -> None:
    global model, tok, eos_id, model_info
    ck, tj = MODEL_DIR / "model_best.pt", MODEL_DIR / "tokenizer.json"
    if ck.exists() and tj.exists():
        from transformer.infer_v2 import load as load_bpe
        model, tok, eos_id, cfg, _vs = load_bpe(str(ck), str(tj))
        model_info = {"loaded": True, "kind": "bpe", "config": cfg,
                      "params": int(model.num_params())}
        print(f"[server] loaded {cfg} ({model_info['params']:,} params) from {MODEL_DIR}")
        return
    ck1 = ROOT / "checkpoints" / "transformer_1m_best.pt"
    if ck1.exists():
        from transformer.chat import load as load_char
        model, tok = load_char(str(ck1))
        eos_id = None
        model_info = {"loaded": True, "kind": "char", "config": "proto-1m",
                      "params": int(model.num_params())}
        print(f"[server] loaded 1M char prototype from {ck1}")
        return
    print(f"[server] WARNING: no model found (looked in {MODEL_DIR} and "
          f"{ck1}) — /api/chat will return 503")


_load_models()


# ---------- auth ----------------------------------------------------------------
def require_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
                authorization: Optional[str] = Header(None)) -> None:
    """No-op when TRANSFORMER_API_KEY is unset (local dev only)."""
    if not API_KEY:
        return
    supplied = x_api_key
    if not supplied and authorization and \
            authorization.lower().startswith("bearer "):
        supplied = authorization[7:]
    if supplied != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# ---------- API -----------------------------------------------------------------
class ChatIn(BaseModel):
    message: str
    history: list[dict] = []          # [{role: "user"|"ai", text: "..."}]
    max_new: Optional[int] = None     # reply length cap (server caps at 600)
    temperature: Optional[float] = None
    top_k: Optional[int] = None


def _build_history(items: list[dict]) -> str:
    parts = []
    for m in items:
        role = "User:" if m.get("role") == "user" else "Transformer:"
        parts.append(f"{role} {str(m.get('text', ''))}")
    return "\n".join(parts)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": model_info, "auth": bool(API_KEY)}


@app.get("/api/verify")
def verify(_: None = Depends(require_key)):
    """Used by the app's setup screen to test URL + key."""
    return {"ok": True, "model": model_info}


@app.post("/api/chat")
def chat(body: ChatIn, _: None = Depends(require_key)):
    if not model_info["loaded"]:
        raise HTTPException(status_code=503, detail="model not loaded on server")
    try:
        hist = _build_history(body.history)
        if body.message:
            hist = (hist + "\n" if hist else "") + f"User: {body.message}"
        max_new = min(int(body.max_new or 220), MAX_NEW_CAP)
        temperature = min(max(float(body.temperature or 0.8), 0.05), 2.0)
        top_k = min(int(body.top_k or 40), 200)
        if model_info["kind"] == "bpe":
            from transformer.infer_v2 import reply as reply_bpe
            out = reply_bpe(model, tok, eos_id, hist, max_new=max_new,
                            temperature=temperature, top_k=top_k)
        else:
            from transformer.chat import reply as reply_char
            out = reply_char(model, tok, hist, max_new=max_new,
                             temperature=temperature, top_k=top_k)
    except HTTPException:
        raise
    except Exception as e:  # keep client-facing errors clean
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")
    return {"reply": out}
