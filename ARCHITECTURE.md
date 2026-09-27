# Architecture

```
                       ┌──────────────────────────────────────────┐
                       │              TRAINING                     │
                       │  data/ ─> transformer.train ─> *.pt ckpt │
                       └───────────────┬──────────────────────────┘
                                       │ export (parity-checked)
                                       ▼
 ┌───────────── web/ (GitHub Pages) ──────────────┐   ┌── server/ (FastAPI) ──┐
 │  index.html ─ app.js ─ onnxruntime-web (WASM) │   │  /chat   /tools      │
 │  model/model.onnx + vocab.json                │   │  loads same ckpt     │
 │  history: localStorage ONLY                    │   └──────────┬───────────┘
 └───────────────┬────────────────────────────────┘              │
                 │ same origin, PWA                             ▼
                 ▼                                    agent/loop.py
        android/ WebView shell ──> APK          observe -> route -> tool
        (CI builds via Actions)                  -> observe -> respond
                                                      │
                                              agent/tools.py
                                   calculator | python sandbox | clock | search*
```

## Model

`transformer/model.py` — decoder-only GPT.

- Token embeddings + learned positional embeddings (context 256)
- 5 pre-LayerNorm blocks: causal multi-head attention (4 heads, GELU MLP
  4× width), dropout only matters at scale
- Output head weight-tied to the input embedding
- Sampling: temperature + top-k, stop-token aware

Parameters: **1,032,704** (999,936 in the transformer body).

`transformer/tokenizer.py` — character-level (65 symbols). Deliberate at 1M:
a BPE vocab of 8–50k would eat the entire parameter budget in embeddings.
Swap point is documented in the file; the rest of the codebase only calls
`encode()/decode()`.

## Training

`transformer/train.py` — AdamW + OneCycleLR, grad clipping, 90/10 train/val
split, best-checkpoint tracking, JSONL logs, and two properties that matter
when you train on weak hardware: it is **resumable** (`transformer_1m.pt`
carries optimizer + scheduler state) and **time-budgeted** (`--max-seconds`),
so N runs of M minutes equal one longer run. Prototype reached
val loss 2.17 in 835 steps across several CPU-only sessions.

## Agent loop

`agent/loop.py` — `respond()`:

1. **observe**: user message + history
2. **route**: which tools (if any) — currently deterministic rules in
   `agent/tools.route()`; at 500M the model emits tool calls itself
3. **act**: run tools, fold results back into context
4. **respond**: tool answers win when they fully answer; otherwise the LLM
   generates (context includes tool outputs as `[tool x says] ...` notes)

Tools are `(name, description, JSON-ish args, fn)` records in a registry —
the shape mirrors OpenAI-style function calling, so the upgrade path is:
train the model on tool-use traces, then swap `route()` for the model's own
emitted calls. `SearchProvider` is the seam for real-time research: implement
`search(query)` (DuckDuckGo/Brave/Tavily) and inject it into the Agent.

The Python tool runs in a subprocess with a 10s timeout and no network egress
by default — it is a demo sandbox, not a security boundary. Before exposing
the server publicly, put it behind auth and a real sandbox (firejail/nsjail,
or a container).

## Serving

- **`server/app.py`** — FastAPI: `POST /chat` (message + history → reply +
  tools used), `GET /tools`, `GET /health`. CORS open for local dev.
- **`web/`** — static PWA; the ONNX model runs client-side, so the site is
  stateless: no user data can exist server-side even in principle. Service
  worker caches app + model for offline use.
- **Deployment** — GitHub Actions deploys `web/` to GitHub Pages on every
  push to `main`. The first deploy trains and exports the model in CI
  (cached by `actions/cache` afterwards). The APK workflow builds the
  Android shell on demand.

## Scaling to 500M (the plan, not the prototype)

1. **Tokenizer**: byte-level BPE (32k vocab). `TARGET_500M` already sized for it.
2. **Data**: 50–100 GB curated text (web, code, dialogue). TinyShakespeare
   is to the prototype what a sketch is to a building.
3. **Training**: multi-GPU (DDP/FSDP), bf16, flash-attention; `train.py`
   already has the resume/checkpoint/log bones.
4. **Instruction tuning**: SFT on chat + tool-use traces → the deterministic
   router retires.
5. **Alignment**: DPO on preference pairs.
6. **Efficiency**: int8/int4 quantized ONNX export; the 500M sibling still
   runs on-device on flagship phones, same `web/` shell.
7. **Apps**: unchanged — that's the point of the architecture.

## Repository layout

```
transformer-llm/
├── transformer/        # model, tokenizer, train, chat, ONNX export
├── agent/              # loop + tools
├── server/             # FastAPI
├── web/                # PWA (model + vocab + service worker)
├── android/            # WebView shell -> APK
├── data/               # fetched by quickstart / CI (not stored)
└── .github/workflows/  # pages deploy + APK build
```
