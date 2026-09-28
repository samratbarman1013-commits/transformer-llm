# Transformer LLM

**Real transformer LLMs, built from scratch and trained for ₹0 on free GitHub Actions runners.**

Live site: **https://samratbarman1013-commits.github.io/transformer-llm/**

Two models are learning right now, in 5-hour self-chaining CI chunks with an
hourly supervisor that dispatches the next chunk automatically:

- **`ci-150m`** (164.9M params, TinyStories) — 2,000 steps done (best val loss
  2.34, released as `v0.2.0`), now extending to 6,000 steps → `v0.3.0`
- **`ci-300m`** (~325M params, **85% Python code + 15% stories**) — training
  in progress, target 6,000 steps → `v0.4.0`

The website demo runs the 1M prototype **entirely in your browser** via
onnxruntime-web (WASM): no server call, no telemetry, and chat history lives
in your browser's localStorage only. The bigger models are served by the
FastAPI server behind an API key — model weights are never published
(releases are notes-only; weights stay in private CI artifacts).

## What's inside

| Piece | What it does |
|---|---|
| `transformer/` | Model, char + BPE tokenizers, training, chat CLI, ONNX export |
| `agent/` | Tool registry (calculator, Python sandbox, clock, pluggable web search) + the agent loop |
| `server/` | FastAPI app: `/chat`, `/tools`, `/health` |
| `web/` | Sarvam-style chat PWA — model runs client-side, history stays on-device, works offline (service worker) |
| `android/` | WebView shell → real APK, built in CI |
| `.github/workflows/` | Deploy the site, build the APK, and run the 150M/300M self-chaining training |

## Quickstart

```bash
git clone https://github.com/samratbarman1013-commits/transformer-llm.git
cd transformer-llm
pip install -r requirements.txt

# training data (1 MB, public domain)
mkdir -p data && curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt

# 1. train (CPU is fine — ~1M params; rerun to continue training)
python -m transformer.train --data data/tinyshakespeare.txt --steps 5000

# 2. chat in the terminal
python -m transformer.chat

# 3. export for the browser (PyTorch -> ONNX, parity-checked)
python -m transformer.export_onnx --out web/model/model.onnx

# 4. serve locally, or just open web/index.html
uvicorn server.app:app --port 8000
```

## Model

Decoder-only GPT: pre-LayerNorm blocks, multi-head causal attention
(GELU MLP, weight-tied head). The same code scales by config:

| Config | d_model | Layers | Context | ~Params |
|---|---|---|---|---|
| `proto-1m` ✅ | 128 | 5 | 256 | 1.0M |
| `small-15m` | 384 | 8 | 512 | 15M |
| `base-50m` | 512 | 16 | 1024 | 50M |
| `ci-150m` ✅ | 896 | 14 | 512 | 165M |
| `ci-300m` 🚀 | 1152 | 18 | 512 | 325M |
| `target-500m` | 1024 | 40 | 2048 | 500M |

Prototype facts: char-level tokenizer (65 symbols), trained on
TinyShakespeare, val loss **2.17** after 835 steps on 2 CPU cores.

**Honest scope.** A 1M-parameter model trained on 1 MB of Shakespeare is a
*real* language model — it learns letters, words, punctuation, dialogue
format — but it does not answer questions or follow instructions. It is the
architectural prototype: the training pipeline, agent loop, tools, API,
web app and APK are all production-shaped and stay as the model grows.
The next milestones (15M → 50M → 500M, BPE tokenizer, instruction tuning)
replace the brain, not the skeleton.

## Apps

- **Website (PWA)** — installable from the browser ("Add to Home screen"), works offline once loaded. The first deploy trains the model in CI and caches it; later deploys are instant.
- **Android APK** — Actions tab → *Build Android APK* → *Run workflow*; download the artifact, `adb install` it. Tags (`v*`) attach the APK to a GitHub Release automatically.

## Privacy

The web app and APK send **zero** requests of their own: the model file is
fetched from the site itself and inference happens on-device. History is
stored in localStorage / WebView DOM storage and can be wiped by clearing
site data. The only external script is the onnxruntime-web CDN (also cached
by the service worker for offline use).

## Roadmap

1. ✅ 1M prototype: train → export → in-browser inference
2. ✅ 150M `ci-150m` trained on GitHub Actions (2,000 steps, `v0.2.0`) — 🔄 extension to 6,000 steps in progress → `v0.3.0`
3. 🚀 300M `ci-300m` code-focused model (85% Python code) — self-chaining CI training in progress → `v0.4.0`
4. ⬜ 500M pretraining — see **[TRAINING_500M.md](TRAINING_500M.md)** and the self-resuming [Colab notebook](notebooks/train_500m_colab.ipynb) (needs a GPU)
5. ⬜ SFT: chat + tool-calling fine-tune (replaces the deterministic router in `agent/tools.py`), then DPO
6. ⬜ Desktop shells (Tauri/Electron) using the same ONNX core

## License

MIT
