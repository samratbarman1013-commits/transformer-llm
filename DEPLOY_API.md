# Deploying the Transformer API (private model)

The Android APK and the `/app` web build are **thin clients**: they hold no
model and cannot chat offline. All inference happens on a server you control,
behind an API key. This keeps the large model private.

```
APK / web app  ──POST /api/chat (X-API-Key)──▶  your server (holds the model)
```

## 1. Get the model files (owner only)

The training chain never publishes the model publicly. When a chain finishes,
download the checkpoint from the completed CI run (Artifacts → `ckpt`):

```bash
gh run download <run-id> --repo samratbarman1013-commits/transformer-llm \
   --name ckpt --dir server-data
```

You need `server-data/model_best.pt`. The matching tokenizer is
`data/tokenizer.json` from the corpus cache (or the chain's artifact).

## 2. Run the server

**Anywhere with Python 3.12:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r server/requirements.txt
export TRANSFORMER_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Docker:**

```bash
docker build -t transformer-api .
docker run -p 8000:8000 -e TRANSFORMER_API_KEY=<secret> \
  -v $(pwd)/server-data:/app/server-data transformer-api
```

**Free hosting options:**

| Host | Notes |
|---|---|
| Hugging Face Spaces (CPU basic, free) | 2 vCPU / 16 GB RAM — enough for the 150M model. Make the Space **private**; clients still reach the API with the key. Sleeps when idle, wakes on first request (slow first reply). |
| Home PC / laptop | `uvicorn --host 0.0.0.0` + port-forward on your router, or a tunnel (cloudflared, tailscale). Zero cost, always on your terms. |
| Any VPS (~₹400/mo) | Most reliable. Docker one-liner above. |

The 150M model needs roughly 1.5–2 GB RAM (fp32 weights + inference).
The 1M prototype runs in well under 1 GB.

## 3. Point the app at it

In the app: ⋮ menu → **Server settings** → paste the server URL
(e.g. `https://your-space.hf.space`) and the API key → Connect.
The URL and key live only in the device's localStorage.

## 4. Security notes (honest limits)

- An API key shipped inside an APK can be extracted by a determined user.
  For real distribution, issue per-user keys and rate-limit on the server.
  For personal use (your phone, your friends), a single key is fine.
- The server sees each prompt sent for inference. **Chat history is NOT
  stored** anywhere except on the device (the app keeps it in localStorage).
- Keep `server-data/` out of git (already in `.gitignore`).
