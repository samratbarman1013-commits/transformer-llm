# Training the 500M model — the 10–12 day plan

**Model:** `target-500m` — 539,502,592 parameters (d=1024, 40 layers, 16 heads,
ctx 1024–2048, 32k BPE vocab). Same architecture as the 1M prototype; nothing
else changes.

## Why this needs a GPU (and the plan around that)

| Resource | 500M feasibility |
|---|---|
| This repo's CI (2-core CPU, 7 GB) | ❌ weights + AdamW state alone = 6.5 GB |
| Colab free T4 16 GB | ✅ ~10–12 days (12 h/day session limits) |
| Kaggle free 2×T4 (30 h/week) | ✅ ~5–7 days — DDP via `torchrun` handled by the notebook |
| Colab Pro L4 / A100 | ✅ ~2–4 days |
| 1× A100 (any cloud) | ✅ ~1.5–2 days |

Token budget: default 40,000 steps × 65,536 tokens/step ≈ **2.6B tokens**
(~5 tokens/param — solid for this class). Raise `TOTAL_STEPS` in the notebook
for more.

## How to start (5 minutes)

1. Open `notebooks/train_500m_colab.ipynb` in **Google Colab** (or upload to
   Kaggle): File → Open notebook → GitHub →
   `samratbarman1013-commits/transformer-llm`.
2. Runtime → Change runtime type → **GPU**.
3. Run cells 1–5 once (download + tokenize, ~25 min).
4. Run **cell 6** and leave it. It trains in 2-hour chunks, checkpoints to
   Google Drive (`transformer-500m/`), and **auto-resumes** — after any
   disconnect just re-run cell 6. Cell 7 lets you chat with the result.

The training loop (`transformer/train_v2.py`) features: bf16/fp16 autocast
(auto), gradient accumulation, gradient checkpointing (needed at 500M on
16 GB), DDP for multi-GPU, cosine LR with warmup, exact optimizer-state
resume, tokens/s + ETA logging, and a `status.json` heartbeat.

## Supervisor

A scheduled agent checks `transformer-500m/status.json` in your Drive every
6 hours and reports: step / val loss / tokens seen / ETA, and alerts if the
run stalls >24 h. It's read-only — it never touches training.

## After pretraining

1. **Export & ship**: `python -m transformer.export_onnx` the best checkpoint
   (notebook cell 7 + README quickstart), swap `web/model/model.onnx`,
   push — the site redeploys. 539M fp32 = 2.2 GB; quantize (int8) or ship
   a distilled browser model — at this size on-device web inference is
   flagship-phone territory, the APK WebView still works via server mode.
2. **SFT** (makes it chatty): fine-tune on ~50–200k instruction pairs
   (e.g. `HuggingFaceTB/smoltalk`) — same trainer, new data, LoRA optional.
3. **DPO** alignment on preference pairs.
4. Publish weights on the GitHub release (assets up to 2 GB — save fp16).

## Honest expectations

A 500M model trained on ~2.6B tokens of TinyStories+WikiText writes fluent,
coherent English stories and text — it will *not* answer questions or
follow instructions until SFT. That's the normal order of operations:
pretrain → SFT → align.
