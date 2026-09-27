"""Train the Transformer.

Usage:
    python -m transformer.train --data data/tinyshakespeare.txt \
        --steps 5000 --batch 32 --max-seconds 240

Features: resume from checkpoint, periodic eval on a held-out split,
best-checkpoint tracking, JSONL training log.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .config import CONFIGS
from .model import GPT
from .tokenizer import CharTokenizer


def get_batch(data: torch.Tensor, block: int, batch: int):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, data, block, batch, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block, batch)
        losses.append(model.loss(x, y).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--config", default="proto-1m", choices=list(CONFIGS))
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--steps", type=int, default=5000, help="total steps across runs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-seconds", type=int, default=240,
                    help="wall-clock budget for this run; training resumes next run")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- data & tokenizer ---------------------------------------------------
    text = Path(args.data).read_text(encoding="utf-8")
    tok_path = out / "vocab.json"
    if tok_path.exists():
        tok = CharTokenizer.load(tok_path)
    else:
        tok = CharTokenizer.fit(text)
        tok.save(tok_path)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = len(ids)
    split = int(n * 0.9)
    train_data, val_data = ids[:split], ids[split:]
    print(f"data: {n:,} chars | vocab {tok.vocab_size} | device {device}")

    cfg = CONFIGS[args.config]

    # --- model / resume -----------------------------------------------------
    model = GPT(cfg, tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    step, best_val = 0, float("inf")

    ckpt_path = out / "transformer_1m.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed at step {step} (best val {best_val:.3f})")

    print(f"params: {model.num_params():,} ({cfg.name}: d={cfg.d_model} "
          f"L={cfg.n_layers} h={cfg.n_heads} ctx={cfg.ctx})")

    # --- train ---------------------------------------------------------------
    model.train()
    t0 = time.time()
    log = out / "train_log.jsonl"
    while step < args.steps:
        x, y = get_batch(train_data, cfg.ctx, args.batch)
        loss = model.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step:>5}/{args.steps} | train {loss.item():.3f} "
                  f"| lr {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if step % args.eval_every == 0 or step == args.steps:
            vl = eval_loss(model, val_data, cfg.ctx, args.batch)
            with open(log, "a") as f:
                f.write(json.dumps({"step": step, "train": round(loss.item(), 4),
                                    "val": round(vl, 4)}) + "\n")
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "config": cfg.name,
                            "vocab_size": tok.vocab_size, "step": step,
                            "best_val": best_val}, out / "transformer_1m_best.pt")

        if time.time() - t0 > args.max_seconds:
            print(f"time budget reached at step {step}; checkpoint saved — rerun to continue")
            break

    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "config": cfg.name,
                "vocab_size": tok.vocab_size, "step": step, "best_val": best_val},
               ckpt_path)
    print(f"done: step {step} | best val {best_val:.4f} | saved {ckpt_path.name} "
          f"+ transformer_1m_best.pt")


if __name__ == "__main__":
    main()
