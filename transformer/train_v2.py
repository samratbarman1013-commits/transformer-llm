"""GPU trainer for the 15M -> 500M configs. (The 1M prototype uses train.py.)

Single GPU:
    python -m transformer.train_v2 --config target-500m --data data/bins \
        --steps 40000 --batch 8 --grad-accum 16 --grad-checkpoint

Multi GPU (e.g. Kaggle 2xT4):
    torchrun --nproc_per_node=2 -m transformer.train_v2 ...same flags...

Features: bf16/fp16 autocast (auto-detected), gradient accumulation,
gradient checkpointing, DDP, auto-resume from ckpt_last.pt, best-model
tracking, tokens/s + ETA logging, status.json for the external supervisor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import CONFIGS, ModelConfig
from .model import GPT


# ---------------- distributed helpers ----------------

def dist_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank, rank, world
    return 0, 0, 1


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_batch(data: np.memmap, ctx: int, batch: int, device):
    ix = np.random.randint(0, len(data) - ctx - 1, size=batch)
    x = torch.from_numpy(np.stack([data[i:i + ctx].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + ctx].astype(np.int64) for i in ix]))
    if device.type == "cuda":  # pin_memory is only meaningful for accelerator devices
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, val, ctx, batch, device, iters=20, amp_dtype=None):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val, ctx, batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            losses.append(model.loss(x, y).item())
    model.train()
    return sum(losses) / len(losses)


def write_status(path: Path, payload: dict):
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="target-500m", choices=list(CONFIGS))
    ap.add_argument("--data", default="data/bins", help="dir with train.bin/val.bin + meta.json")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--vocab-size", type=int, default=0, help="override (default: from meta.json)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8, help="per-process micro-batch")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=0, help="override config context")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-scale", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-seconds", type=int, default=0, help="0 = no limit")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--status-out", default="status.json")
    args = ap.parse_args()

    local_rank, rank, world = dist_setup()
    is_main = rank == 0
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available() and world > 1:
        torch.distributed.barrier()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text())
    vocab_size = args.vocab_size or meta["vocab_size"]
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    # ---------------- model ----------------
    cfg = CONFIGS[args.config]
    cfg = ModelConfig(**{**cfg.__dict__, "gradient_checkpointing": args.grad_checkpoint})
    if args.ctx:
        cfg.ctx = args.ctx
    if args.dropout is not None:
        cfg.dropout = args.dropout
    model = GPT(cfg, vocab_size).to(device)
    if is_main:
        print(f"config {cfg.name} | params {model.num_params():,} | ctx {cfg.ctx} "
              f"| vocab {vocab_size} | world {world} | device {device.type}")
        print(f"tokens/step (global): {args.batch * args.grad_accum * cfg.ctx * world:,}")

    raw_model = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None)

    # AMP dtype: bf16 if the hardware supports it, else fp16 + scaler
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype is torch.float16))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.min_lr_scale + (1 - args.min_lr_scale) * 0.5 * (1 + math.cos(math.pi * p)))

    # ---------------- resume ----------------
    step, best_val, tokens_seen = 0, float("inf"), 0
    ckpt_last = out / "ckpt_last.pt"
    if ckpt_last.exists() and not args.no_resume:
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best_val, tokens_seen = ck["step"], ck["best_val"], ck.get("tokens_seen", 0)
        if is_main:
            print(f"resumed: step {step}/{args.steps} | best val {best_val:.4f} "
                  f"| {tokens_seen:,} tokens seen")

    # ---------------- train ----------------
    model.train()
    t0 = time.time()
    loss_acc = 0.0
    status_path = Path(args.status_out)
    stop = False
    while step < args.steps and not stop:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, cfg.ctx, args.batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                 enabled=amp_dtype is not None):
                loss = model.loss(x, y) / args.grad_accum
            scaler.scale(loss).backward()
            micro_losses.append(loss.item())
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens_seen += args.batch * cfg.ctx * world
        loss_acc += sum(micro_losses) / len(micro_losses)

        if step % 50 == 0 and is_main:
            el = time.time() - t0
            tps = tokens_seen / el if el else 0
            print(f"step {step:>6}/{args.steps} | loss {loss_acc/50:.4f} "
                  f"| lr {lr_at(step):.2e} | tok/s {tps:,.0f} | elapsed {el/60:.1f}m")
            loss_acc = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            vl = eval_loss(model, val_data, cfg.ctx, max(1, min(args.batch, 8)), device,
                           amp_dtype=amp_dtype)
            if is_main:
                el = time.time() - t0
                remaining = (args.steps - step) * (el / max(1, step))
                write_status(status_path, {
                    "phase": "pretraining", "config": cfg.name,
                    "params": raw_model.num_params(),
                    "step": step, "steps_total": args.steps,
                    "val_loss": round(vl, 4), "tokens_seen": tokens_seen,
                    "tokens_per_s": round(tokens_seen / el) if el else 0,
                    "eta_hours": round(remaining / 3600, 1),
                    "best_val": round(best_val, 4),
                })
                with open(out / "train_log.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, "val": round(vl, 4),
                                        "tokens": tokens_seen}) + "\n")
                if vl < best_val:
                    best_val = vl
                    torch.save({"model": raw_model.state_dict(), "config": cfg.name,
                                "vocab_size": vocab_size, "step": step,
                                "best_val": best_val}, out / "model_best.pt")
                print(f"  eval @ {step}: val {vl:.4f} (best {best_val:.4f}) "
                      f"| ETA {remaining/3600:.1f}h")
            if world > 1:
                torch.distributed.barrier()

        if args.max_seconds and time.time() - t0 > args.max_seconds:
            if is_main:
                print("time budget reached — checkpoint saved, rerun to continue")
            stop = True

    if is_main:
        torch.save({"model": raw_model.state_dict(), "optimizer": opt.state_dict(),
                    "config": cfg.name, "vocab_size": vocab_size, "step": step,
                    "best_val": best_val, "tokens_seen": tokens_seen}, ckpt_last)
        write_status(status_path, {"phase": "stopped" if step < args.steps else "done",
                                    "config": cfg.name, "params": raw_model.num_params(),
                                    "step": step, "steps_total": args.steps,
                                    "best_val": round(best_val, 4),
                                    "tokens_seen": tokens_seen})
        print(f"saved {ckpt_last.name} at step {step}")
    cleanup()


if __name__ == "__main__":
    main()
