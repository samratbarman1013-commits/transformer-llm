"""Inference for the BPE models (15M-500M ladder): load + chat reply.

    from transformer.infer_v2 import load, reply
    model, tok, eos_id, config, vocab_size = load("model_best.pt", "tokenizer.json")
    print(reply(model, tok, eos_id, "User: hi"))
"""

from __future__ import annotations

import torch

from .config import CONFIGS
from .model import GPT
from .tokenizer_bpe import SPECIALS, BpeTokenizer

PROMPT = "Transformer:"
USER = "User:"


def load(checkpoint: str, tokenizer_json: str, device: str = "cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = CONFIGS[ck["config"]]
    model = GPT(cfg, ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BpeTokenizer(tokenizer_json)
    eos_id = tok.tok.token_to_id(SPECIALS[0]) if SPECIALS else None
    return model, tok, eos_id, ck["config"], ck["vocab_size"]


def reply(model, tok, eos_id, history: str, max_new=220, temperature=0.8,
          top_k=40) -> str:
    prompt = f"{history.strip()}\n{PROMPT}"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    stop_ids = (eos_id,) if eos_id is not None else ()
    out = model.generate(ids, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, stop_ids=stop_ids)[0].tolist()
    text = tok.decode(out[len(ids[0]):])
    for stop in (f"\n{USER}", f"\n{PROMPT}"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()
