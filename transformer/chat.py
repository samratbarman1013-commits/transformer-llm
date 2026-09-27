"""Interactive chat with the trained model (plain continuation engine).

    python -m transformer.chat --checkpoint checkpoints/transformer_1m_best.pt
"""

import argparse
from pathlib import Path

import torch

from .config import CONFIGS
from .model import GPT
from .tokenizer import CharTokenizer

PROMPT = "Transformer:"
USER = "User:"


def load(checkpoint: str, device: str = "cpu") -> tuple[GPT, CharTokenizer]:
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = CONFIGS[ck["config"]]
    model = GPT(cfg, ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = CharTokenizer.load(Path(checkpoint).parent / "vocab.json")
    return model, tok


def reply(model, tok, history: str, max_new=200, temperature=0.8, top_k=40) -> str:
    prompt = f"{history.strip()}\n{PROMPT}"
    ids = torch.tensor([tok.encode_unk_safe(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k)[0].tolist()
    text = tok.decode(out[len(ids[0]):])
    # stop at the next speaker turn, if the model produces one
    for stop in (f"\n{USER}", f"\n{PROMPT}"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/transformer_1m_best.pt")
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    model, tok = load(args.checkpoint)
    print("Transformer prototype ready. 'exit' to quit. History stays local.\n")
    history = ""
    while True:
        try:
            user = input(f"{USER} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        history += f"{USER} {user}\n"
        ans = reply(model, tok, history, max_new=args.max_new)
        history += f"{PROMPT} {ans}\n"
        print(f"{PROMPT} {ans}\n")


if __name__ == "__main__":
    main()
