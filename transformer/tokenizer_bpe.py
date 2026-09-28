"""Byte-level BPE tokenizer for the 15M-500M configs.

The char tokenizer (tokenizer.py) is right for the 1M prototype; at 500M a
32k BPE vocab is the correct trade (fewer tokens per text -> more effective
context). Uses the Rust `tokenizers` library — training on ~1GB takes minutes.

CLI:
    python -m transformer.tokenizer_bpe --train data/corpus.txt --out tokenizer.json
    python -m transformer.tokenizer_bpe --encode data/corpus.txt --tokenizer tokenizer.json \
        --out-dir data/bins --val-frac 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPECIALS = ["<|endoftext|>"]


class BpeTokenizer:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))
        self.path = str(path)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    def save(self, path: str | Path):
        self.tok.save(str(path))


def train(text_path: str, out_path: str, vocab_size: int = 32768):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS + ["<|padding|>"],
        show_progress=True,
    )
    with open(text_path, encoding="utf-8") as f:
        tok.train_from_iterator(batched_read(f, 1 << 20), trainer=trainer)
    tok.save(out_path)
    t = BpeTokenizer(out_path)
    print(f"tokenizer: vocab {t.vocab_size} -> {out_path}")
    return t


def batched_read(f, chunk_chars: int):
    while True:
        chunk = f.read(chunk_chars)
        if not chunk:
            return
        yield chunk


ENCODE_PROGRESS_BYTES = 200_000_000  # print progress every ~200MB of text

def encode_file(text_path: str, tokenizer_path: str, out_dir: str,
                val_frac: float = 0.001):
    """Pre-tokenize corpus into train.bin / val.bin (uint16, memory-mappable).
    Streams in chunks to keep memory flat; chunks are split on newlines so
    BPE merges never cross chunk boundaries."""
    t = BpeTokenizer(tokenizer_path)
    parts = []
    total = last_p = 0
    with open(text_path, encoding="utf-8") as f:
        buf = []
        size = 0
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= 2 << 20:  # ~2MB chunks (small for low-RAM machines)
                parts.append(np.array(t.encode("".join(buf)), dtype=np.uint32))
                total += size
                buf, size = [], 0
                if total - last_p >= ENCODE_PROGRESS_BYTES:
                    last_p = total
                    print(f"  tokenized {total/1e9:.1f} GB of text so far...")
        if buf:
            parts.append(np.array(t.encode("".join(buf)), dtype=np.uint32))
    if not parts:
        raise RuntimeError(
            f"corpus file is empty or unreadable: {text_path} "
            "(an earlier download was interrupted - delete the file and re-run the corpus cell)")
    ids = np.concatenate(parts)
    del parts
    assert t.vocab_size <= 65536, "vocab too large for uint16 storage"
    n_val = max(1, int(len(ids) * val_frac))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.array(ids[:-n_val], dtype=np.uint16).tofile(out / "train.bin")
    np.array(ids[-n_val:], dtype=np.uint16).tofile(out / "val.bin")
    meta = {"vocab_size": t.vocab_size, "tokenizer": str(tokenizer_path),
            "train_tokens": int(len(ids) - n_val), "val_tokens": int(n_val)}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"train.bin {meta['train_tokens']:,} tokens | val.bin {meta['val_tokens']:,} tokens -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", help="text file to train the tokenizer on")
    ap.add_argument("--encode", help="text file to pre-tokenize to .bin")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--out-dir", default="data/bins")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--val-frac", type=float, default=0.001)
    a = ap.parse_args()
    if a.train:
        train(a.train, a.out, a.vocab_size)
    if a.encode:
        encode_file(a.encode, a.tokenizer, a.out_dir, a.val_frac)
