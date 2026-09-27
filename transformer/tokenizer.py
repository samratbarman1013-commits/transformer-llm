"""Character-level tokenizer — tiny, dependency-free, deterministic.

At 1M params a char tokenizer keeps the embedding table small so parameters
go to the transformer body. Swap in a BPE tokenizer (tiktoken-style) when the
model scales past ~50M params — the rest of the codebase only needs
encode()/decode().
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = sorted(vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"])

    def save(self, path: str | Path):
        Path(path).write_text(
            json.dumps({"type": "char", "vocab": self.vocab}, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def encode_unk_safe(self, text: str, unk: int | None = None) -> list[int]:
        """Map unknown chars to `unk` instead of dropping them (inference)."""
        unk_id = unk if unk is not None else (max(self.stoi.values()) if self.vocab else 0)
        return [self.stoi.get(ch, unk_id) for ch in text]
