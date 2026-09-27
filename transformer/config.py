"""Model configurations. The same architecture scales from prototype to production."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "proto-1m"
    d_model: int = 128        # embedding / residual width
    n_layers: int = 5         # transformer blocks
    n_heads: int = 4          # attention heads (d_model must be divisible)
    ctx: int = 256            # context window (tokens)
    dropout: float = 0.0      # 0.0 for tiny models; raise when scaling up
    gradient_checkpointing: bool = False  # trade compute for memory (big configs)


# --- Ladder of configs on the same architecture -------------------------
# param_count ~= n_layers * 12 * d_model^2  (+ embeddings, tied head)

PROTO_1M = ModelConfig(name="proto-1m", d_model=128, n_layers=5, n_heads=4, ctx=256)
SMALL_15M = ModelConfig(name="small-15m", d_model=384, n_layers=8, n_heads=6, ctx=512)
BASE_50M = ModelConfig(name="base-50m", d_model=512, n_layers=16, n_heads=8, ctx=1024)
TARGET_500M = ModelConfig(name="target-500m", d_model=1024, n_layers=40, n_heads=16, ctx=2048)
# Max that fits a free GitHub Actions runner (7-16 GB, CPU) with fp32 AdamW
CI_150M = ModelConfig(name="ci-150m", d_model=896, n_layers=14, n_heads=14, ctx=512)

CONFIGS = {c.name: c for c in (PROTO_1M, SMALL_15M, BASE_50M, CI_150M, TARGET_500M)}
