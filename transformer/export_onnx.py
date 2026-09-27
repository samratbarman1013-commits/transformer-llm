"""Export the trained GPT to ONNX for in-browser (onnxruntime-web) inference.

    python -m transformer.export_onnx --checkpoint checkpoints/transformer_1m_best.pt \
        --out web/model/model.onnx
"""

import argparse

import numpy as np
import torch

from .chat import load


def export(checkpoint: str, out_path: str):
    model, tok = load(checkpoint)
    model.eval()
    dummy = torch.zeros(1, model.cfg.ctx, dtype=torch.long)

    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
        dynamo=False,  # legacy exporter: opset-17 output, runs on onnxruntime-web
    )
    print(f"exported {out_path} | vocab {tok.vocab_size} | ctx {model.cfg.ctx}")

    # parity check with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = torch.randint(0, tok.vocab_size, (2, 64), dtype=torch.long)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input_ids": x.numpy()})[0]
    print(f"parity: max abs diff {np.abs(ref - got).max():.2e}")
    assert np.abs(ref - got).max() < 1e-4, "ONNX output diverges from PyTorch"
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/transformer_1m_best.pt")
    ap.add_argument("--out", default="web/model/model.onnx")
    args = ap.parse_args()
    export(args.checkpoint, args.out)
