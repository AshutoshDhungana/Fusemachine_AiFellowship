"""Export the W14 fine-tuned encoder router to ONNX and apply inference optimisations.

    uv sync --extra export
    uv run python -m assistant.intent.export_onnx --src ../../Wk14_FineTuningTransformersforIntentClassification/artifacts/router_encoder

Produces artifacts/router_onnx/{model.onnx, model.int8.onnx, tokenizer.json, labels.json, benchmark.json}
Optimisations:
  1. ONNX graph + ONNX Runtime graph optimisations (constant folding, attention/GELU fusion at ORT_ENABLE_ALL)
  2. Dynamic INT8 quantisation of the Linear layers (quantize_dynamic)  -> ~4x smaller, faster on CPU
  3. Dynamic batch/sequence axes so the server can micro-batch requests
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

SAMPLES = [
    "I placed an order yesterday but haven't received any confirmation email",
    "My card was charged twice for the same order, I need a refund",
    "how do i change my shipping address",
    "i want to delete my account",
    "what's the fee if I cancel",
    "can I get my invoice for last month",
    "unsubscribe me from the newsletter",
    "I want to talk to a real person",
]


def main():
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir saved by the W14 notebook (artifacts/router_encoder)")
    ap.add_argument("--out", default="artifacts/router_onnx")
    ap.add_argument("--max-len", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.src)
    model = AutoModelForSequenceClassification.from_pretrained(a.src).eval()
    enc = tok(SAMPLES[:2], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")

    fp32 = out / "model.onnx"
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    int8 = out / "model.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    # tokenizer + labels for the lightweight runtime (no torch/transformers at serve time)
    tok.save_pretrained(out / "hf_tokenizer")
    shutil.copy(out / "hf_tokenizer" / "tokenizer.json", out / "tokenizer.json")
    (out / "labels.json").write_text(json.dumps({int(k): v for k, v in model.config.id2label.items()}, indent=1))
    (out / "meta.json").write_text(json.dumps({"max_len": a.max_len, "source": a.src}))

    # ---- benchmark: PyTorch vs ONNX fp32 vs ONNX int8 (CPU, batch 1) + label agreement
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {n: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
                for n, p in [("onnx_fp32", fp32), ("onnx_int8", int8)]}

    def run_torch(t):
        e = tok(t, return_tensors="pt", truncation=True, max_length=a.max_len)
        with torch.no_grad():
            return int(model(**e).logits.argmax(-1))

    def run_ort(sess, t):
        e = tok(t, return_tensors="np", truncation=True, max_length=a.max_len)
        return int(sess.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                                   "attention_mask": e["attention_mask"].astype(np.int64)})[0].argmax(-1))

    bench, preds = {}, {}
    runners = {"pytorch_fp32": run_torch, **{n: (lambda s: (lambda t: run_ort(s, t)))(s) for n, s in sessions.items()}}
    for name, fn in runners.items():
        for t in SAMPLES[:3]:
            fn(t)
        t0 = time.perf_counter()
        reps = 20
        preds[name] = [fn(t) for _ in range(reps) for t in SAMPLES][: len(SAMPLES)]
        bench[name] = {"ms_per_query": (time.perf_counter() - t0) * 1000 / (reps * len(SAMPLES))}
    for name in bench:
        bench[name]["agreement_with_pytorch"] = float(np.mean(np.array(preds[name]) == np.array(preds["pytorch_fp32"])))
    bench["size_mb"] = {"onnx_fp32": fp32.stat().st_size / 2**20, "onnx_int8": int8.stat().st_size / 2**20}
    (out / "benchmark.json").write_text(json.dumps(bench, indent=2))
    print(json.dumps(bench, indent=2))


if __name__ == "__main__":
    main()
