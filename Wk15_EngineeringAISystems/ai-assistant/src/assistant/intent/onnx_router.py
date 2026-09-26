"""Serving-time router: ONNX Runtime + `tokenizers` only (no torch), with async micro-batching.

Concurrent /route requests are queued and coalesced into one batched ONNX call
(up to `router_max_batch` items or `router_max_wait_ms`), which raises throughput
under load while keeping single-request latency at a few ms.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings

log = logging.getLogger(__name__)


class OnnxIntentRouter:
    def __init__(self, s: Settings | None = None, prefer_int8: bool = True):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.s = s or get_settings()
        d: Path = self.s.router_dir
        model = d / ("model.int8.onnx" if prefer_int8 and (d / "model.int8.onnx").exists() else "model.onnx")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        self.model_file = model.name
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {"max_len": 64}
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(meta["max_len"])
        self.tok.enable_padding()
        self.labels = {int(k): v for k, v in json.loads((d / "labels.json").read_text()).items()}
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self.batches = 0
        self.items = 0

    # ---------------- sync batch inference
    def predict_batch(self, texts: list[str]) -> list[dict]:
        encs = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        out = []
        for p in probs:
            i = int(p.argmax())
            top = np.argsort(-p)[:3]
            out.append({"agent": self.labels[i], "confidence": float(p[i]),
                        "top3": [{"agent": self.labels[int(j)], "p": round(float(p[j]), 4)} for j in top]})
        return out

    # ---------------- async micro-batching
    async def start(self):
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._loop())

    async def stop(self):
        if self._worker:
            self._worker.cancel()

    async def route(self, text: str) -> dict:
        if self._queue is None:
            return (await asyncio.to_thread(self.predict_batch, [text]))[0]
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((text, fut))
        return await fut

    async def _loop(self):
        assert self._queue is not None
        while True:
            text, fut = await self._queue.get()
            batch = [(text, fut)]
            deadline = time.monotonic() + self.s.router_max_wait_ms / 1000
            while len(batch) < self.s.router_max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            try:
                res = await asyncio.to_thread(self.predict_batch, [t for t, _ in batch])
                for (_, f), r in zip(batch, res):
                    f.set_result(r)
                self.batches += 1
                self.items += len(batch)
            except Exception as e:  # noqa: BLE001
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
