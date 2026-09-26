"""FastAPI backend (async). Endpoints:
  GET  /health            component status (vector store, router, providers)
  POST /chat              W15 RAG assistant (structured JSON answer)
  POST /agent             W16 research agent (returns answer + full trace)
  POST /route             W14 router via ONNX Runtime (micro-batched)
  POST /route/batch       batch routing
  GET  /stats             cache / provider / router counters

Run: uv run uvicorn assistant.api:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent.loop import ResearchAgent
from .assistant import Assistant
from .config import get_settings
from .llm.client import ResilientLLM, TokenBucket
from .rag.store import VectorStore
from .tools import ToolBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    STATE["store"] = VectorStore(s)
    if STATE["store"].count() == 0:
        log.warning("vector store empty -> ingesting papers now")
        from .rag.ingest import ingest
        ingest(s)
        STATE["store"] = VectorStore(s)
    try:
        from .intent.onnx_router import OnnxIntentRouter
        STATE["router"] = OnnxIntentRouter(s)
        await STATE["router"].start()
    except Exception as e:  # noqa: BLE001 -- degrade: /route returns 503, chat still works
        log.warning("intent router not loaded: %s", e)
        STATE["router"] = None
    STATE["llm"] = ResilientLLM(s)
    STATE["tb"] = ToolBox(STATE["store"], STATE["router"], s)
    STATE["assistant"] = Assistant(STATE["llm"], STATE["tb"])
    yield
    if STATE.get("router"):
        await STATE["router"].stop()


app = FastAPI(title="PaperPilot AI Assistant", version="0.3.0", lifespan=lifespan)
_buckets: dict[str, TokenBucket] = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Inbound per-client token bucket (graceful 429 with Retry-After)."""
    if request.url.path in ("/chat", "/agent"):  # LLM-backed endpoints only; /route is cheap
        key = request.headers.get("x-api-key") or (request.client.host if request.client else "anon")
        b = _buckets.setdefault(key, TokenBucket(get_settings().api_rpm_per_client))
        if not b.try_acquire():
            return JSONResponse({"error": "rate limit exceeded", "retry_after_s": 5}, status_code=429,
                                headers={"Retry-After": "5"})
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return resp


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"error": "internal error", "detail": type(exc).__name__}, status_code=500)


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    history: list[dict] = Field(default_factory=list)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    k: int | None = Field(None, ge=1, le=8)


class AgentIn(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    history: list[dict] = Field(default_factory=list)
    max_steps: int | None = Field(None, ge=1, le=15)
    verify: bool | None = None
    prompt: str | None = None


class RouteIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class RouteBatchIn(BaseModel):
    messages: list[str] = Field(min_length=1, max_length=256)


@app.get("/health")
async def health():
    llm: ResilientLLM | None = STATE.get("llm")
    return {"status": "ok", "chunks": STATE["store"].count() if STATE.get("store") else 0,
            "router": STATE.get("router").model_file if STATE.get("router") else None,
            "providers": [{"name": p.name, "model": p.model, "circuit_open": not p.available()}
                          for p in (llm.providers if llm else [])]}


@app.post("/chat")
async def chat(body: ChatIn):
    return await STATE["assistant"].answer(body.question, body.history, body.temperature, body.top_p, body.k)


@app.post("/agent")
async def agent(body: AgentIn):
    ag = ResearchAgent(STATE["llm"], STATE["tb"], prompt=body.prompt, max_steps=body.max_steps,
                       verify_answers=body.verify)
    return (await ag.run(body.question, body.history)).to_dict()


@app.post("/route")
async def route(body: RouteIn):
    if not STATE.get("router"):
        raise HTTPException(503, "intent router not available (export the W14 model to ONNX first)")
    t0 = time.perf_counter()
    out = await STATE["router"].route(body.message)
    return {**out, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/route/batch")
async def route_batch(body: RouteBatchIn):
    if not STATE.get("router"):
        raise HTTPException(503, "intent router not available")
    import asyncio
    t0 = time.perf_counter()
    res = await asyncio.to_thread(STATE["router"].predict_batch, body.messages)
    return {"results": res, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/stats")
async def stats():
    r = STATE.get("router")
    return {**STATE["llm"].stats(),
            "router": {"batches": r.batches, "items": r.items,
                       "avg_batch": round(r.items / r.batches, 2) if r and r.batches else 0} if r else None}
