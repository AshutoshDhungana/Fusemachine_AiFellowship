"""Resilient multi-provider LLM client.

Every provider is reached through the OpenAI-compatible Chat Completions protocol
(Gemini exposes one, vLLM serves one), so tool calling / JSON mode work the same way.

Reliability features
- token-bucket rate limiter per provider (stay under free-tier RPM)
- retry with exponential backoff + jitter on 429 / 5xx / timeouts
- circuit breaker per provider
- ordered fallback chain: gemini-flash -> gemini-flash-lite -> local vLLM
- TTL/LRU response cache (prompt/response caching bonus)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from ..config import Settings, get_settings

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- data types
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as returned by the model

    def parsed_args(self) -> dict:
        return json.loads(self.arguments or "{}")


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    provider: str
    model: str
    latency_ms: float
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def assistant_message(self) -> dict:
        """The message to append to the conversation (OpenAI format)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}
                for t in self.tool_calls
            ]
        return msg


class AllProvidersFailed(RuntimeError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("All LLM providers failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


# ----------------------------------------------------------------- helpers
class TokenBucket:
    """Async token bucket: `rate` requests per 60 s, burst = rate."""

    def __init__(self, rpm: int):
        self.capacity = max(1, rpm)
        self.tokens = float(self.capacity)
        self.refill_per_s = self.capacity / 60.0
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_s)
        self.updated = now

    def try_acquire(self) -> bool:
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def acquire(self, max_wait_s: float = 30.0) -> bool:
        async with self._lock:
            deadline = time.monotonic() + max_wait_s
            while True:
                if self.try_acquire():
                    return True
                wait = (1 - self.tokens) / self.refill_per_s
                if time.monotonic() + wait > deadline:
                    return False
                await asyncio.sleep(wait)


class TTLCache:
    def __init__(self, max_items: int, ttl_s: int):
        self.max_items, self.ttl_s = max_items, ttl_s
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key: str):
        item = self._d.get(key)
        if item and time.time() - item[0] < self.ttl_s:
            self._d.move_to_end(key)
            self.hits += 1
            return item[1]
        self._d.pop(key, None)
        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        self._d[key] = (time.time(), value)
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


@dataclass
class Provider:
    name: str
    model: str
    client: AsyncOpenAI
    bucket: TokenBucket
    supports_json_schema: bool = True
    consecutive_failures: int = 0
    open_until: float = 0.0
    calls: int = 0
    failures: int = 0
    tokens: int = 0

    def available(self) -> bool:
        return time.monotonic() >= self.open_until


RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)


# ----------------------------------------------------------------- client
class ResilientLLM:
    def __init__(self, settings: Settings | None = None, providers: list[Provider] | None = None):
        self.s = settings or get_settings()
        self.providers = providers if providers is not None else self._default_providers()
        self.cache = TTLCache(self.s.cache_max_items, self.s.cache_ttl_seconds)

    def _default_providers(self) -> list[Provider]:
        s, out = self.s, []
        if s.gemini_api_key:
            gem = AsyncOpenAI(api_key=s.gemini_api_key, base_url=s.gemini_base_url,
                              timeout=s.request_timeout_s, max_retries=0)
            out.append(Provider("gemini-primary", s.primary_model, gem, TokenBucket(s.llm_rpm)))
            if s.fallback_model:
                out.append(Provider("gemini-fallback", s.fallback_model, gem, TokenBucket(s.llm_rpm)))
        if s.enable_vllm:
            vllm = AsyncOpenAI(api_key="EMPTY", base_url=s.vllm_base_url, timeout=s.request_timeout_s, max_retries=0)
            out.append(Provider("vllm-local", s.vllm_model, vllm, TokenBucket(600)))
        if not out:
            raise RuntimeError("No LLM provider configured: set GEMINI_API_KEY and/or ENABLE_VLLM")
        return out

    # ---- public API --------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        json_schema: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        params = dict(
            temperature=self.s.temperature if temperature is None else temperature,
            top_p=self.s.top_p if top_p is None else top_p,
            max_tokens=max_tokens or self.s.max_output_tokens,
        )
        key = self._cache_key(messages, tools, json_schema, params)
        if use_cache and (hit := self.cache.get(key)):
            return LLMResponse(**{**hit.__dict__, "cached": True, "latency_ms": 0.0})

        errors: dict[str, str] = {}
        for p in self.providers:
            if not p.available():
                errors[p.name] = "circuit open"
                continue
            try:
                resp = await self._call_with_retry(p, messages, tools, tool_choice, json_schema, params)
                p.consecutive_failures = 0
                if use_cache:
                    self.cache.set(key, resp)
                return resp
            except Exception as e:  # noqa: BLE001 -- we fall through to the next provider
                p.failures += 1
                p.consecutive_failures += 1
                if p.consecutive_failures >= self.s.circuit_fail_threshold:
                    p.open_until = time.monotonic() + self.s.circuit_cooldown_s
                    log.warning("circuit opened for %s", p.name)
                errors[p.name] = f"{type(e).__name__}: {str(e)[:200]}"
                log.warning("provider %s failed, falling back: %s", p.name, errors[p.name])
        raise AllProvidersFailed(errors)

    async def chat_json(self, messages: list[dict], schema_model, **kw):
        """Structured output: JSON-schema constrained generation + pydantic validation + 1 repair try."""
        schema = schema_model.model_json_schema()
        resp = await self.chat(messages, json_schema=schema, **kw)
        for attempt in range(2):
            try:
                return schema_model.model_validate_json(_strip_fences(resp.content or "")), resp
            except Exception as e:  # invalid JSON -> ask the model to repair once
                if attempt == 1:
                    raise
                repair = messages + [
                    {"role": "assistant", "content": resp.content or ""},
                    {"role": "user", "content": f"That was not valid JSON for the schema ({e}). "
                                                f"Return ONLY a JSON object matching: {json.dumps(schema)}"},
                ]
                resp = await self.chat(repair, json_schema=schema, use_cache=False, **kw)

    def stats(self) -> dict:
        return {
            "cache": {"hits": self.cache.hits, "misses": self.cache.misses, "size": len(self.cache._d)},
            "providers": [
                {"name": p.name, "model": p.model, "calls": p.calls, "failures": p.failures,
                 "tokens": p.tokens, "circuit_open": not p.available()}
                for p in self.providers
            ],
        }

    # ---- internals ---------------------------------------------------------
    async def _call_with_retry(self, p: Provider, messages, tools, tool_choice, json_schema, params) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.s.retry_attempts):
            if not await p.bucket.acquire(max_wait_s=self.s.request_timeout_s):
                raise TimeoutError(f"rate limiter: no capacity for {p.name}")
            try:
                return await self._call_once(p, messages, tools, tool_choice, json_schema, params)
            except RETRYABLE as e:
                last = e
                delay = self.s.retry_base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                log.info("retryable error on %s (attempt %d): %s; sleeping %.1fs", p.name, attempt + 1, e, delay)
                await asyncio.sleep(delay)
            except openai.BadRequestError as e:
                # JSON-schema mode not supported by this provider -> degrade to json_object once
                if json_schema and p.supports_json_schema and "response_format" in str(e).lower():
                    p.supports_json_schema = False
                    continue
                raise
        raise last or RuntimeError("retries exhausted")

    async def _call_once(self, p: Provider, messages, tools, tool_choice, json_schema, params) -> LLMResponse:
        kwargs: dict[str, Any] = dict(model=p.model, messages=messages, **params)
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if json_schema is not None:
            if p.supports_json_schema:
                kwargs["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "output", "schema": json_schema}}
            else:
                kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        p.calls += 1
        r = await p.client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        choice = r.choices[0].message
        calls = [ToolCall(tc.id or f"call_{i}", tc.function.name, tc.function.arguments or "{}")
                 for i, tc in enumerate(choice.tool_calls or [])]
        pt = getattr(r.usage, "prompt_tokens", 0) or 0
        ct = getattr(r.usage, "completion_tokens", 0) or 0
        p.tokens += pt + ct
        return LLMResponse(choice.content, calls, pt, ct, p.name, p.model, latency)

    @staticmethod
    def _cache_key(messages, tools, json_schema, params) -> str:
        blob = json.dumps([messages, tools, json_schema, params], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()
