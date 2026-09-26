"""W15 assistant: single-pass RAG + tool calling + structured JSON output.

Flow (fixed pipeline):
  1. retrieve top-k chunks for the question (hybrid search)
  2. LLM call with the context + tools (list_papers, route_support_intent); up to 2 tool rounds
  3. final LLM call constrained to the AnswerOut JSON schema (validated with pydantic)
  4. graceful degradation: if every LLM provider fails, return the retrieved passages instead
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .config import ROOT, get_settings
from .llm.client import AllProvidersFailed, ResilientLLM
from .tools import ToolBox, dumps

log = logging.getLogger(__name__)


class Citation(BaseModel):
    paper_id: str
    page: int
    chunk_id: str


class AnswerOut(BaseModel):
    answer: str = Field(description="Answer in markdown, grounded ONLY in the provided context")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, description="0-1 self-assessed confidence")
    follow_up_questions: list[str] = Field(default_factory=list, max_length=3)


SYSTEM_PROMPT = (ROOT / "prompts" / "assistant_system.md").read_text(encoding="utf-8") \
    if (ROOT / "prompts" / "assistant_system.md").exists() else "You are a research assistant."


class Assistant:
    def __init__(self, llm: ResilientLLM, toolbox: ToolBox):
        self.s = get_settings()
        self.llm, self.tb = llm, toolbox

    async def answer(self, question: str, history: list[dict] | None = None,
                     temperature: float | None = None, top_p: float | None = None, k: int | None = None) -> dict:
        t0 = time.perf_counter()
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        ok, ret = await self.tb.call("search_papers", {"query": question, "k": k or self.s.top_k})
        hits = ret["results"] if ok else []
        context = "\n\n".join(f"[{h['chunk_id']}] (paper={h['paper_id']}, p.{h['page']})\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or [])[-6:],
                    {"role": "user", "content": f"CONTEXT:\n{context or '(retrieval failed)'}\n\nQUESTION: {question}"}]
        tools = self.tb.schemas(["list_papers", "route_support_intent"])
        tool_trace = []
        try:
            for _ in range(2):  # bounded tool rounds
                r = await self.llm.chat(messages, tools=tools, temperature=temperature, top_p=top_p)
                usage["prompt_tokens"] += r.prompt_tokens
                usage["completion_tokens"] += r.completion_tokens
                if not r.tool_calls:
                    break
                messages.append(r.assistant_message())
                for tc in r.tool_calls:
                    try:
                        args = tc.parsed_args()
                    except json.JSONDecodeError:
                        args = {}
                    tok, res = await self.tb.call(tc.name, args)
                    tool_trace.append({"tool": tc.name, "args": args, "ok": tok})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": dumps(res)[:4000]})
            messages.append({"role": "user", "content": "Now produce the final answer as JSON matching the schema."})
            out, r = await self.llm.chat_json(messages, AnswerOut, temperature=temperature, top_p=top_p)
            usage["prompt_tokens"] += r.prompt_tokens
            usage["completion_tokens"] += r.completion_tokens
            valid_ids = {h["chunk_id"] for h in hits}
            out.citations = [c for c in out.citations if c.chunk_id in valid_ids]  # drop hallucinated citations
            return {"status": "ok", **out.model_dump(), "provider": r.provider, "model": r.model,
                    "cached": r.cached, "tools_used": tool_trace, "usage": usage,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except AllProvidersFailed as e:
            log.error("degraded mode: %s", e)
            return {"status": "degraded",
                    "answer": "All language-model providers are currently unavailable. "
                              "Here are the most relevant passages I found:\n\n" +
                              "\n\n".join(f"- **{h['paper_id']} p.{h['page']}**: {h['text'][:300]}…" for h in hits[:3]),
                    "citations": [{"paper_id": h["paper_id"], "page": h["page"], "chunk_id": h["chunk_id"]} for h in hits[:3]],
                    "confidence": 0.0, "follow_up_questions": [], "errors": e.errors, "usage": usage,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
