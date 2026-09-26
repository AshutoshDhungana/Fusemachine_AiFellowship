"""W16 agentic feature: cross-source research agent with self-check.

The model drives the loop: each turn it chooses ONE tool (search / read / note / ask_user / finish)
based on what the previous observation revealed. Stopping conditions:
  finish accepted by verifier | ask_user | max_steps | token budget | provider failure.

Context engineering (see README §a):
  * Structured external notes  -- `write_note` stores verified facts outside the transcript; the
    current notes are re-injected each turn as a single compact message.
  * Clearing tool results      -- raw search results older than the last 2 are replaced by a stub
    once they have been processed, so the transcript does not grow with ~4k tokens per search.
  * Capped/re-ranked retrieval -- search is hybrid (dense+BM25, RRF), capped at max_top_k, chunks truncated.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ROOT, get_settings
from ..llm.client import AllProvidersFailed, ResilientLLM
from ..tools import Tool, ToolBox, dumps
from .verifier import verify

log = logging.getLogger(__name__)
PROMPTS_DIR = ROOT / "prompts"

REASON = {"type": "string", "description": "One sentence: why this action now (logged in the trace)"}


@dataclass
class Step:
    step: int
    tool: str
    args: dict
    ok: bool
    result: Any
    reasoning: str
    tokens: int
    latency_ms: float
    provider: str = ""


@dataclass
class AgentResult:
    run_id: str
    question: str
    status: str                 # answered | needs_clarification | insufficient_evidence | max_steps | budget | error
    answer: str
    citations: list[str]
    confidence: float
    steps: list[Step] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    verifier_calls: int = 0
    verifier_tokens: int = 0
    termination_reason: str = ""
    latency_s: float = 0.0
    config: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.verifier_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["n_steps"] = len(self.steps)
        return d


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


class ResearchAgent:
    def __init__(self, llm: ResilientLLM, toolbox: ToolBox, prompt: str | None = None,
                 max_steps: int | None = None, verify_answers: bool | None = None,
                 clear_tool_results: bool = True, keep_last_results: int = 2,
                 temperature: float | None = None, top_k: int | None = None, max_tokens: int | None = None):
        s = get_settings()
        self.llm, self.tb = llm, toolbox
        self.prompt_name = prompt or s.agent_prompt
        self.system_prompt = load_prompt(self.prompt_name)
        self.max_steps = max_steps or s.agent_max_steps
        self.verify_answers = s.agent_verify if verify_answers is None else verify_answers
        self.clear_tool_results = clear_tool_results
        self.keep_last_results = keep_last_results
        self.temperature = s.temperature if temperature is None else temperature
        self.top_k = top_k or s.top_k
        self.token_budget = max_tokens or s.agent_max_tokens
        self._control_tools = self._make_control_tools()

    def config(self) -> dict:
        return {"prompt_version": self.prompt_name, "max_steps": self.max_steps, "verify": self.verify_answers,
                "clear_tool_results": self.clear_tool_results, "temperature": self.temperature,
                "top_k": self.top_k, "token_budget": self.token_budget,
                "models": [p.model for p in self.llm.providers], "fail_inject": self.tb.fail_inject or "none"}

    # ------------------------------------------------------------------ tools
    def _make_control_tools(self) -> dict[str, Tool]:
        async def noop(**_):
            return None
        return {
            "write_note": Tool("write_note",
                               "Record a fact you have verified from a chunk (external memory that survives "
                               "context clearing). Use after reading useful search results.",
                               {"type": "object", "properties": {
                                   "reason": REASON,
                                   "claim": {"type": "string"},
                                   "chunk_ids": {"type": "array", "items": {"type": "string"}},
                                   "paper_id": {"type": "string"}},
                                "required": ["reason", "claim", "chunk_ids"]}, noop),
            "ask_user": Tool("ask_user", "Ask the user a clarifying question when the request is ambiguous "
                                         "(e.g. an unclear referent). Ends the turn.",
                             {"type": "object", "properties": {"reason": REASON, "question": {"type": "string"}},
                              "required": ["reason", "question"]}, noop),
            "finish": Tool("finish", "Submit the final answer. Only cite chunk ids you actually retrieved.",
                           {"type": "object", "properties": {
                               "reason": REASON,
                               "answer": {"type": "string", "description": "Final answer in markdown"},
                               "citations": {"type": "array", "items": {"type": "string"}},
                               "confidence": {"type": "number"},
                               "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]}},
                            "required": ["reason", "answer", "citations", "status"]}, noop),
        }

    def _tool_schemas(self) -> list[dict]:
        schemas = []
        for name in ("search_papers", "get_chunk", "list_papers"):
            sch = json.loads(json.dumps(self.tb.tools[name].schema()))
            sch["function"]["parameters"]["properties"]["reason"] = REASON
            schemas.append(sch)
        return schemas + [t.schema() for t in self._control_tools.values()]

    # ------------------------------------------------------------------ context engineering
    def _notes_message(self, notes: list[dict]) -> dict:
        if not notes:
            body = "(no notes yet)"
        else:
            body = "\n".join(f"{i+1}. {n['claim']}  [{', '.join(n['chunk_ids'])}]" for i, n in enumerate(notes))
        return {"role": "user", "content": f"NOTES (your external memory, verified facts so far):\n{body}"}

    def _clear_old_results(self, messages: list[dict]) -> int:
        """Replace raw search/get_chunk results except the most recent `keep_last_results` with a stub."""
        idx = [i for i, m in enumerate(messages) if m.get("role") == "tool" and m.get("_kind") == "retrieval"
               and not m.get("_cleared")]
        cleared = 0
        for i in idx[:-self.keep_last_results] if self.keep_last_results else idx:
            m = messages[i]
            m["content"] = f"[cleared: {m.get('_summary', 'results')} -- facts worth keeping are in NOTES; " \
                           f"call get_chunk(<id>) to re-read a chunk]"
            m["_cleared"] = True
            cleared += 1
        return cleared

    @staticmethod
    def _wire(messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]

    # ------------------------------------------------------------------ main loop
    async def run(self, question: str, history: list[dict] | None = None, trace_path: Path | None = None) -> AgentResult:
        t0 = time.perf_counter()
        res = AgentResult(uuid.uuid4().hex[:10], question, "error", "", [], 0.0, config=self.config())
        notes: list[dict] = []
        retrieved: dict[str, dict] = {}  # chunk_id -> full chunk (for verifier + citation validation)
        catalog = "\n".join(f"- {k}: {v['title']}" for k, v in self.tb.catalog.items())
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt + f"\n\nCORPUS (paper_id: title):\n{catalog}"},
            *(history or [])[-4:],
            {"role": "user", "content": question},
            self._notes_message(notes),
        ]
        notes_idx = len(messages) - 1
        tools = self._tool_schemas()
        verifications = 0

        for step_no in range(1, self.max_steps + 1):
            if res.total_tokens > self.token_budget:
                res.status, res.termination_reason = "budget", f"token budget {self.token_budget} exceeded"
                break
            messages[notes_idx] = self._notes_message(notes)
            if self.clear_tool_results:
                self._clear_old_results(messages)
            try:
                r = await self.llm.chat(self._wire(messages), tools=tools, temperature=self.temperature,
                                        use_cache=False)
            except AllProvidersFailed as e:
                res.status, res.termination_reason = "error", f"LLM providers failed: {e.errors}"
                break
            res.llm_calls += 1
            res.prompt_tokens += r.prompt_tokens
            res.completion_tokens += r.completion_tokens

            if not r.tool_calls:  # model answered in prose -> nudge (counts as a step)
                messages.append({"role": "assistant", "content": r.content or ""})
                messages.append({"role": "user", "content": "Use a tool. To answer, call `finish`."})
                res.steps.append(Step(step_no, "(none)", {}, False, (r.content or "")[:500],
                                      "model replied without a tool call", r.total_tokens, r.latency_ms, r.provider))
                continue

            messages.append(r.assistant_message())
            done = False
            for tc in r.tool_calls:
                try:
                    args = tc.parsed_args()
                    parse_ok = isinstance(args, dict)
                except json.JSONDecodeError:
                    args, parse_ok = {}, False
                reasoning = str(args.get("reason", "")) or (r.content or "")
                ok, result, kind, summary = await self._execute(tc.name, args, parse_ok, notes, retrieved)
                args = {k: v for k, v in args.items() if k != "reason"}

                # ---- terminal actions
                if tc.name == "finish" and ok:
                    cits = [c for c in args.get("citations", []) if c in retrieved]
                    dropped = [c for c in args.get("citations", []) if c not in retrieved]
                    if self.verify_answers and args.get("status") == "answered" and verifications < 2:
                        verifications += 1
                        v, vr = await verify(self.llm, question, args["answer"], [retrieved[c] for c in cits])
                        res.verifier_calls += 1
                        res.verifier_tokens += vr.total_tokens
                        result = {"verifier": v.model_dump(), "dropped_citations": dropped}
                        if v.verdict != "supported" or dropped:
                            ok = False
                            result["instruction"] = ("Answer REJECTED by verifier. Fix the unsupported claims "
                                                     "(search for evidence or remove them), then call finish again.")
                    if ok:
                        res.status = args.get("status", "answered")
                        res.answer, res.citations = args["answer"], cits
                        res.confidence = float(args.get("confidence", 0.5) or 0.5)
                        res.termination_reason = f"finish ({res.status})"
                        done = True
                elif tc.name == "ask_user" and ok:
                    res.status, res.answer = "needs_clarification", args["question"]
                    res.termination_reason = "asked user for clarification"
                    done = True

                res.steps.append(Step(step_no, tc.name, args, ok, _short(result), reasoning,
                                      r.total_tokens, r.latency_ms, r.provider))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": dumps(result)[:6000],
                                 "_kind": kind, "_summary": summary})
                if done:
                    break
            if done:
                break
        else:
            res.status = "max_steps"
            res.termination_reason = f"max_steps={self.max_steps} reached without finish"
            if notes:  # best-effort, explicitly flagged partial answer from external notes
                res.answer = "PARTIAL (step limit reached). Verified notes so far:\n" + \
                             "\n".join(f"- {n['claim']}" for n in notes)
                res.citations = sorted({c for n in notes for c in n["chunk_ids"] if c in retrieved})

        res.notes = notes
        res.latency_s = round(time.perf_counter() - t0, 2)
        if trace_path:
            write_trace(res, trace_path)
        return res

    async def _execute(self, name, args, parse_ok, notes, retrieved):
        """Returns (ok, result, kind, summary)."""
        if not parse_ok:
            return False, "ERROR: arguments were not valid JSON", "error", ""
        if name in self._control_tools:
            problems = self._control_tools[name].validate(args)
            if problems:
                return False, "ERROR: invalid arguments: " + "; ".join(problems), "error", ""
            if name == "write_note":
                bad = [c for c in args["chunk_ids"] if c not in retrieved]
                if bad:
                    return False, f"ERROR: chunk ids {bad} were never retrieved; notes must cite retrieved chunks", "error", ""
                notes.append({"claim": args["claim"], "chunk_ids": args["chunk_ids"], "paper_id": args.get("paper_id")})
                return True, {"notes_count": len(notes)}, "note", ""
            return True, {"accepted": True}, "control", ""
        args = {k: v for k, v in args.items() if k != "reason"}
        if name == "search_papers" and "k" in args and isinstance(args["k"], int):
            args["k"] = min(args["k"], self.tb.s.max_top_k)
        ok, result = await self.tb.call(name, args)
        if ok and name == "search_papers":
            valid = [h for h in result["results"] if h.get("chunk_id") and h.get("paper_id")]
            if len(valid) < len(result["results"]):
                result["warning"] = (f"{len(result['results']) - len(valid)} results were MALFORMED "
                                     f"(missing ids/garbled text) and must not be used as evidence")
            for h in valid:
                retrieved[h["chunk_id"]] = h
            return ok, result, "retrieval", f"{len(valid)} chunks for '{args.get('query')}'" + \
                (f" in {args['paper_id']}" if args.get("paper_id") else "")
        if ok and name == "get_chunk":
            retrieved[result["chunk_id"]] = result
            return ok, result, "retrieval", f"chunk {result['chunk_id']}"
        return ok, result, "retrieval" if ok else "error", name


def _short(x: Any, n: int = 2500) -> Any:
    s = dumps(x)
    return x if len(s) <= n else s[:n] + " …[truncated in trace]"


def write_trace(res: AgentResult, path: Path) -> None:
    """One JSON object per step + a final summary record (JSONL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for st in res.steps:
            f.write(json.dumps({"run_id": res.run_id, **asdict(st)}, ensure_ascii=False, default=str) + "\n")
        f.write(json.dumps({"run_id": res.run_id, "summary": True, "question": res.question, "status": res.status,
                            "termination_reason": res.termination_reason, "n_steps": len(res.steps),
                            "total_tokens": res.total_tokens, "answer": res.answer, "citations": res.citations,
                            "config": res.config}, ensure_ascii=False) + "\n")
