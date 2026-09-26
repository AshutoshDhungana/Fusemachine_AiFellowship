"""Offline tests: scripted fake LLM + fake vector store. No network / API key needed."""
import asyncio
import json

from assistant.agent.loop import ResearchAgent
from assistant.llm.client import LLMResponse, ToolCall
from assistant.rag.store import Hit
from assistant.tools import ToolBox


class FakeStore:
    def search(self, q, k=5, paper_id=None):
        return [Hit("ai4va:p1:c0", "ai4va", "AI4VA", 1,
                    "AI4VA encodes source code into a Code Property Graph and trains a Gated Graph Neural Network.", .9)]

    def get(self, cid):
        return self.search("")[0] if cid == "ai4va:p1:c0" else None


class ScriptedLLM:
    """Returns pre-scripted tool calls; json calls (verifier) return a verdict."""
    def __init__(self, script, verdict="supported"):
        self.script, self.verdict, self.i = script, verdict, 0
        self.providers = []

    async def chat(self, messages, tools=None, **kw):
        name, args = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return LLMResponse(None, [ToolCall(f"c{self.i}", name, json.dumps(args))], 100, 20, "fake", "fake", 1.0)

    async def chat_json(self, messages, model, **kw):
        v = model(verdict=self.verdict, unsupported_claims=[], feedback="ok")
        return v, LLMResponse("{}", [], 50, 10, "fake", "fake", 1.0)


FINISH = ("finish", {"reason": "done", "answer": "AI4VA uses a Code Property Graph and a GGNN.",
                     "citations": ["ai4va:p1:c0"], "confidence": 0.9, "status": "answered"})
SEARCH = ("search_papers", {"reason": "find evidence", "query": "AI4VA representation"})
NOTE = ("write_note", {"reason": "save", "claim": "AI4VA uses CPG + GGNN", "chunk_ids": ["ai4va:p1:c0"]})


def make(script, inject="", verdict="supported", max_steps=6):
    tb = ToolBox(FakeStore(), None, fail_inject=inject)
    return ResearchAgent(ScriptedLLM(script, verdict), tb, max_steps=max_steps, verify_answers=True)


def test_happy_path():
    r = asyncio.run(make([SEARCH, NOTE, FINISH]).run("What does AI4VA use?"))
    assert r.status == "answered" and r.citations == ["ai4va:p1:c0"] and len(r.steps) == 3
    assert r.verifier_calls == 1 and r.notes


def test_hallucinated_citation_rejected():
    bad = ("finish", {**FINISH[1], "citations": ["linevd:p9:c9"]})
    r = asyncio.run(make([bad, SEARCH, FINISH]).run("q"))
    assert r.status == "answered" and r.steps[0].ok is False  # first finish rejected (uncited evidence)


def test_note_must_cite_retrieved_chunk():
    r = asyncio.run(make([NOTE, SEARCH, FINISH]).run("q"))
    assert r.steps[0].ok is False and "never retrieved" in r.steps[0].result


def test_max_steps_stop():
    r = asyncio.run(make([SEARCH], max_steps=3).run("q"))
    assert r.status == "max_steps" and len(r.steps) == 3


def test_unavailable_injection_reports_error():
    r = asyncio.run(make([SEARCH, ("finish", {"reason": "tools down", "answer": "Search is unavailable; cannot answer.",
                                             "citations": [], "status": "insufficient_evidence"})],
                         inject="unavailable").run("q"))
    assert r.steps[0].ok is False and "unavailable" in r.steps[0].result and r.status == "insufficient_evidence"


def test_malformed_results_not_citable():
    r = asyncio.run(make([SEARCH, FINISH, FINISH], inject="malformed", max_steps=3).run("q"))
    assert "MALFORMED" in json.dumps(r.steps[0].result)
    assert r.status != "answered"  # cannot cite garbage chunks -> never accepted


def test_context_clearing():
    ag = make([SEARCH])
    msgs = [{"role": "tool", "content": "x" * 1000, "_kind": "retrieval", "_summary": "s"} for _ in range(4)]
    assert ag._clear_old_results(msgs) == 2 and msgs[0]["content"].startswith("[cleared")


def test_invalid_args_counted():
    r = asyncio.run(make([("search_papers", {"reason": "x"}), SEARCH, FINISH]).run("q"))
    assert r.steps[0].ok is False and "missing required argument 'query'" in r.steps[0].result
