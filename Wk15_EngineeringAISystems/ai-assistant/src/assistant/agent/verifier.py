"""Verifier sub-agent: an independent, fresh-context check of a draft answer against the cited evidence.

It sees ONLY the draft answer and the full text of the chunks it cites -- not the agent's
search history or reasoning -- so it cannot be anchored by the agent's own trajectory
(the self-verification paradox).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VERIFIER_PROMPT = """You are a strict fact-checker. You receive a DRAFT ANSWER and the EVIDENCE passages it cites.
For every factual claim in the draft (numbers, method names, datasets, comparisons), check whether the
EVIDENCE explicitly supports it. Claims that rely on background knowledge not in the evidence are UNSUPPORTED.
Return JSON only."""


class Verdict(BaseModel):
    verdict: Literal["supported", "partially_supported", "unsupported"]
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims not backed by the evidence")
    feedback: str = Field(description="One or two sentences telling the writer what to fix or search for")


async def verify(llm, question: str, answer: str, evidence: list[dict]) -> tuple[Verdict, object]:
    ev = "\n\n".join(f"[{e['chunk_id']}] {e['text']}" for e in evidence) or "(no evidence cited)"
    msgs = [{"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": f"QUESTION: {question}\n\nDRAFT ANSWER:\n{answer}\n\nEVIDENCE:\n{ev}"}]
    return await llm.chat_json(msgs, Verdict, temperature=0.0)
