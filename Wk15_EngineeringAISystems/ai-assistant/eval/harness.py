"""W16 evaluation harness (built from scratch, no eval framework).

  uv run python -m eval.harness                       # agent with verifier (multi-agent)
  uv run python -m eval.harness --no-verify           # single-agent baseline (token comparison)
  uv run python -m eval.harness --inject unavailable  # failure-injection run (also: malformed, timeout)
  uv run python -m eval.harness --compare             # runs verify + no-verify + all injections, one report

Metrics per query: task completion, tool-call correctness, trajectory length, tokens, failure class.
Outputs: eval/results/<run>.md (report), <run>.json (raw), traces/<run>/<case>.jsonl (step traces)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.agent.loop import AgentResult, ResearchAgent  # noqa: E402
from assistant.config import get_settings  # noqa: E402
from assistant.llm.client import ResilientLLM  # noqa: E402
from assistant.rag.store import VectorStore  # noqa: E402
from assistant.tools import ToolBox  # noqa: E402

RESULTS = ROOT / "eval" / "results"
AGENT_TOOLS = {"search_papers", "get_chunk", "list_papers", "write_note", "ask_user", "finish"}


@dataclass
class CaseResult:
    case_id: str
    type: str
    mode: str
    status: str
    completed: bool
    checks: dict
    n_steps: int
    reasonable_length: bool
    tool_calls: int
    valid_tool_calls: int
    expected_tools_used: bool
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    verifier_tokens: int
    latency_s: float
    failure_class: str = ""       # "" | hard | soft | cascading_soft
    failure_reason: str = ""
    answer: str = ""
    trajectory: list = field(default_factory=list)


# ------------------------------------------------------------------ scoring
def tool_call_valid(step, catalog) -> bool:
    if step.tool not in AGENT_TOOLS:
        return False
    if isinstance(step.result, str) and step.result.startswith("ERROR: invalid arguments"):
        return False
    if isinstance(step.result, str) and step.result.startswith("ERROR: arguments were not valid JSON"):
        return False
    if step.tool == "search_papers" and step.args.get("paper_id") and step.args["paper_id"] not in catalog:
        return False
    if step.tool == "write_note" and isinstance(step.result, str) and "never retrieved" in step.result:
        return False
    return True


def cited_papers(res: AgentResult) -> set[str]:
    return {c.split(":")[0] for c in res.citations}


def score(case: dict, res: AgentResult, mode: str, inject: str, catalog) -> CaseResult:
    ans = (res.answer or "").lower()
    checks: dict = {}
    if inject:
        # correct behaviour under failure: recognise it, no confident fabricated answer
        checks["status_ok"] = res.status in ("insufficient_evidence", "error", "max_steps", "needs_clarification")
        checks["no_citations"] = len(res.citations) == 0
        checks["acknowledges_failure"] = any(w in ans for w in ("unavailable", "could not", "unable", "error",
                                                                 "insufficient", "malformed", "timed out", "cannot")) \
            or res.status == "error"
    elif case["type"] == "answer":
        checks["status_ok"] = res.status == "answered"
        need = set(case["expected_papers"])
        checks["papers_cited"] = need <= cited_papers(res)
        checks["keywords"] = all(any(k.lower() in ans for k in grp) for grp in case["keywords"])
    elif case["type"] == "clarify":
        checks["status_ok"] = res.status == "needs_clarification"
    elif case["type"] == "refuse":
        checks["status_ok"] = res.status == "insufficient_evidence" or \
            any(w in ans for w in ("not covered", "no information", "does not", "doesn't", "not found", "not discuss"))
        checks["no_fabricated_citations"] = res.status == "insufficient_evidence" or len(res.citations) == 0 or \
            "not" in ans
    completed = all(checks.values())

    steps = res.steps
    valid = sum(tool_call_valid(s, catalog) for s in steps)
    used = {s.tool for s in steps}
    exp_used = set(case["expected_tools"]) <= used if not inject else True

    # ---------- failure taxonomy
    fclass, reason = "", ""
    if not completed:
        hard = res.status in ("error", "budget") and not inject or \
            (res.status == "max_steps" and not inject) or \
            any(s.tool == "(none)" for s in steps[-2:]) and not res.answer
        if inject and not completed:
            # under injection, a confident answer = failure to recognise the fault
            fclass = "cascading_soft" if res.status == "answered" and res.citations else "soft"
            reason = "produced an answer despite injected tool failure" if res.status == "answered" \
                else "failure not acknowledged in answer"
        elif hard:
            fclass, reason = "hard", f"terminated without a usable answer ({res.termination_reason})"
        else:
            bad_early = [s for s in steps[:-1] if not s.ok or
                         (s.tool == "search_papers" and isinstance(s.result, dict) and s.result.get("n", 1) == 0) or
                         (s.tool == "search_papers" and s.args.get("paper_id")
                          and case["expected_papers"] and s.args["paper_id"] not in case["expected_papers"]
                          and len(case["expected_papers"]) == 1)]
            failed_checks = [k for k, v in checks.items() if not v]
            if bad_early:
                fclass = "cascading_soft"
                reason = (f"early step {bad_early[0].step} ({bad_early[0].tool}) went wrong and the error propagated "
                          f"to the final answer; failed checks: {failed_checks}")
            else:
                fclass, reason = "soft", f"finished but failed checks: {failed_checks}"
    return CaseResult(
        case["id"], case["type"], mode, res.status, completed, checks, len(steps),
        len(steps) <= case["max_steps_expected"], len(steps), valid, exp_used, res.total_tokens,
        res.prompt_tokens, res.completion_tokens, res.verifier_tokens, res.latency_s, fclass, reason,
        res.answer[:600], [{"step": s.step, "tool": s.tool, "ok": s.ok} for s in steps])


# ------------------------------------------------------------------ running
async def run_mode(cases, mode: str, verify: bool, inject: str, prompt: str | None, trace_dir: Path,
                   store: VectorStore, llm: ResilientLLM, max_steps: int | None = None,
                   top_k: int | None = None, temperature: float | None = None,
                   clear: bool = True) -> list[CaseResult]:
    tb = ToolBox(store, None, fail_inject=inject)
    agent = ResearchAgent(llm, tb, prompt=prompt, verify_answers=verify, max_steps=max_steps,
                          top_k=top_k, temperature=temperature, clear_tool_results=clear)
    out = []
    for c in cases:
        t = time.time()
        res = await agent.run(c["query"], trace_path=trace_dir / f"{c['id']}.jsonl")
        cr = score(c, res, mode, inject, tb.catalog)
        out.append(cr)
        print(f"[{mode}] {c['id']:<32} {'PASS' if cr.completed else 'FAIL':4} status={res.status:<22} "
              f"steps={cr.n_steps} tokens={cr.total_tokens} ({time.time()-t:.0f}s) {cr.failure_class}")
    return out


def summarize(rs: list[CaseResult]) -> dict:
    n = len(rs) or 1
    tc = sum(r.tool_calls for r in rs) or 1
    return {
        "n": len(rs),
        "task_completion_rate": round(sum(r.completed for r in rs) / n, 3),
        "tool_call_correctness": round(sum(r.valid_tool_calls for r in rs) / tc, 3),
        "expected_tool_usage": round(sum(r.expected_tools_used for r in rs) / n, 3),
        "avg_steps": round(sum(r.n_steps for r in rs) / n, 2),
        "reasonable_length_rate": round(sum(r.reasonable_length for r in rs) / n, 3),
        "avg_tokens": round(sum(r.total_tokens for r in rs) / n),
        "total_tokens": sum(r.total_tokens for r in rs),
        "verifier_tokens": sum(r.verifier_tokens for r in rs),
        "avg_latency_s": round(sum(r.latency_s for r in rs) / n, 1),
        "hard_failures": sum(r.failure_class == "hard" for r in rs),
        "soft_failures": sum(r.failure_class == "soft" for r in rs),
        "cascading_soft_failures": sum(r.failure_class == "cascading_soft" for r in rs),
    }


def report_md(run_name: str, groups: dict[str, list[CaseResult]], cfg: dict) -> str:
    L = [f"# Agent evaluation report: `{run_name}`", "", f"Config: `{json.dumps(cfg)}`", "", "## Summary", "",
         "| mode | n | completion | tool-call correctness | expected tools used | avg steps | reasonable length "
         "| avg tokens | total tokens | verifier tokens | hard | soft | cascading |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, rs in groups.items():
        s = summarize(rs)
        L.append(f"| {m} | {s['n']} | {s['task_completion_rate']:.0%} | {s['tool_call_correctness']:.0%} | "
                 f"{s['expected_tool_usage']:.0%} | {s['avg_steps']} | {s['reasonable_length_rate']:.0%} | "
                 f"{s['avg_tokens']} | {s['total_tokens']} | {s['verifier_tokens']} | {s['hard_failures']} | "
                 f"{s['soft_failures']} | {s['cascading_soft_failures']} |")
    for m, rs in groups.items():
        L += ["", f"## Per-query results: {m}", "",
              "| case | type | pass | status | steps (≤exp) | valid calls | tokens | trajectory |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            traj = " → ".join(f"{t['tool']}{'' if t['ok'] else '✗'}" for t in r.trajectory)
            L.append(f"| {r.case_id} | {r.type} | {'✅' if r.completed else '❌'} | {r.status} | "
                     f"{r.n_steps} ({'ok' if r.reasonable_length else 'long'}) | {r.valid_tool_calls}/{r.tool_calls} | "
                     f"{r.total_tokens} | {traj} |")
    L += ["", "## Failure log", "", "| mode | case | class | reason |", "|---|---|---|---|"]
    for m, rs in groups.items():
        for r in rs:
            if r.failure_class:
                L.append(f"| {m} | {r.case_id} | **{r.failure_class}** | {r.failure_reason} |")
    return "\n".join(L) + "\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--inject", default="", choices=["", "unavailable", "malformed", "timeout"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-clear", action="store_true", help="disable tool-result clearing (context-engineering ablation)")
    a = ap.parse_args()

    spec = yaml.safe_load((ROOT / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    cases, inj_ids = spec["cases"], set(spec.get("injection_cases", []))
    if a.only:
        cases = [c for c in cases if c["id"] in a.only]

    s = get_settings()
    store, llm = VectorStore(s), ResilientLLM(s)
    run_name = a.name or time.strftime("eval_%Y%m%d_%H%M%S")
    tdir = RESULTS / "traces" / run_name
    groups: dict[str, list[CaseResult]] = {}
    if a.compare:
        groups["agent+verifier"] = await run_mode(cases, "agent+verifier", True, "", a.prompt, tdir / "verify", store, llm)
        groups["single-agent baseline"] = await run_mode(cases, "single", False, "", a.prompt, tdir / "single", store, llm)
        icases = [c for c in cases if c["id"] in inj_ids]
        for mode in ("unavailable", "malformed", "timeout"):
            groups[f"inject:{mode}"] = await run_mode(icases, f"inject:{mode}", True, mode, a.prompt,
                                                      tdir / f"inject_{mode}", store, llm)
    else:
        mode = f"inject:{a.inject}" if a.inject else ("single" if a.no_verify else "agent+verifier")
        cs = [c for c in cases if c["id"] in inj_ids] if a.inject else cases
        if a.no_clear:
            mode += "+no-clear"
        groups[mode] = await run_mode(cs, mode, not a.no_verify, a.inject, a.prompt, tdir / mode.replace(":", "_"),
                                      store, llm, clear=not a.no_clear)

    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = ResearchAgent(llm, ToolBox(store, None), prompt=a.prompt).config()
    md = report_md(run_name, groups, cfg)
    (RESULTS / f"{run_name}.md").write_text(md, encoding="utf-8")
    (RESULTS / f"{run_name}.json").write_text(json.dumps(
        {"config": cfg, "summary": {m: summarize(r) for m, r in groups.items()},
         "cases": {m: [r.__dict__ for r in rs] for m, rs in groups.items()}}, indent=1, default=str))
    print(md)
    return groups


if __name__ == "__main__":
    asyncio.run(main())
