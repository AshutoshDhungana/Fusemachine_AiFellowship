"""W17 Track B: track agent prompt/config versions with MLflow.

  uv run python -m mlops.run_experiment --config mlops/configs/v1.yaml
  uv run python -m mlops.run_experiment --all            # v1, v2, v3 in sequence
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare

Per run we log
  params   : prompt version + sha, top_k, temperature, max_steps, verifier, model, chunking
  metrics  : W16 harness metrics (completion, tool-call correctness, steps, tokens, failure counts),
             injection robustness, est. cost; (regression_suite later adds pct_tests_passed to the same run)
  artifacts: prompt text, harness report (md/json), per-case table, 2-3 representative traces
             (clean success, failure, injected-failure) as JSONL: one record per step {step, tool, args,
             result, reasoning} + summary record (termination reason, n_steps, tokens)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from assistant.config import get_settings  # noqa: E402
from assistant.llm.client import ResilientLLM  # noqa: E402
from assistant.rag.store import VectorStore  # noqa: E402
from eval.harness import report_md, run_mode, summarize  # noqa: E402

EXPERIMENT = "paperpilot-agent-prompts"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
# Gemini 2.5 Flash list price (USD / 1M tokens), used only for a comparable cost estimate
PRICE_IN, PRICE_OUT = 0.30, 2.50


def pick_traces(results, trace_dir: Path) -> list[Path]:
    ok = [r for r in results if r.completed and r.type == "answer"]
    bad = [r for r in results if not r.completed]
    chosen = []
    if ok:
        chosen.append(("success", max(ok, key=lambda r: r.n_steps)))   # most informative clean success
    for r in bad[:2]:
        chosen.append((f"failure_{r.failure_class}", r))
    out = []
    for label, r in chosen:
        p = trace_dir / f"{r.case_id}.jsonl"
        if p.exists():
            out.append((label, p))
    return out


async def run_config(cfg_path: Path, include_injection: bool = True) -> str:
    cfg = yaml.safe_load(cfg_path.read_text())
    s = get_settings()
    s.primary_model = cfg.get("primary_model", s.primary_model)
    s.top_k = cfg["top_k"]
    spec = yaml.safe_load((ROOT / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    cases, inj = spec["cases"], set(spec["injection_cases"])
    prompt_text = (ROOT / "prompts" / f"{cfg['prompt']}.md").read_text(encoding="utf-8")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    store, llm = VectorStore(s), ResilientLLM(s)
    with mlflow.start_run(run_name=f"prompt_{cfg['version']}") as run, tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mlflow.set_tags({"prompt_version": cfg["version"], "stage": "candidate", "component": "research-agent"})
        mlflow.log_params({
            "prompt_version": cfg["version"], "prompt_file": cfg["prompt"],
            "prompt_sha": hashlib.sha256(prompt_text.encode()).hexdigest()[:12],
            "top_k": cfg["top_k"], "temperature": cfg["temperature"], "max_steps": cfg["max_steps"],
            "verifier": cfg["verify"], "primary_model": s.primary_model, "fallback_model": s.fallback_model,
            "embed_model": s.embed_model, "chunk_words": s.chunk_words, "retrieval": "hybrid-rrf",
            "n_cases": len(cases),
        })
        kw = dict(prompt=cfg["prompt"], store=store, llm=llm, max_steps=cfg["max_steps"],
                  top_k=cfg["top_k"], temperature=cfg["temperature"])
        groups = {"main": await run_mode(cases, "main", cfg["verify"], "", trace_dir=td / "main", **kw)}
        if include_injection:
            icases = [c for c in cases if c["id"] in inj]
            groups["inject_unavailable"] = await run_mode(icases, "inject:unavailable", cfg["verify"], "unavailable",
                                                          trace_dir=td / "inject", **kw)
        main = summarize(groups["main"])
        metrics = {k: v for k, v in main.items() if isinstance(v, (int, float))}
        metrics["est_cost_usd"] = sum(r.prompt_tokens * PRICE_IN + r.completion_tokens * PRICE_OUT
                                      for r in groups["main"]) / 1e6
        for t in ("answer", "clarify", "refuse"):
            sub = [r for r in groups["main"] if r.type == t]
            if sub:
                metrics[f"completion_{t}"] = sum(r.completed for r in sub) / len(sub)
        if "inject_unavailable" in groups:
            metrics["injection_handled_rate"] = summarize(groups["inject_unavailable"])["task_completion_rate"]
        mlflow.log_metrics(metrics)

        # artifacts
        (td / "prompt.md").write_text(prompt_text, encoding="utf-8")
        mlflow.log_artifact(str(td / "prompt.md"), "prompt")
        mlflow.log_artifact(str(cfg_path), "config")
        (td / "report.md").write_text(report_md(f"prompt_{cfg['version']}", groups, cfg), encoding="utf-8")
        mlflow.log_artifact(str(td / "report.md"), "harness")
        rows = [{k: v for k, v in r.__dict__.items() if k != "trajectory"} | {"group": g}
                for g, rs in groups.items() for r in rs]
        mlflow.log_table(pd.DataFrame(rows).astype(str), "harness/per_case.json")
        for label, p in pick_traces(groups["main"], td / "main"):
            dst = td / f"trace_{label}_{p.stem}.jsonl"
            shutil.copy(p, dst)
            mlflow.log_artifact(str(dst), "traces")
        for p in list((td / "inject").glob("*.jsonl"))[:1]:
            dst = td / f"trace_injected_unavailable_{p.stem}.jsonl"
            shutil.copy(p, dst)
            mlflow.log_artifact(str(dst), "traces")
        # keep a local copy of all traces for diagnosis
        local = ROOT / "mlops" / "results" / "traces" / cfg["version"]
        shutil.rmtree(local, ignore_errors=True)
        shutil.copytree(td / "main", local)
        print(json.dumps(metrics, indent=1))
        return run.info.run_id


def export_comparison(out: Path = ROOT / "mlops" / "results" / "run_comparison.md"):
    mlflow.set_tracking_uri(TRACKING_URI)
    df = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["attributes.start_time ASC"])
    cols = ["tags.mlflow.runName", "params.prompt_version", "params.top_k", "params.temperature", "params.max_steps",
            "metrics.task_completion_rate", "metrics.completion_answer", "metrics.completion_clarify",
            "metrics.completion_refuse", "metrics.injection_handled_rate", "metrics.tool_call_correctness",
            "metrics.avg_steps", "metrics.avg_tokens", "metrics.est_cost_usd", "metrics.pct_tests_passed",
            "metrics.hard_failures", "metrics.soft_failures", "metrics.cascading_soft_failures", "run_id"]
    df = df[[c for c in cols if c in df.columns]]
    df.columns = [c.split(".", 1)[-1] for c in df.columns]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# MLflow run comparison (experiment `%s`)\n\n%s\n" % (EXPERIMENT, df.to_markdown(index=False)),
                   encoding="utf-8")
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(out.read_text())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-injection", action="store_true")
    ap.add_argument("--export", action="store_true", help="only export the comparison table")
    a = ap.parse_args()
    if not a.export:
        cfgs = sorted((ROOT / "mlops" / "configs").glob("v*.yaml")) if a.all else [a.config]
        for c in cfgs:
            print(f"=== running {c.name}")
            await run_config(c, not a.no_injection)
    export_comparison()


if __name__ == "__main__":
    asyncio.run(main())
