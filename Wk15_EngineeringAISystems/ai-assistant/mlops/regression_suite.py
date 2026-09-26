"""W17 Track B: LLM-as-a-judge regression testing with Evidently.

  uv run python -m mlops.regression_suite --config mlops/configs/v3.yaml [--mlflow-run-id <id>]

1. re-runs the fixed regression queries (golden.jsonl) through the candidate prompt/config
2. Evidently TestSuite with two LLM-judge descriptors (Gemini as judge via its OpenAI-compatible API):
     * Correctness  (reference-based): does the new response CONTRADICT the approved reference?
     * Completeness (reference-based): does the new response LOSE key facts present in the reference?
3. tests: share of 'incorrect' <= MAX_INCORRECT, share of 'incomplete' <= MAX_INCOMPLETE
4. saves HTML report, logs pct_tests_passed (+ per-case pass rate) to the version's MLflow run
5. promotion gate: exit code 1 if the suite fails -> the version must not be promoted
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from assistant.agent.loop import ResearchAgent  # noqa: E402
from assistant.config import get_settings  # noqa: E402
from assistant.llm.client import ResilientLLM  # noqa: E402
from assistant.rag.store import VectorStore  # noqa: E402
from assistant.tools import ToolBox  # noqa: E402
from mlops.run_experiment import EXPERIMENT, TRACKING_URI  # noqa: E402

MAX_INCORRECT = 0.10   # at most 1 in 10 regression answers may contradict its reference
MAX_INCOMPLETE = 0.20
RESULTS = ROOT / "mlops" / "results"

CORRECTNESS_CRITERIA = """An ANSWER is INCORRECT if it contradicts the REFERENCE: different numbers, different
method/model names, opposite conclusions, or it claims the corpus lacks information the REFERENCE provides
(or vice versa). Differences in wording, order, formatting or extra correct detail are fine: those are CORRECT.

REFERENCE:
=====
{reference}
====="""

COMPLETENESS_CRITERIA = """An ANSWER is INCOMPLETE if it omits a key fact present in the REFERENCE: a number,
a named method/model/dataset, or one side of a comparison. Missing minor phrasing is fine.
An ANSWER that contains every key fact of the REFERENCE is COMPLETE.

REFERENCE:
=====
{reference}
====="""


def judges(model: str):
    from evidently.descriptors import LLMEval
    from evidently.features.llm_judge import BinaryClassificationPromptTemplate

    def make(name, criteria, target, non_target):
        return LLMEval(
            subcolumn="category",
            additional_columns={"reference": "reference"},
            template=BinaryClassificationPromptTemplate(
                criteria=criteria, target_category=target, non_target_category=non_target,
                uncertainty="unknown", include_reasoning=True,
                pre_messages=[("system", "You are an impartial expert evaluator of research-assistant answers.")],
            ),
            provider="openai", model=model, display_name=name,
        )
    return (make("Correctness", CORRECTNESS_CRITERIA, "incorrect", "correct"),
            make("Completeness", COMPLETENESS_CRITERIA, "incomplete", "complete"))


def run_suite(df: pd.DataFrame, judge_model: str, html_path: Path) -> tuple[dict, pd.DataFrame]:
    from evidently.test_suite import TestSuite
    from evidently.tests import TestCategoryShare

    correctness, completeness = judges(judge_model)
    suite = TestSuite(tests=[
        TestCategoryShare(column_name=correctness.on("response"), category="incorrect", lte=MAX_INCORRECT),
        TestCategoryShare(column_name=completeness.on("response"), category="incomplete", lte=MAX_INCOMPLETE),
    ])
    suite.run(reference_data=None, current_data=df)
    suite.save_html(str(html_path))
    res = suite.as_dict()
    # per-case verdicts (for interpretation / judge sanity check)
    per_case = pd.DataFrame()
    try:
        cur = suite.datasets()[1] if hasattr(suite, "datasets") else None
        if cur is not None:
            per_case = cur
    except Exception as e:  # noqa: BLE001
        print("per-case verdicts not exposed by this evidently version:", e)
    return res, per_case


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mlflow-run-id", default=None, help="log into this run (default: latest run of the version)")
    ap.add_argument("--judge-model", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    s = get_settings()
    s.top_k = cfg["top_k"]
    judge_model = a.judge_model or s.primary_model
    # Evidently's OpenAI provider -> Gemini's OpenAI-compatible endpoint
    os.environ.setdefault("OPENAI_API_KEY", s.gemini_api_key)
    os.environ.setdefault("OPENAI_BASE_URL", s.gemini_base_url)

    golden = [json.loads(l) for l in (ROOT / "mlops/regression/golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not golden:
        raise SystemExit("golden.jsonl is empty: run `python -m mlops.make_golden` and review it first")
    agent = ResearchAgent(ResilientLLM(s), ToolBox(VectorStore(s), None), prompt=cfg["prompt"],
                          max_steps=cfg["max_steps"], temperature=cfg["temperature"], verify_answers=cfg["verify"])
    rows = []
    for g in golden:
        r = await agent.run(g["query"])
        rows.append({"id": g["id"], "question": g["query"], "reference": g["reference"], "response": r.answer,
                     "status": r.status, "reference_status": g["reference_status"]})
        print(g["id"], r.status)
    df = pd.DataFrame(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    html = RESULTS / f"regression_{cfg['version']}.html"
    res, per_case = run_suite(df[["question", "reference", "response"]], judge_model, html)
    tests = res.get("tests", [])
    passed = sum(t.get("status") == "SUCCESS" for t in tests)
    pct = passed / max(1, len(tests))
    summary = {"version": cfg["version"], "tests": [{"name": t.get("name"), "status": t.get("status"),
                                                      "description": t.get("description")} for t in tests],
               "pct_tests_passed": pct}
    if not per_case.empty:
        per_case = pd.concat([df[["id", "status", "reference_status"]].reset_index(drop=True),
                              per_case.reset_index(drop=True)], axis=1)
        verdict_cols = [c for c in per_case.columns if "Correctness" in c or "Completeness" in c]
        per_case.to_csv(RESULTS / f"regression_{cfg['version']}_per_case.csv", index=False)
        cat_cols = [c for c in verdict_cols if "reasoning" not in c.lower()]
        if len(cat_cols) >= 2:
            ok = (per_case[cat_cols[0]].astype(str).str.lower() == "correct") & \
                 (per_case[cat_cols[1]].astype(str).str.lower() == "complete")
            summary["pct_cases_passed"] = float(ok.mean())
            summary["failed_cases"] = per_case.loc[~ok, "id"].tolist()
    (RESULTS / f"regression_{cfg['version']}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    mlflow.set_tracking_uri(TRACKING_URI)
    run_id = a.mlflow_run_id
    if not run_id:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                                  filter_string=f"params.prompt_version = '{cfg['version']}'",
                                  order_by=["attributes.start_time DESC"], max_results=1)
        run_id = runs.iloc[0]["run_id"] if len(runs) else None
    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"regression_{cfg['version']}"):
        mlflow.log_metric("pct_tests_passed", pct)
        if "pct_cases_passed" in summary:
            mlflow.log_metric("pct_regression_cases_passed", summary["pct_cases_passed"])
        mlflow.log_artifact(str(html), "evidently")
        mlflow.log_artifact(str(RESULTS / f"regression_{cfg['version']}.json"), "evidently")
        mlflow.set_tag("regression_gate", "pass" if pct == 1.0 else "fail")
    if pct < 1.0:
        print("REGRESSION: suite failed, do not promote this version")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
