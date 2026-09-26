"""Monitoring with Evidently (legacy Report/TestSuite API, evidently<0.7).

  uv run python -m churn.drift [--no-inject] [--retrain-if-drift]

reference = random 70% (training-time data), current = remaining 30% + injected drift (see data.inject_drift).
Produces (results/ and logged to an MLflow run 'drift_check'):
  * data_drift_report.html       DataDriftPreset over all features
  * target_drift_report.html     TargetDriftPreset (Churn rate shift) + prediction drift of the Production model
  * custom_metrics_report.html   custom metrics: Δ mean MonthlyCharges, Δ churn rate in Month-to-month segment
  * drift_tests.html             TestSuite: per-perturbed-column drift tests + share of drifted columns
  * model_performance_current.html  ClassificationPreset of the Production model on the drifted current data
  * drift_summary.json / .md     which columns drifted, whether each injected drift was detected, decision
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns

from .config import CATEGORICAL, EXPERIMENT, FEATURES, MODEL_NAME, NUMERIC, RESULTS, TARGET, TRACKING_URI
from .data import inject_drift, load_clean, reference_current_split

DRIFT_SHARE_ALERT = 0.25        # retrain if > 25% of features drift ...
KEY_FEATURES = ["tenure", "MonthlyCharges", "Contract"]   # ... or any of the top churn drivers drifts


# ------------------------------------------------------------------ custom metrics
def mean_monthly_charges_delta(data) -> float:
    """Custom metric 1: current mean MonthlyCharges - reference mean (USD)."""
    return float(data.current_data["MonthlyCharges"].mean() - data.reference_data["MonthlyCharges"].mean())


def m2m_churn_rate_delta(data) -> float:
    """Custom metric 2: churn-rate shift inside the Month-to-month contract segment (percentage points)."""
    def rate(df):
        seg = df[df["Contract"] == "Month-to-month"]
        return seg[TARGET].mean() if len(seg) else np.nan
    return float((rate(data.current_data) - rate(data.reference_data)) * 100)


def custom_metrics():
    from evidently.metrics.custom_metric import CustomValueMetric
    return [CustomValueMetric(func=mean_monthly_charges_delta, title="Δ mean MonthlyCharges (current − reference, $)"),
            CustomValueMetric(func=m2m_churn_rate_delta, title="Δ churn rate, Month-to-month segment (pp)")]


# ------------------------------------------------------------------ main
def load_production_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    except Exception as e:  # noqa: BLE001
        print("no Production model yet (run churn.train first):", e)
        return None


def run(inject: bool = True) -> dict:
    RESULTS.mkdir(exist_ok=True)
    ref, cur = reference_current_split(load_clean())
    injected = {}
    if inject:
        cur, injected = inject_drift(cur)

    model = load_production_model()
    mapping = ColumnMapping(target=TARGET, numerical_features=NUMERIC, categorical_features=CATEGORICAL)
    if model is not None:
        ref, cur = ref.copy(), cur.copy()
        ref["prediction"] = model.predict_proba(ref[FEATURES])[:, 1]
        cur["prediction"] = model.predict_proba(cur[FEATURES])[:, 1]
        mapping.prediction = "prediction"

    feats = ref[FEATURES + [TARGET]]
    # 1. data drift (features only)
    data_drift = Report(metrics=[DataDriftPreset(columns=FEATURES)])
    data_drift.run(reference_data=ref, current_data=cur, column_mapping=mapping)
    data_drift.save_html(str(RESULTS / "data_drift_report.html"))
    # 2. target (+ prediction) drift
    target_drift = Report(metrics=[TargetDriftPreset()])
    target_drift.run(reference_data=ref, current_data=cur, column_mapping=mapping)
    target_drift.save_html(str(RESULTS / "target_drift_report.html"))
    # 3. custom metrics
    custom_values = {"delta_mean_monthly_charges": mean_monthly_charges_delta(type("D", (), {"current_data": cur, "reference_data": ref})),
                     "delta_m2m_churn_rate_pp": m2m_churn_rate_delta(type("D", (), {"current_data": cur, "reference_data": ref}))}
    try:
        custom = Report(metrics=[*custom_metrics(), ColumnDriftMetric(column_name=TARGET), DatasetDriftMetric()])
        custom.run(reference_data=ref, current_data=cur, column_mapping=mapping)
        custom.save_html(str(RESULTS / "custom_metrics_report.html"))
    except Exception as e:  # noqa: BLE001
        print("CustomValueMetric not available in this evidently version; values computed directly:", e)
    # 4. tests
    tests = TestSuite(tests=[TestShareOfDriftedColumns(lt=DRIFT_SHARE_ALERT),
                             *[TestColumnDrift(column_name=c) for c in ["MonthlyCharges", "tenure", "Contract", TARGET,
                                                                        "gender", "PaymentMethod"]]])
    tests.run(reference_data=feats, current_data=cur[FEATURES + [TARGET]], column_mapping=mapping)
    tests.save_html(str(RESULTS / "drift_tests.html"))
    # 5. model performance on the drifted data
    if model is not None:
        cur_eval = cur.copy()
        perf = Report(metrics=[ClassificationPreset()])
        perf.run(reference_data=ref, current_data=cur_eval,
                 column_mapping=ColumnMapping(target=TARGET, prediction="prediction", numerical_features=NUMERIC,
                                              categorical_features=CATEGORICAL))
        perf.save_html(str(RESULTS / "model_performance_current.html"))

    # ---------------- summarise
    dd = data_drift.as_dict()
    table = next(m["result"] for m in dd["metrics"] if "drift_by_columns" in m["result"])
    by_col = {c: {"drift_detected": v["drift_detected"], "stattest": v["stattest_name"],
                  "score": round(float(v["drift_score"]), 5)} for c, v in table["drift_by_columns"].items()}
    drifted = sorted(c for c, v in by_col.items() if v["drift_detected"])
    td = target_drift.as_dict()
    target_res = next((m["result"] for m in td["metrics"] if m["result"].get("column_name") == TARGET), {})
    share = table["share_of_drifted_columns"]
    retrain = share > DRIFT_SHARE_ALERT or any(c in drifted for c in KEY_FEATURES) or target_res.get("drift_detected", False)
    summary = {
        "injected": injected,
        "detected_injected": {c: by_col.get(c, {}).get("drift_detected", target_res.get("drift_detected"))
                              if c != TARGET else target_res.get("drift_detected") for c in injected},
        "drifted_columns": drifted,
        "share_of_drifted_columns": round(share, 3),
        "target_drift": {"detected": target_res.get("drift_detected"), "stattest": target_res.get("stattest_name"),
                         "score": target_res.get("drift_score"),
                         "churn_rate_reference": round(float(ref[TARGET].mean()), 4),
                         "churn_rate_current": round(float(cur[TARGET].mean()), 4)},
        "custom_metrics": {k: round(v, 3) for k, v in custom_values.items()},
        "tests": [{"name": t["name"], "status": t["status"]} for t in tests.as_dict()["tests"]],
        "by_column": by_col,
        "decision": "RETRAIN recommended" if retrain else "no action",
    }
    if model is not None:
        from sklearn.metrics import f1_score, roc_auc_score
        for name, d in (("reference", ref), ("current", cur)):
            summary[f"model_{name}"] = {"f1": round(f1_score(d[TARGET], d["prediction"] >= 0.5), 4),
                                        "roc_auc": round(roc_auc_score(d[TARGET], d["prediction"]), 4)}
    (RESULTS / "drift_summary.json").write_text(json.dumps(summary, indent=1))
    write_md(summary)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="drift_check"):
        mlflow.set_tag("type", "monitoring")
        mlflow.log_params({"inject_drift": inject, "reference_rows": len(ref), "current_rows": len(cur)})
        mlflow.log_metrics({"share_of_drifted_columns": share, "n_drifted_columns": len(drifted),
                            "target_drift_detected": float(bool(summary["target_drift"]["detected"])),
                            **summary["custom_metrics"],
                            **({f"current_{k}": v for k, v in summary.get("model_current", {}).items()})})
        for f in RESULTS.glob("*.html"):
            mlflow.log_artifact(str(f), "evidently")
        mlflow.log_artifact(str(RESULTS / "drift_summary.json"), "evidently")
        mlflow.log_artifact(str(RESULTS / "drift_summary.md"), "evidently")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_column"}, indent=1))
    return summary


def write_md(s: dict):
    L = ["# Drift check summary", "", f"**Decision:** {s['decision']}", "",
         f"Share of drifted features: **{s['share_of_drifted_columns']:.0%}** (alert > {DRIFT_SHARE_ALERT:.0%})", "",
         "| injected perturbation | column | detected? |", "|---|---|---|"]
    for c, how in s["injected"].items():
        L.append(f"| {how} | {c} | {'✅' if s['detected_injected'].get(c) else '❌'} |")
    L += ["", f"Drifted columns: {', '.join(s['drifted_columns']) or 'none'}", "",
          f"Churn rate reference → current: {s['target_drift']['churn_rate_reference']:.1%} → "
          f"{s['target_drift']['churn_rate_current']:.1%} (drift detected: {s['target_drift']['detected']})", "",
          "Custom metrics: " + ", ".join(f"{k} = {v}" for k, v in s["custom_metrics"].items())]
    if "model_current" in s:
        L += ["", f"Production model F1 reference/current: {s['model_reference']['f1']} → {s['model_current']['f1']}; "
                  f"ROC-AUC {s['model_reference']['roc_auc']} → {s['model_current']['roc_auc']}"]
    (RESULTS / "drift_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--retrain-if-drift", action="store_true")
    a = ap.parse_args()
    s = run(inject=not a.no_inject)
    if s["decision"].startswith("RETRAIN") and a.retrain_if_drift:
        print("drift threshold crossed -> retraining")
        subprocess.run([sys.executable, "-m", "churn.retrain"], check=True)


if __name__ == "__main__":
    main()
