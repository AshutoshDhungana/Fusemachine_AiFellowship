"""Offline checks on synthetic Telco-like data (no download / MLflow server needed)."""
import numpy as np
import pandas as pd

from churn.config import CATEGORICAL, FEATURES, TARGET
from churn.data import inject_drift, reference_current_split


def synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["Yes", "No"], n) for c in CATEGORICAL})
    df["gender"] = rng.choice(["Male", "Female"], n)
    df["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
    df["InternetService"] = rng.choice(["DSL", "Fiber optic", "No"], n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(18, 118, n)
    df["TotalCharges"] = df.tenure * df.MonthlyCharges
    logit = -1 + 1.5 * (df.Contract == "Month-to-month") - 0.04 * df.tenure + 0.01 * df.MonthlyCharges
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_split_and_drift():
    ref, cur = reference_current_split(synthetic())
    assert len(ref) == 1050 and len(cur) == 450
    d, info = inject_drift(cur)
    assert len(d) == len(cur)
    assert abs((d.Contract == "Month-to-month").mean() - 0.8) < 0.01
    assert d.MonthlyCharges.mean() - cur.MonthlyCharges.mean() > 10
    assert d[TARGET].mean() > cur[TARGET].mean()
    assert set(info) == {"MonthlyCharges", "tenure", "Contract", "Churn"}


def test_models_train():
    from churn.train import CONFIGS, build, evaluate
    ref, _ = reference_current_split(synthetic())
    for cfg in CONFIGS:
        m = build(cfg).fit(ref[FEATURES], ref[TARGET])
        metrics, _ = evaluate(m, ref[FEATURES], ref[TARGET])
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"} and metrics["roc_auc"] > 0.6
