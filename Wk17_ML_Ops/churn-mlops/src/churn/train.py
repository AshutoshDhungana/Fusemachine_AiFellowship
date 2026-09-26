"""Training with MLflow tracking + model registry.

  uv run python -m churn.train            # trains 4 configs, registers the best, Staging -> Production
Selection metric: F1 on the held-out test split (target is imbalanced, ~26.5% churn), ROC-AUC tie-break.
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from .config import (CATEGORICAL, EXPERIMENT, FEATURES, MODEL_NAME, NUMERIC, RESULTS, SEED,  # noqa: E402
                     TARGET, TRACKING_URI)
from .data import load_clean, reference_current_split  # noqa: E402

CONFIGS = [
    {"name": "logreg_C1_l2", "family": "logistic_regression",
     "params": {"C": 1.0, "penalty": "l2", "class_weight": None, "solver": "lbfgs", "max_iter": 2000}},
    {"name": "logreg_C0.1_l1_balanced", "family": "logistic_regression",
     "params": {"C": 0.1, "penalty": "l1", "class_weight": "balanced", "solver": "liblinear", "max_iter": 2000}},
    {"name": "rf_300_d8_balanced", "family": "random_forest",
     "params": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5, "class_weight": "balanced"}},
    {"name": "hgb_lr0.05_d4_balanced", "family": "hist_gradient_boosting",
     "params": {"learning_rate": 0.05, "max_depth": 4, "max_iter": 300, "class_weight": "balanced",
                "l2_regularization": 1.0}},
]


def build(cfg: dict) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    fam, p = cfg["family"], cfg["params"]
    if fam == "logistic_regression":
        clf = LogisticRegression(random_state=SEED, **p)
    elif fam == "random_forest":
        clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **p)
    else:
        clf = HistGradientBoostingClassifier(random_state=SEED, **p)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(model, X, y) -> tuple[dict, pd.Series]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}, pred


def plots(model, X, y, pred, name, outdir) -> list:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["stay", "churn"], ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {name}")
    cm = outdir / f"confusion_matrix_{name}.png"
    fig.tight_layout(); fig.savefig(cm, dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    roc = outdir / f"roc_curve_{name}.png"
    fig.tight_layout(); fig.savefig(roc, dpi=120); plt.close(fig)
    return [cm, roc]


def train_all(ref: pd.DataFrame, tag: str = "initial") -> pd.DataFrame:
    """Train every config on the reference data (80/20 stratified train/test inside it)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    X, y = ref[FEATURES], ref[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows = []
    for cfg in CONFIGS:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.set_tags({"model_family": cfg["family"], "training_round": tag})
            mlflow.log_params({"model_family": cfg["family"], **cfg["params"], "train_rows": len(Xtr),
                               "test_rows": len(Xte), "churn_rate_train": round(ytr.mean(), 4),
                               "threshold": 0.5, "seed": SEED})
            model = build(cfg)
            t0 = time.time()
            model.fit(Xtr, ytr)
            fit_s = time.time() - t0
            m, pred = evaluate(model, Xte, yte)
            mlflow.log_metrics({**m, "fit_seconds": fit_s})
            for p in plots(model, Xte, yte, pred, cfg["name"], RESULTS / "plots"):
                mlflow.log_artifact(str(p), "plots")
            mlflow.sklearn.log_model(model, "model", signature=infer_signature(Xte, model.predict_proba(Xte)[:, 1]),
                                     input_example=Xte.head(3))
            rows.append({"run": cfg["name"], "run_id": run.info.run_id, "family": cfg["family"],
                         **{k: round(v, 4) for k, v in m.items()}, "fit_s": round(fit_s, 2)})
            print(rows[-1])
    table = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"run_comparison_{tag}.csv", index=False)
    (RESULTS / f"run_comparison_{tag}.md").write_text(
        f"# MLflow run comparison ({tag}), held-out test split, threshold 0.5\n\n"
        + table.drop(columns=["run_id"]).to_markdown(index=False) + "\n", encoding="utf-8")
    return table


def register_best(table: pd.DataFrame) -> dict:
    """Register the best run (F1, then ROC-AUC) and move it None -> Staging -> Production."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    best = table.iloc[0]
    mv = mlflow.register_model(f"runs:/{best.run_id}/model", MODEL_NAME)
    client.update_model_version(MODEL_NAME, mv.version,
                                description=f"{best.run}: F1={best.f1}, ROC-AUC={best.roc_auc}, recall={best.recall}")
    history = []
    for stage in ("Staging", "Production"):
        client.transition_model_version_stage(MODEL_NAME, mv.version, stage, archive_existing_versions=True)
        history.append({"version": mv.version, "stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"{MODEL_NAME} v{mv.version} -> {stage}")
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)  # MLflow>=2.9 alias (forward-compatible)
    info = {"model_name": MODEL_NAME, "version": mv.version, "run": best.run, "run_id": best.run_id,
            "metrics": {k: float(best[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "stage_history": history,
            "all_versions": [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                             for v in client.search_model_versions(f"name='{MODEL_NAME}'")]}
    (RESULTS / "registered_model.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="initial")
    ap.add_argument("--data", default=None, help="optional CSV to train on (e.g. reference+new data for retraining)")
    a = ap.parse_args()
    if a.data:
        ref = pd.read_csv(a.data)
    else:
        ref, _ = reference_current_split(load_clean())
    table = train_all(ref, a.tag)
    print(table.to_string(index=False))
    print(json.dumps(register_best(table), indent=1))


if __name__ == "__main__":
    main()
