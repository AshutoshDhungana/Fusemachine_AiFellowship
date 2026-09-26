"""Retraining step (triggered by drift): train on reference + newly labelled current data, register a new
version, and promote it only if it beats the current Production model on a common holdout."""
from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, FEATURES, MODEL_NAME, SEED, TARGET, TRACKING_URI
from .data import inject_drift, load_clean, reference_current_split
from .train import register_best, train_all


def main():
    ref, cur = reference_current_split(load_clean())
    cur, _ = inject_drift(cur)                     # the "production" data that arrived, now labelled
    cur_train, cur_hold = train_test_split(cur, test_size=0.3, stratify=cur[TARGET], random_state=SEED)
    combined = pd.concat([ref, cur_train], ignore_index=True)
    combined.to_csv(DATA_DIR / "retrain_data.csv", index=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    old = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    old_f1 = f1_score(cur_hold[TARGET], old.predict(cur_hold[FEATURES]))
    table = train_all(combined, tag="retrain")
    best = mlflow.sklearn.load_model(f"runs:/{table.iloc[0].run_id}/model")
    new_f1 = f1_score(cur_hold[TARGET], best.predict(cur_hold[FEATURES]))
    print(f"holdout of new data: production F1={old_f1:.4f}  challenger F1={new_f1:.4f}")
    if new_f1 > old_f1:
        print(register_best(table))
    else:
        mv = mlflow.register_model(f"runs:/{table.iloc[0].run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(MODEL_NAME, mv.version, "Staging")
        print(f"challenger v{mv.version} kept in Staging (did not beat Production)")


if __name__ == "__main__":
    main()
