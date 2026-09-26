# Telco Churn: MLOps pipeline (Fusemachines AI Fellowship W17, Track A)

A predictive churn model brought to production MLOps standards: a reproducible **uv** environment, **MLflow** tracking of 4 models, a registry with Staging → Production transitions, a serving API, **Evidently** data/target drift monitoring with custom metrics, and an **Airflow** retraining DAG.

```
data ──► training (4 configs) ──► MLflow tracking ──► registry (Staging → Production)
                                                            │
                           serving (FastAPI / mlflow serve) ◄┘
                                                            │
monitoring: Evidently (reference 70% vs drifted current 30%) ─► drift? ──► retrain challenger ──► promote if better
```

## Quick start

```bash
git clone <repo> && cd churn-mlops
uv sync                           # exact environment from uv.lock
uv run churn-pipeline             # download → train 4 models → register best → Production → drift check → retrain if needed
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
uv run uvicorn churn.serve:app --port 8000                   # serve the Production model
```

The individual steps are also available as commands:

```bash
uv run python -m churn.train      # tracking + registry
uv run python -m churn.drift      # Evidently reports → results/*.html + MLflow run "drift_check"
uv run python -m churn.retrain    # challenger vs champion on new labelled data
uv run pytest -q                  # offline tests on synthetic data
```

Dataset: IBM Telco Customer Churn (same file as Kaggle `blastchar/telco-customer-churn`): 7,043 customers, 19 features, target `Churn` (26.5% Yes). `churn/data.py` downloads it automatically. If the download is blocked, put the Kaggle CSV in `data/raw/`.

Example request:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '[{"tenure": 2, "Contract": "Month-to-month", "MonthlyCharges": 95.5, "InternetService": "Fiber optic"}]'
```

Alternative serving path: `uv run mlflow models serve -m "models:/telco-churn-classifier/Production" -p 5001 --env-manager local`.

---

## a. Environment & reproducibility (uv)

`pyproject.toml` + committed `uv.lock`. What this fixes for *this* project:

1. **MLflow, Evidently and scikit-learn share heavy transitive dependencies** (`numpy`, `scipy`, `pandas`, `pydantic`, `pyarrow`). The lock resolves them together once for Windows, Linux and macOS, instead of whichever versions a given `pip install` order happens to produce.
2. **Evidently's API changed in 0.7:** the legacy `Report`/`TestSuite`/`TargetDriftPreset` API used here moved. The `<0.7` bound means a fresh clone never picks up the breaking version.
3. **Registry stages:** `mlflow<3` keeps `transition_model_version_stage` (stages) working, alongside the alias `champion`.

`uv sync` from a clean clone produces an identical `.venv`, and `uv run …` executes inside it. There are no manual `pip install` steps.

## b. Experiment tracking strategy (MLflow)

**What varied.** Hyperparameters *and* model families, all on the same 80/20 stratified split of the reference data, seed 42:

| run | family | hyperparameters |
|---|---|---|
| `logreg_C1_l2` | Logistic regression | C=1.0, L2, no class weighting (baseline) |
| `logreg_C0.1_l1_balanced` | Logistic regression | C=0.1, L1 (sparse), class_weight=balanced |
| `rf_300_d8_balanced` | Random forest | 300 trees, max_depth=8, min_samples_leaf=5, balanced |
| `hgb_lr0.05_d4_balanced` | HistGradientBoosting | lr=0.05, max_depth=4, 300 iters, L2=1.0, balanced |

**Logged per run:** all hyperparameters; accuracy, precision, recall, F1, ROC-AUC and fit time; artifacts: the model (with signature and input example), the confusion matrix and the ROC curve.

**Selection rule.** The model with the highest **F1**, with ROC-AUC as tie-break, is registered. Churn is imbalanced (≈26.5% positives), so a model that predicts "stay" for everyone already scores about 73% accuracy. Accuracy therefore rewards the unweighted baseline for ignoring churners, while F1 and recall measure what the business needs: finding the customers who will leave.

**Run comparison** (auto-generated in `results/run_comparison_initial.md` after `churn-pipeline`; paste the table here):

| run | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| _run the pipeline to fill in the measured values_ | | | | | |

**Justification** (fill in with the numbers from the table): *e.g. "`<run>` was registered because it has the highest F1 (x.xx) and ROC-AUC (x.xx). `logreg_C1_l2` has slightly higher accuracy (x.xx) but a recall of only x.xx: it misses roughly half of the churners, and accuracy is misleading on this imbalanced target."*

**Registry.** `telco-churn-classifier` goes None → **Staging** → **Production** (previous versions archived), and the alias `champion` is set. The history is written to `results/registered_model.json`. Take the screenshots for submission from the MLflow UI: the Experiments → compare view for all 4 runs, and the Models page.

## c. Monitoring & drift strategy (Evidently)

* **Reference:** a random 70% of the dataset, which is exactly the data the model was trained and validated on.
* **Current:** the remaining 30%, treated as incoming production data, with **deliberate drift** (`churn/data.py::inject_drift`):

| perturbation | column | expected effect |
|---|---|---|
| +U(10, 30) $ per row (price rise) | `MonthlyCharges` | numeric drift |
| −U(0, 12) months (newer customer base) | `tenure` | numeric drift |
| Month-to-month resampled to 80% (natural ≈55%) | `Contract` | categorical drift |
| 10% of `No` labels flipped to `Yes` | `Churn` | label / concept drift |

**Reports** (`results/`, and logged to the MLflow run `drift_check` under `evidently/`):

* `data_drift_report.html`: DataDriftPreset over all 19 features.
* `target_drift_report.html`: TargetDriftPreset on `Churn`, plus prediction drift of the Production model.
* `custom_metrics_report.html`: two **custom metrics** built with `CustomValueMetric`:
  * Δ mean `MonthlyCharges` (current − reference, in $)
  * Δ churn rate inside the Month-to-month segment (percentage points)
* `drift_tests.html`: TestSuite with share of drifted columns < 25%, plus per-column drift tests for the perturbed columns and two untouched controls (`gender`, `PaymentMethod`).
* `model_performance_current.html`: the Production model's classification quality on the drifted data.
* `drift_summary.md` / `.json`: the machine-readable verdict: which injected columns were detected, the drift share, the custom metric values, reference vs current F1 and the decision.

**Interpretation** (update with `results/drift_summary.md`):

* All three injected feature drifts should be detected, with near-zero p-values or large Wasserstein/Jensen-Shannon distances.
* `Churn` should show target drift: the rate rises from ≈26% to ≈34% because of the label flips and because Month-to-month customers churn more.
* The untouched controls should not drift. If they do, that is a false positive of the chosen statistical test.

For the model, this means that tenure, contract type and monthly charges, which are its strongest churn signals, have moved. The model was calibrated on a population with longer tenure and cheaper plans, so its probability estimates are now off, as the F1 and ROC-AUC change on the current data shows. The label flips mean that P(churn | features) itself changed. That is concept drift, which re-weighting cannot fix; the model needs fresh labels.

**Action policy.** A retrain is triggered when the share of drifted features is > 25%, *or* any key feature (`tenure`, `MonthlyCharges`, `Contract`) drifts, *or* target drift is detected. `churn.retrain` trains a challenger on reference data plus newly labelled current data. The challenger replaces Production only if it beats the champion's F1 on a holdout of the new data. Otherwise it stays in Staging.

## d. Orchestration (Airflow, bonus)

`dags/churn_drift_dag.py` (`telco_churn_weekly_drift_check`) runs **weekly (Mondays 06:00)**:

1. `evidently_drift_check` runs `churn.drift` and logs the result to MLflow.
2. `drift_significant` is a branch task that reads `results/drift_summary.json`.
3. On a positive drift decision the `retrain` task runs (challenger vs champion, then promote or keep in Staging). Otherwise the DAG goes to `no_drift`.

To run it locally: `pip install apache-airflow` in a separate venv, set `AIRFLOW__CORE__DAGS_FOLDER=<repo>/dags`, set the Airflow Variable `churn_repo`, then run `airflow standalone`.

## Repository layout

```
src/churn/  config.py · data.py (download, split, drift injection) · train.py (MLflow + registry)
            serve.py (FastAPI) · drift.py (Evidently) · retrain.py · pipeline.py
dags/churn_drift_dag.py · tests/ · results/ (reports, comparison tables, plots) · pyproject.toml · uv.lock
```
