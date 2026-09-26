"""Thin FastAPI wrapper that loads the Production model from the MLflow registry.

  uv run uvicorn churn.serve:app --port 8000
  # alternative: uv run mlflow models serve -m "models:/telco-churn-classifier/Production" -p 5001 --env-manager local
"""
from __future__ import annotations

from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from .config import FEATURES, MODEL_NAME, TRACKING_URI

YN = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: YN = "No"
    Partner: YN = "No"
    Dependents: YN = "No"
    tenure: int = Field(12, ge=0, le=100)
    PhoneService: YN = "Yes"
    MultipleLines: str = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YN = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(70.0, ge=0)
    TotalCharges: float = Field(840.0, ge=0)


app = FastAPI(title="Telco churn model")
STATE: dict = {}


@app.on_event("startup")
def load():
    mlflow.set_tracking_uri(TRACKING_URI)
    STATE["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    v = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    STATE["version"] = v.version


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "stage": "Production", "version": STATE.get("version")}


@app.post("/predict")
def predict(customers: list[Customer], threshold: float = 0.5):
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    p = STATE["model"].predict_proba(df)[:, 1]
    return {"model_version": STATE["version"],
            "predictions": [{"churn_probability": round(float(x), 4), "churn": bool(x >= threshold)} for x in p]}
