"""Data: download → clean → reference/current split → synthetic drift injection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .config import CATEGORICAL, DATA_URLS, RAW_CSV, SEED, TARGET


def download(force: bool = False) -> None:
    if RAW_CSV.exists() and not force:
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    for url in DATA_URLS:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            RAW_CSV.write_bytes(r.content)
            print(f"downloaded {url}")
            return
        except requests.RequestException as e:
            print(f"failed {url}: {e}")
    raise SystemExit(f"Could not download dataset. Put the Kaggle CSV at {RAW_CSV}")


def load_clean() -> pd.DataFrame:
    download()
    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)          # 11 brand-new customers (tenure 0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype(str)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df.drop(columns=["customerID"])


def reference_current_split(df: pd.DataFrame, ref_frac: float = 0.7, seed: int = SEED):
    """Random 70% = 'training-time' reference, remaining 30% = 'incoming production' current."""
    ref = df.sample(frac=ref_frac, random_state=seed)
    cur = df.drop(ref.index)
    return ref.reset_index(drop=True), cur.reset_index(drop=True)


def inject_drift(cur: pd.DataFrame, seed: int = SEED, m2m_share: float = 0.80, label_flip: float = 0.10):
    """Deliberate, documented perturbations of the current set:
    1. MonthlyCharges: + U(10, 30) per row (price increase)            -> numeric feature drift
    2. tenure: - U(0, 12) months, clipped at 0 (newer customer base)   -> numeric feature drift
    3. Contract: resample so Month-to-month is ~80% (natural ~55%)     -> categorical drift
    4. Churn: flip 10% of labels 0->1                                   -> label / concept drift
    """
    rng = np.random.default_rng(seed)
    d = cur.copy()
    d["MonthlyCharges"] = d["MonthlyCharges"] + rng.uniform(10, 30, len(d))
    d["tenure"] = np.clip(d["tenure"] - rng.integers(0, 13, len(d)), 0, None)
    m2m, other = d[d.Contract == "Month-to-month"], d[d.Contract != "Month-to-month"]
    n = len(d)
    n_m2m = int(n * m2m_share)
    d = pd.concat([m2m.sample(n_m2m, replace=True, random_state=seed),
                   other.sample(n - n_m2m, replace=len(other) < n - n_m2m, random_state=seed)])
    d = d.sample(frac=1, random_state=seed).reset_index(drop=True)
    flip = (rng.random(len(d)) < label_flip) & (d["Churn"] == 0)
    d.loc[flip, "Churn"] = 1
    return d, {"MonthlyCharges": "+U(10,30)", "tenure": "-U(0,12) months",
               "Contract": f"Month-to-month share -> {m2m_share:.0%}", "Churn": f"{label_flip:.0%} of 0 labels flipped to 1"}
