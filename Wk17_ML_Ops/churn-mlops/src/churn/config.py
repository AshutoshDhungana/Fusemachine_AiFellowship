from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS = ROOT / "results"
DATA_URLS = [  # IBM sample dataset (same file as Kaggle blastchar/telco-customer-churn)
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
SEED = 42
TARGET = "Churn"
MODEL_NAME = "telco-churn-classifier"
EXPERIMENT = "telco-churn"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
               "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
