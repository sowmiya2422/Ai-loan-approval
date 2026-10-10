"""Central configuration: paths, column names and constants used across the project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Kaggle "Loan Approval Classification Dataset" (taweilo), saved as data/loan_data.csv
DATA_PATH = DATA_DIR / "loan_data.csv"
DB_PATH = DATA_DIR / "app.db"

PIPELINE_PATH = MODELS_DIR / "best_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
ANOMALY_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "loan_status"  # 1 = approved, 0 = rejected

# Protected attribute: used ONLY for fairness auditing, never as a model input.
SENSITIVE_COLS = ["person_gender"]

NUMERIC_COLS = [
    "person_age", "person_income", "person_emp_exp", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score",
]
CATEGORICAL_COLS = [
    "person_education", "person_home_ownership", "loan_intent",
    "previous_loan_defaults_on_file",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

REQUIRED_COLUMNS = FEATURE_COLS + SENSITIVE_COLS + [TARGET_COL]

# Allowed values for the prediction form (must match the training data exactly).
CATEGORY_OPTIONS = {
    "person_education": ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                    "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
    "previous_loan_defaults_on_file": ["No", "Yes"],
    "person_gender": ["female", "male"],
}

# Cleaning rules: values outside these ranges are treated as data-entry errors
# (set to missing, then imputed inside the pipeline using training data only).
VALID_RANGES = {
    "person_age": (18, 100),
    "person_emp_exp": (0, 60),
    "credit_score": (300, 850),
}

DISCLAIMER = (
    "This is a decision-support estimate produced by a machine learning model. "
    "It is not an actual bank decision."
)