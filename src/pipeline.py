"""Preprocessing + model pipelines. Everything is fitted on training data only."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import config


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones.

    Only config.FEATURE_COLS are used. Sensitive columns (e.g. gender) are
    dropped here, so the model never sees them.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_COLS),
            ("cat", categorical, config.CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def get_models() -> dict:
    """Candidate models to compare during training."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }


def build_pipeline(model_name: str) -> Pipeline:
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(models)}.")
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", models[model_name]),
    ])