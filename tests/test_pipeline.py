import numpy as np
import pandas as pd
import pytest

import config
from src import data, pipeline


@pytest.fixture(scope="module")
def sample():
    """A 2000-row sample of the cleaned dataset (keeps tests fast)."""
    df = data.clean(data.load_raw()).sample(2000, random_state=config.RANDOM_STATE)
    X = df[config.FEATURE_COLS + config.SENSITIVE_COLS]
    y = df[config.TARGET_COL]
    return X, y


@pytest.fixture(scope="module")
def fitted(sample):
    X, y = sample
    return pipeline.build_pipeline("logistic_regression").fit(X, y)


def test_clean_blanks_out_of_range_values():
    raw = data.load_raw().head(5).copy()
    raw.loc[0, "person_age"] = 144
    raw.loc[0, "credit_score"] = 1000
    cleaned = data.clean(raw)
    assert pd.isna(cleaned.loc[0, "person_age"])
    assert pd.isna(cleaned.loc[0, "credit_score"])


def test_split_sizes_and_class_balance():
    df = data.clean(data.load_raw())
    X_train, X_test, y_train, y_test = data.split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(len(X_test) / len(df) - config.TEST_SIZE) < 0.01
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_pipeline_outputs_valid_probabilities(sample, fitted):
    X, _ = sample
    proba = fitted.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_sensitive_column_is_not_a_model_input(sample, fitted):
    X, _ = sample
    flipped = X.copy()
    flipped["person_gender"] = flipped["person_gender"].map(
        {"male": "female", "female": "male"}
    )
    assert np.allclose(fitted.predict_proba(X), fitted.predict_proba(flipped))


def test_handles_missing_value_and_unseen_category(sample, fitted):
    X, _ = sample
    row = X.iloc[[0]].copy()
    row["person_age"] = np.nan
    row["loan_intent"] = "SOMETHING_NEW"
    proba = fitted.predict_proba(row)
    assert not np.isnan(proba).any()


def test_unknown_model_name_rejected():
    with pytest.raises(ValueError):
        pipeline.build_pipeline("not_a_model")