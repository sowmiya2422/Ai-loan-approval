import json

import pytest

import config
from src import analytics, auth, db, predict

INPUTS = {
    "person_age": 30,
    "person_income": 60000,
    "person_emp_exp": 5,
    "loan_amnt": 10000,
    "loan_int_rate": 11.0,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 6,
    "credit_score": 690,
    "person_education": "Bachelor",
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "previous_loan_defaults_on_file": "No",
}


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


@pytest.fixture()
def seeded(db_path):
    """Two users: alice has 2 predictions, bob has 1."""
    alice = auth.create_user("alice", "secret123", db_path=db_path)
    bob = auth.create_user("bob", "secret123", db_path=db_path)
    approved = {"probability": 0.9, "decision": "Approved"}
    rejected = {"probability": 0.2, "decision": "Rejected"}
    predict.save_prediction(alice, INPUTS, approved, db_path=db_path)
    predict.save_prediction(alice, INPUTS, rejected, db_path=db_path)
    predict.save_prediction(bob, INPUTS, approved, db_path=db_path)
    return {"db_path": db_path, "alice": alice, "bob": bob}


def test_empty_predictions_have_expected_columns(db_path):
    df = analytics.load_predictions(db_path=db_path)
    assert df.empty
    assert list(df.columns) == analytics.PREDICTION_COLUMNS


def test_all_predictions_newest_first(seeded):
    df = analytics.load_predictions(db_path=seeded["db_path"])
    assert len(df) == 3
    assert df["id"].is_monotonic_decreasing


def test_user_filter_returns_only_that_users_rows(seeded):
    df = analytics.load_predictions(seeded["alice"], db_path=seeded["db_path"])
    assert len(df) == 2
    assert set(df["username"]) == {"alice"}


def test_input_fields_are_parsed_from_json(seeded):
    df = analytics.load_predictions(db_path=seeded["db_path"])
    row = df.iloc[0]
    assert row["loan_amnt"] == INPUTS["loan_amnt"]
    assert row["loan_intent"] == INPUTS["loan_intent"]
    assert row["credit_score"] == INPUTS["credit_score"]


def test_approval_rate_by_loan_intent():
    df = analytics.approval_rate_by("loan_intent")
    assert list(df.columns) == ["loan_intent", "approval_rate", "applications"]
    assert df["approval_rate"].between(0, 1).all()
    assert df["approval_rate"].is_monotonic_decreasing
    assert set(df["loan_intent"]) <= set(config.CATEGORY_OPTIONS["loan_intent"])


def test_load_metrics_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "nope.json")
    assert analytics.load_metrics() is None


def test_load_metrics_reads_json(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"best_model": "xgboost"}))
    monkeypatch.setattr(config, "METRICS_PATH", path)
    assert analytics.load_metrics() == {"best_model": "xgboost"}