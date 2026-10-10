import pytest

import config
from src import auth, data, db, explain, pipeline, predict

SAMPLE_INPUT = {
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


@pytest.fixture(scope="module")
def pipe():
    """A small XGBoost pipeline trained on 2000 rows (keeps tests fast)."""
    df = data.clean(data.load_raw()).sample(2000, random_state=config.RANDOM_STATE)
    X = df[config.FEATURE_COLS + config.SENSITIVE_COLS]
    y = df[config.TARGET_COL]
    return pipeline.build_pipeline("xgboost").fit(X, y)


def test_predict_returns_valid_result(pipe):
    result = predict.predict(SAMPLE_INPUT, pipe=pipe)
    assert 0.0 <= result["probability"] <= 1.0
    expected = "Approved" if result["probability"] >= predict.APPROVAL_THRESHOLD else "Rejected"
    assert result["decision"] == expected


def test_missing_input_rejected(pipe):
    incomplete = {k: v for k, v in SAMPLE_INPUT.items() if k != "credit_score"}
    with pytest.raises(ValueError):
        predict.predict(incomplete, pipe=pipe)


def test_save_prediction_roundtrip(tmp_path, pipe):
    path = tmp_path / "test.db"
    db.init_db(path)
    user_id = auth.create_user("alice", "secret123", db_path=path)
    result = predict.predict(SAMPLE_INPUT, pipe=pipe)
    row_id = predict.save_prediction(user_id, SAMPLE_INPUT, result, db_path=path)
    with db.get_connection(path) as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (row_id,)).fetchone()
    assert row["user_id"] == user_id
    assert row["decision"] == result["decision"]
    assert abs(row["probability"] - result["probability"]) < 1e-9


def test_explain_returns_top_factors(pipe):
    out = explain.explain(SAMPLE_INPUT, pipe=pipe, top_n=5)
    assert len(out) == 5
    assert list(out.columns) == ["feature", "value", "impact"]
    assert set(out["feature"]) <= set(config.FEATURE_COLS)
    # Sorted by strength of impact, strongest first.
    assert out["impact"].abs().is_monotonic_decreasing