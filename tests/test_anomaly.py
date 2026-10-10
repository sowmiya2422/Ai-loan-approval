import pytest

import config
from src import anomaly, data

NORMAL = {
    "person_age": 30,
    "person_income": 90000,
    "person_emp_exp": 5,
    "loan_amnt": 15000,
    "loan_int_rate": 11.0,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 6,
    "credit_score": 720,
    "person_education": "Bachelor",
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "previous_loan_defaults_on_file": "No",
}

EXTREME = {
    "person_age": 99,
    "person_income": 1000,
    "person_emp_exp": 60,
    "loan_amnt": 5_000_000,
    "loan_int_rate": 25.0,
    "loan_percent_income": 5000.0,
    "cb_person_cred_hist_length": 50,
    "credit_score": 300,
    "person_education": "High School",
    "person_home_ownership": "OTHER",
    "loan_intent": "MEDICAL",
    "previous_loan_defaults_on_file": "Yes",
}


@pytest.fixture(scope="module")
def detector():
    """A detector trained on a 3000-row sample (keeps the test fast)."""
    df = data.clean(data.load_raw()).sample(3000, random_state=config.RANDOM_STATE)
    X = df[config.FEATURE_COLS + config.SENSITIVE_COLS]
    return anomaly.build_detector().fit(X)


def test_result_has_expected_keys(detector):
    out = anomaly.check(NORMAL, detector)
    assert set(out) == {"available", "is_unusual", "score"}
    assert out["available"] is True


def test_normal_application_not_flagged(detector):
    assert anomaly.check(NORMAL, detector)["is_unusual"] is False


def test_extreme_application_flagged_with_lower_score(detector):
    normal = anomaly.check(NORMAL, detector)
    extreme = anomaly.check(EXTREME, detector)
    assert extreme["is_unusual"] is True
    assert extreme["score"] < normal["score"]


def test_missing_detector_flags_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANOMALY_MODEL_PATH", tmp_path / "nope.joblib")
    monkeypatch.setattr(anomaly, "_cache", {})
    out = anomaly.check(NORMAL)
    assert out == {"available": False, "is_unusual": False, "score": None}