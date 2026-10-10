"""Load the trained pipeline, score one application, and save the result."""
import json
from functools import lru_cache

import joblib
import pandas as pd

import config
from src import db

APPROVAL_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_pipeline():
    """Load the saved pipeline once and reuse it."""
    if not config.PIPELINE_PATH.exists():
        raise FileNotFoundError("Model not found. Run:  python scripts/train.py")
    return joblib.load(config.PIPELINE_PATH)


def make_frame(inputs: dict) -> pd.DataFrame:
    """Turn a dict of form values into a one-row DataFrame in the right column order."""
    missing = [c for c in config.FEATURE_COLS if c not in inputs]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    return pd.DataFrame([{c: inputs[c] for c in config.FEATURE_COLS}])


def predict(inputs: dict, pipe=None) -> dict:
    """Return {'probability': float, 'decision': 'Approved' | 'Rejected'}."""
    pipe = pipe or load_pipeline()
    proba = float(pipe.predict_proba(make_frame(inputs))[0, 1])
    decision = "Approved" if proba >= APPROVAL_THRESHOLD else "Rejected"
    return {"probability": proba, "decision": decision}


def save_prediction(user_id: int, inputs: dict, result: dict, db_path=None) -> int:
    """Store one prediction in the database and return its id."""
    with db.get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO predictions (user_id, inputs_json, probability, decision) "
            "VALUES (?, ?, ?, ?)",
            (user_id, json.dumps(inputs), result["probability"], result["decision"]),
        )
        row_id = cur.lastrowid
    assert row_id is not None
    return row_id