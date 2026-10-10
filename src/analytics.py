"""Read-only data helpers for the history page and the dashboard."""
import json

import pandas as pd

import config
from src import data, db

PREDICTION_COLUMNS = [
    "id", "created_at", "username", "probability", "decision",
    "loan_amnt", "person_income", "credit_score", "loan_intent",
]


def load_metrics() -> dict | None:
    """Metrics saved by scripts/train.py, or None if the model isn't trained yet."""
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text())


def load_predictions(user_id: int | None = None, db_path=None) -> pd.DataFrame:
    """All predictions (admin view) or one user's predictions, newest first."""
    query = (
        "SELECT p.id, p.created_at, u.username, p.inputs_json, p.probability, p.decision "
        "FROM predictions p JOIN users u ON u.id = p.user_id"
    )
    params: tuple = ()
    if user_id is not None:
        query += " WHERE p.user_id = ?"
        params = (user_id,)
    query += " ORDER BY p.id DESC"

    with db.get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    records = []
    for r in rows:
        inputs = json.loads(r["inputs_json"])
        records.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "username": r["username"],
            "probability": r["probability"],
            "decision": r["decision"],
            "loan_amnt": inputs.get("loan_amnt"),
            "person_income": inputs.get("person_income"),
            "credit_score": inputs.get("credit_score"),
            "loan_intent": inputs.get("loan_intent"),
        })
    df = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def approval_rate_by(column: str) -> pd.DataFrame:
    """Approval rate and application count per category in the training dataset."""
    df = data.clean(data.load_raw())
    out = (
        df.groupby(column)[config.TARGET_COL]
        .agg(approval_rate="mean", applications="count")
        .reset_index()
    )
    return out.sort_values("approval_rate", ascending=False).reset_index(drop=True)