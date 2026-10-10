"""Unusual-application detection with an Isolation Forest.

Trained on the same cleaned training data as the loan model. It does not decide
approval; it only flags applications that look unlike anything seen in training.
"""
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

import config
from src import pipeline, predict

_cache: dict = {}


def build_detector(contamination: float = 0.01) -> Pipeline:
    """Same preprocessing as the loan model, followed by an Isolation Forest.

    contamination = share of training rows treated as 'unusual' (1%).
    """
    return Pipeline([
        ("preprocess", pipeline.build_preprocessor()),
        ("forest", IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def load_detector():
    """Load the saved detector once. Returns None if it hasn't been trained yet."""
    if "detector" not in _cache:
        if not config.ANOMALY_MODEL_PATH.exists():
            return None
        _cache["detector"] = joblib.load(config.ANOMALY_MODEL_PATH)
    return _cache["detector"]


def check(inputs: dict, detector=None) -> dict:
    """Return {'available': bool, 'is_unusual': bool, 'score': float | None}.

    Lower score = more unusual. If no detector is trained, nothing is flagged.
    """
    if detector is None:
        detector = load_detector()
    if detector is None:
        return {"available": False, "is_unusual": False, "score": None}

    frame = predict.make_frame(inputs)
    is_unusual = int(detector.predict(frame)[0]) == -1
    score = float(detector.score_samples(frame)[0])
    return {"available": True, "is_unusual": is_unusual, "score": score}