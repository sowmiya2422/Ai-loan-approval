"""Train candidate models, pick the best by cross-validated ROC-AUC, evaluate once
on the held-out test set, and save the pipeline + metrics.

Run:  python scripts/train.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

import config  # noqa: E402
from src import data, pipeline  # noqa: E402


def main():
    df = data.clean(data.load_raw())
    X_train, X_test, y_train, y_test = data.split_data(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # 1) Compare models using cross-validation on the TRAINING set only.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = {}
    for name in pipeline.get_models():
        pipe = pipeline.build_pipeline(name)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = float(scores.mean())
        print(f"{name:20s} CV ROC-AUC: {cv_scores[name]:.4f}")

    # 2) Fit the winner on the full training set, test it once.
    best_name = max(cv_scores, key=lambda n: cv_scores[n])
    best = pipeline.build_pipeline(best_name).fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # 3) Fairness audit: approval rate per gender (gender is NOT a model input).
    fairness = {}
    for group in sorted(X_test["person_gender"].dropna().unique()):
        mask = (X_test["person_gender"] == group).to_numpy()
        fairness[group] = {
            "n": int(mask.sum()),
            "approval_rate": round(float(pred[mask].mean()), 4),
        }

    metrics = {
        "best_model": best_name,
        "cv_roc_auc": {k: round(v, 4) for k, v in cv_scores.items()},
        "test": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        },
        "fairness_approval_rate_by_gender": fairness,
    }

    # 4) Save.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, config.PIPELINE_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nBest model: {best_name}")
    print(json.dumps(metrics["test"], indent=2))
    print("Approval rate by gender:", fairness)
    print(f"Saved: {config.PIPELINE_PATH.name}, {config.METRICS_PATH.name}")


if __name__ == "__main__":
    main()