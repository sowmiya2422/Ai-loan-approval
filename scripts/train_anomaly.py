"""Train the unusual-application detector and save it.

Run:  python scripts/train_anomaly.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from src import anomaly, data  # noqa: E402


def main():
    df = data.clean(data.load_raw())
    X_train, X_test, _, _ = data.split_data(df)

    detector = anomaly.build_detector().fit(X_train)

    # Sanity check: roughly 1% of unseen normal data should be flagged.
    flagged = np.mean(detector.predict(X_test) == -1)
    print(f"Flagged as unusual on held-out test data: {flagged:.2%}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, config.ANOMALY_MODEL_PATH)
    print(f"Saved: {config.ANOMALY_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
