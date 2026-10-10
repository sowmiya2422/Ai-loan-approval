"""Load, validate, clean and split the loan dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_raw(path=None) -> pd.DataFrame:
    """Read the CSV and make sure all required columns exist."""
    df = pd.read_csv(path or config.DATA_PATH)
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, tidy text columns, and blank out impossible values.

    Out-of-range values become NaN here; the pipeline imputes them later
    using training data only (so no information leaks from the test set).
    """
    df = df.drop_duplicates().copy()

    for col in config.CATEGORICAL_COLS + config.SENSITIVE_COLS:
        df[col] = df[col].str.strip()
    for col in config.SENSITIVE_COLS:
        df[col] = df[col].str.lower()

    for col, (low, high) in config.VALID_RANGES.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Stratified train/test split. Sensitive columns are kept in X for
    fairness auditing only; the pipeline never uses them as model inputs."""
    X = df[config.FEATURE_COLS + config.SENSITIVE_COLS]
    y = df[config.TARGET_COL]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )