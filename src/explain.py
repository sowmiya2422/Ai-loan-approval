"""SHAP explanations: which inputs pushed one prediction towards approve or reject."""
import numpy as np
import pandas as pd
import shap

from src import predict
import config


def _original_feature(transformed_name: str) -> str:
    """'num__credit_score' -> 'credit_score'; 'cat__loan_intent_PERSONAL' -> 'loan_intent'."""
    kind, _, rest = transformed_name.partition("__")
    if kind == "num":
        return rest
    for col in config.CATEGORICAL_COLS:
        if rest.startswith(col + "_"):
            return col
    return rest


def explain(inputs: dict, pipe=None, top_n: int = 6) -> pd.DataFrame:
    """Return the top_n features driving this prediction.

    Columns: feature, value (what was entered), impact (SHAP, in log-odds;
    positive = pushes towards approval, negative = towards rejection).
    One-hot columns are summed back into their original feature.
    """
    pipe = pipe or predict.load_pipeline()
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    X = pre.transform(predict.make_frame(inputs))
    names = pre.get_feature_names_out()

    values = np.asarray(shap.TreeExplainer(model).shap_values(X))
    if values.ndim == 3:  # some tree models return one array per class
        values = values[:, :, 1]
    values = values[0]

    df = pd.DataFrame({
        "feature": [_original_feature(n) for n in names],
        "impact": values,
    })
    df = df.groupby("feature", as_index=False)["impact"].sum()
    df["value"] = df["feature"].map(inputs)
    df = df.reindex(df["impact"].abs().sort_values(ascending=False).index)
    top = df.head(top_n)[["feature", "value", "impact"]]
    return pd.DataFrame(top).reset_index(drop=True)