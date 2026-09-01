"""Portable linear residual correction helpers.

Training may use scikit-learn, but inference needs only NumPy and pandas.  The
preprocessor is stored as plain arrays/dicts in the submission bundle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def fit_residual_preprocessor(
    X: pd.DataFrame,
    base_pred: np.ndarray,
    categorical_cols: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, dict]:
    source_features = list(X.columns)
    categorical = [c for c in categorical_cols if c in source_features]
    numeric = [c for c in source_features if c not in categorical]
    frame = X.loc[:, source_features].copy()
    medians: dict[str, float] = {}
    for col in numeric:
        values = pd.to_numeric(frame[col], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        medians[col] = median
        frame[col] = values.fillna(median).astype("float32")
    for col in categorical:
        frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)
    encoded = pd.get_dummies(
        frame, columns=categorical, dummy_na=False, dtype=np.float32
    )
    columns = encoded.columns.tolist()
    arr = encoded.to_numpy(dtype="float32")
    mean = arr.mean(axis=0, dtype="float64").astype("float32")
    std = arr.std(axis=0, dtype="float64").astype("float32")
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    p = np.clip(np.asarray(base_pred, dtype="float32"), 1e-5, 1 - 1e-5)
    design = np.column_stack([(arr - mean) / std, p, np.log(p / (1 - p))])
    prep = {
        "source_features": source_features,
        "categorical_cols": categorical,
        "numeric_cols": numeric,
        "numeric_medians": medians,
        "encoded_columns": columns,
        "mean": mean,
        "std": std,
    }
    return design.astype("float32"), prep


def transform_residual_features(
    X: pd.DataFrame, base_pred: np.ndarray, prep: dict
) -> np.ndarray:
    frame = X.loc[:, prep["source_features"]].copy()
    for col in prep["numeric_cols"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(
            prep["numeric_medians"][col]
        ).astype("float32")
    for col in prep["categorical_cols"]:
        frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)
    encoded = pd.get_dummies(
        frame,
        columns=prep["categorical_cols"],
        dummy_na=False,
        dtype=np.float32,
    ).reindex(columns=prep["encoded_columns"], fill_value=0.0)
    arr = encoded.to_numpy(dtype="float32")
    p = np.clip(np.asarray(base_pred, dtype="float32"), 1e-5, 1 - 1e-5)
    return np.column_stack([
        (arr - prep["mean"]) / prep["std"],
        p,
        np.log(p / (1 - p)),
    ]).astype("float32")


def fit_ridge_bundle(
    design: np.ndarray,
    residual: np.ndarray,
    alpha: float,
    prep: dict,
    weight: float,
) -> dict:
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
    model.fit(design, np.asarray(residual, dtype="float32"))
    return {
        "family": "ridge",
        "alpha": float(alpha),
        "weight": float(weight),
        "coef": np.asarray(model.coef_, dtype="float32"),
        "intercept": float(model.intercept_),
        "preprocessor": prep,
    }


def predict_ridge_correction(
    X: pd.DataFrame, base_pred: np.ndarray, bundle: dict
) -> np.ndarray:
    design = transform_residual_features(X, base_pred, bundle["preprocessor"])
    return (
        design @ np.asarray(bundle["coef"], dtype="float32")
        + float(bundle["intercept"])
    )
