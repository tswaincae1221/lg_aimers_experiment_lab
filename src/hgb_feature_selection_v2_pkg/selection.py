from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from .common import EngineeredData, EncodedData, _append_checkpoint, _checkpoint_rows, _safe_numeric

LOGGER = logging.getLogger(__name__)

def frequency_encode(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    categorical_features: list[str],
) -> EncodedData:
    """Training-only frequency encoding copied from the working experiment_models pattern."""
    categorical = set(categorical_features)
    x_train = np.empty((len(train), train.shape[1]), dtype="float32")
    x_valid = np.empty((len(valid), valid.shape[1]), dtype="float32")
    state: dict[str, dict] = {"columns": train.columns.tolist(), "encoders": {}}
    for index, column in enumerate(train.columns):
        if column in categorical:
            train_text = train[column].astype("string").fillna("__MISSING__")
            valid_text = valid[column].astype("string").fillna("__MISSING__")
            frequency = train_text.value_counts(normalize=True, dropna=False)
            x_train[:, index] = train_text.map(frequency).fillna(0.0).to_numpy("float32")
            x_valid[:, index] = valid_text.map(frequency).fillna(0.0).to_numpy("float32")
            state["encoders"][column] = {"type": "frequency", "values": frequency.to_dict()}
        else:
            train_values = _safe_numeric(train[column])
            valid_values = _safe_numeric(valid[column])
            median = float(train_values.median())
            if not np.isfinite(median):
                median = 0.0
            x_train[:, index] = train_values.fillna(median).to_numpy("float32")
            x_valid[:, index] = valid_values.fillna(median).to_numpy("float32")
            state["encoders"][column] = {"type": "median", "value": median}
    return EncodedData(
        x_train=x_train,
        x_valid=x_valid,
        feature_names=train.columns.tolist(),
        encoder_state=state,
    )


def _hgb_params(seed: int, max_iter: int) -> dict:
    return {
        "learning_rate": 0.06,
        "max_iter": max_iter,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 250,
        "l2_regularization": 3.0,
        "random_state": seed,
        "early_stopping": False,
    }


def brier(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(brier_score_loss(y_true, np.clip(probability, 1e-6, 1 - 1e-6)))


def score_row(y_true: np.ndarray, probability: np.ndarray, label: str, feature_count: int) -> dict:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    rate = float(np.mean(y_true))
    base = rate * (1.0 - rate)
    return {
        "model": label,
        "n": int(len(y_true)),
        "feature_count": int(feature_count),
        "target_rate": rate,
        "prediction_mean": float(np.mean(probability)),
        "brier": brier(y_true, probability),
        "brier_skill_score": max(0.0, 100000.0 * (1.0 - brier(y_true, probability) / base)) if base else 0.0,
        "auc": float(roc_auc_score(y_true, probability)),
    }


def tune_iterations(
    engineered: EngineeredData,
    *,
    tuning_season: int,
    max_iter: int,
    seed: int,
) -> int:
    tune_train_mask = engineered.seasons.astype(int) < tuning_season
    tune_valid_mask = engineered.seasons.astype(int) == tuning_season
    if tune_train_mask.sum() == 0 or tune_valid_mask.sum() == 0:
        LOGGER.warning("Tuning season unavailable; using max_iter=%d", max_iter)
        return max_iter
    encoded = frequency_encode(
        engineered.features.loc[tune_train_mask],
        engineered.features.loc[tune_valid_mask],
        engineered.categorical_features,
    )
    model = HistGradientBoostingClassifier(**_hgb_params(seed, max_iter))
    model.fit(encoded.x_train, engineered.target.loc[tune_train_mask].to_numpy("int8"))
    y_tune = engineered.target.loc[tune_valid_mask].to_numpy("int8")
    staged = [brier(y_tune, p[:, 1]) for p in model.staged_predict_proba(encoded.x_valid)]
    best = int(np.argmin(staged) + 1)
    LOGGER.info("Tuned HGB iterations on season %d: %d/%d", tuning_season, best, max_iter)
    return best


def fit_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    column_indices: np.ndarray,
    *,
    max_iter: int,
    seed: int,
) -> tuple[HistGradientBoostingClassifier, np.ndarray, float]:
    model = HistGradientBoostingClassifier(**_hgb_params(seed, max_iter))
    model.fit(x_train[:, column_indices], y_train)
    probability = model.predict_proba(x_valid[:, column_indices])[:, 1]
    return model, probability, brier(y_valid, probability)


def run_block_ablation(
    engineered: EngineeredData,
    encoded: EncodedData,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    *,
    max_iter: int,
    seed: int,
    full_brier: float,
    checkpoint_path: Path,
    run_signature: str,
    rerun: bool,
) -> pd.DataFrame:
    name_to_index = {name: i for i, name in enumerate(encoded.feature_names)}
    history = pd.DataFrame() if rerun else _checkpoint_rows(checkpoint_path, run_signature)
    completed = set(history["block"].astype(str)) if not history.empty else set()
    for block, columns in engineered.blocks.items():
        if block in completed:
            LOGGER.info("Block checkpoint hit: %s", block)
            continue
        drop = {column for column in columns if column in name_to_index}
        if not drop or len(drop) == len(encoded.feature_names):
            continue
        keep_idx = np.array(
            [i for i, name in enumerate(encoded.feature_names) if name not in drop], dtype="int32"
        )
        start = time.perf_counter()
        _, _, score = fit_subset(
            encoded.x_train,
            y_train,
            encoded.x_valid,
            y_valid,
            keep_idx,
            max_iter=max_iter,
            seed=seed,
        )
        row = {
            "run_signature": run_signature,
            "block": block,
            "removed_features": len(drop),
            "brier_without": score,
            "delta_brier": score - full_brier,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history = _append_checkpoint(checkpoint_path, history, row)
        LOGGER.info("Block ablation %-24s delta_brier=%+.8f", block, score - full_brier)
    current = history.loc[history["run_signature"].astype("string") == run_signature].copy()
    return current.sort_values("delta_brier", ascending=False, ignore_index=True)


def run_permutation_importance(
    model: HistGradientBoostingClassifier,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    *,
    repeats: int,
    sample_size: int | None,
    seed: int,
    checkpoint_path: Path,
    run_signature: str,
    rerun: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if sample_size is not None and sample_size < len(y_valid):
        sample_idx = np.sort(rng.choice(len(y_valid), size=sample_size, replace=False))
        x = x_valid[sample_idx].copy()
        y = y_valid[sample_idx]
    else:
        x = x_valid.copy()
        y = y_valid
    baseline = brier(y, model.predict_proba(x)[:, 1])
    history = pd.DataFrame() if rerun else _checkpoint_rows(checkpoint_path, run_signature)
    completed = set(history["feature"].astype(str)) if not history.empty else set()
    for j, feature in enumerate(feature_names):
        if feature in completed:
            continue
        deltas = []
        original = x[:, j].copy()
        for _ in range(repeats):
            x[:, j] = rng.permutation(original)
            perm_brier = brier(y, model.predict_proba(x)[:, 1])
            deltas.append(perm_brier - baseline)
        x[:, j] = original
        row = {
            "run_signature": run_signature,
            "feature": feature,
            "permutation_delta_brier_mean": float(np.mean(deltas)),
            "permutation_delta_brier_std": float(np.std(deltas, ddof=0)),
            "repeats": repeats,
            "sample_n": len(y),
        }
        history = _append_checkpoint(checkpoint_path, history, row)
    current = history.loc[history["run_signature"].astype("string") == run_signature].copy()
    return current.sort_values(
        "permutation_delta_brier_mean", ascending=False, ignore_index=True
    )


def run_lofo(
    encoded: EncodedData,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    permutation: pd.DataFrame,
    *,
    max_candidates: int,
    max_iter: int,
    seed: int,
    full_brier: float,
    checkpoint_path: Path,
    run_signature: str,
    rerun: bool,
) -> pd.DataFrame:
    candidates = permutation.sort_values("permutation_delta_brier_mean", ascending=True).head(max_candidates)
    name_to_index = {name: i for i, name in enumerate(encoded.feature_names)}
    history = pd.DataFrame() if rerun else _checkpoint_rows(checkpoint_path, run_signature)
    completed = set(history["feature"].astype(str)) if not history.empty else set()
    for feature in candidates["feature"]:
        if feature in completed:
            LOGGER.info("LOFO checkpoint hit: %s", feature)
            continue
        drop_index = name_to_index[feature]
        keep_idx = np.array(
            [i for i in range(len(encoded.feature_names)) if i != drop_index], dtype="int32"
        )
        start = time.perf_counter()
        _, _, score = fit_subset(
            encoded.x_train,
            y_train,
            encoded.x_valid,
            y_valid,
            keep_idx,
            max_iter=max_iter,
            seed=seed,
        )
        row = {
            "run_signature": run_signature,
            "feature": feature,
            "brier_without": score,
            "lofo_delta_brier": score - full_brier,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history = _append_checkpoint(checkpoint_path, history, row)
        LOGGER.info("LOFO %-42s delta_brier=%+.8f", feature, score - full_brier)
    current = history.loc[history["run_signature"].astype("string") == run_signature].copy()
    if current.empty:
        return current
    return current.sort_values("lofo_delta_brier", ascending=False, ignore_index=True)


def choose_selected_features(
    feature_names: list[str],
    lofo: pd.DataFrame,
    *,
    drop_tolerance: float,
) -> tuple[list[str], list[str]]:
    if lofo.empty:
        return feature_names, []
    drop = lofo.loc[lofo["lofo_delta_brier"] <= drop_tolerance, "feature"].tolist()
    keep = [feature for feature in feature_names if feature not in set(drop)]
    return keep, drop


def _feature_catalog(engineered: EngineeredData) -> pd.DataFrame:
    block_by_feature = {}
    for block, columns in engineered.blocks.items():
        for feature in columns:
            block_by_feature.setdefault(feature, block)
    return pd.DataFrame(
        {
            "feature": engineered.features.columns,
            "block": [block_by_feature.get(feature, "unassigned") for feature in engineered.features.columns],
            "categorical": [
                feature in set(engineered.categorical_features)
                for feature in engineered.features.columns
            ],
        }
    )

