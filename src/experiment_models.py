from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    probability: np.ndarray
    importance: pd.DataFrame
    best_iteration: int | None
    model: Any
    preprocessing: Any = None


def _empty_importance() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "importance", "importance_type"])


def _frequency_encode(
    train: pd.DataFrame,
    other: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Memory-bounded numeric encoding fitted only on the training split."""
    categorical = set(categorical_features)
    train_array = np.empty((len(train), train.shape[1]), dtype="float32")
    other_array = np.empty((len(other), other.shape[1]), dtype="float32")
    state: dict[str, Any] = {"columns": train.columns.tolist(), "encoders": {}}

    for index, column in enumerate(train.columns):
        if column in categorical:
            train_text = train[column].astype("string").fillna("__MISSING__")
            other_text = other[column].astype("string").fillna("__MISSING__")
            frequency = train_text.value_counts(normalize=True, dropna=False)
            train_array[:, index] = train_text.map(frequency).fillna(0).to_numpy("float32")
            other_array[:, index] = other_text.map(frequency).fillna(0).to_numpy("float32")
            state["encoders"][column] = {
                "type": "frequency",
                "values": frequency.to_dict(),
            }
        else:
            train_values = pd.to_numeric(train[column], errors="coerce")
            other_values = pd.to_numeric(other[column], errors="coerce")
            median = float(train_values.median())
            if not np.isfinite(median):
                median = 0.0
            train_array[:, index] = train_values.fillna(median).to_numpy("float32")
            other_array[:, index] = other_values.fillna(median).to_numpy("float32")
            state["encoders"][column] = {"type": "median", "value": median}
    return train_array, other_array, state


def _generic_importance(
    feature_names: list[str], values: np.ndarray, importance_type: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.asarray(values, dtype="float64"),
            "importance_type": importance_type,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def _lightgbm_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    return "brier", float(np.mean(np.square(y_pred - y_true))), False


def fit_validation_model(
    model_config: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str],
    *,
    seed: int,
    n_jobs: int,
    tuning_data: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
    ]
    | None = None,
) -> ModelResult:
    model_type = model_config["type"]
    params = dict(model_config.get("params", {}))
    refit_after_tuning = bool(model_config.get("refit_after_tuning", False))
    if refit_after_tuning and tuning_data is None:
        raise ValueError("refit_after_tuning requires a separate tuning_data split")

    if model_type == "constant":
        probability = np.full(len(x_valid), float(y_train.mean()), dtype="float64")
        return ModelResult(probability, _empty_importance(), None, None)

    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm") from exc

        early_stopping = int(params.pop("early_stopping_rounds", 100))
        defaults = {
            "objective": "binary",
            "metric": "None",
            "n_estimators": 1500,
            "learning_rate": 0.025,
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        defaults.update(params)
        if refit_after_tuning:
            tune_train, tune_y, tune_valid, tune_valid_y = tuning_data
            tune_model = lgb.LGBMClassifier(**defaults)
            tune_model.fit(
                tune_train,
                tune_y.astype("int8"),
                eval_set=[(tune_valid, tune_valid_y.astype("int8"))],
                eval_metric=_lightgbm_eval,
                categorical_feature=categorical_features,
                callbacks=[
                    lgb.early_stopping(
                        early_stopping,
                        first_metric_only=True,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=100),
                ],
            )
            best_iteration = max(1, int(tune_model.best_iteration_))
            del tune_model
            gc.collect()
            final_defaults = dict(defaults)
            final_defaults["n_estimators"] = best_iteration
            model = lgb.LGBMClassifier(**final_defaults)
            model.fit(
                x_train,
                y_train.astype("int8"),
                categorical_feature=categorical_features,
            )
            probability = model.predict_proba(x_valid)[:, 1]
        else:
            model = lgb.LGBMClassifier(**defaults)
            model.fit(
                x_train,
                y_train.astype("int8"),
                eval_set=[(x_valid, y_valid.astype("int8"))],
                eval_metric=_lightgbm_eval,
                categorical_feature=categorical_features,
                callbacks=[
                    lgb.early_stopping(
                        early_stopping,
                        first_metric_only=True,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=100),
                ],
            )
            best_iteration = int(model.best_iteration_)
            probability = model.predict_proba(
                x_valid, num_iteration=model.best_iteration_
            )[:, 1]
        importance = pd.DataFrame(
            {
                "feature": x_train.columns,
                "importance": model.booster_.feature_importance(importance_type="gain"),
                "split": model.booster_.feature_importance(importance_type="split"),
                "importance_type": "gain",
            }
        ).sort_values("importance", ascending=False, ignore_index=True)
        return ModelResult(
            probability=probability,
            importance=importance,
            best_iteration=best_iteration,
            model=model,
        )

    if model_type in {"logistic", "hist_gbdt", "extra_trees", "xgboost"}:
        defer_full_encoding = refit_after_tuning and model_type in {
            "hist_gbdt",
            "xgboost",
        }
        if defer_full_encoding:
            x_train_array = None
            x_valid_array = None
            encoding = None
        else:
            x_train_array, x_valid_array, encoding = _frequency_encode(
                x_train, x_valid, categorical_features
            )

        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler(copy=False)
            x_train_array = scaler.fit_transform(x_train_array)
            x_valid_array = scaler.transform(x_valid_array)
            defaults = {
                "C": 0.2,
                "max_iter": 250,
                "solver": "lbfgs",
                "random_state": seed,
                "n_jobs": n_jobs,
            }
            defaults.update(params)
            model = LogisticRegression(**defaults)
            model.fit(x_train_array, y_train.astype("int8"))
            probability = model.predict_proba(x_valid_array)[:, 1]
            importance = _generic_importance(
                x_train.columns.tolist(), np.abs(model.coef_[0]), "absolute_coefficient"
            )
            return ModelResult(
                probability, importance, int(model.n_iter_.max()), model, (encoding, scaler)
            )

        if model_type == "hist_gbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier

            defaults = {
                "learning_rate": 0.06,
                "max_iter": 350,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 200,
                "l2_regularization": 2.0,
                "random_state": seed,
                "early_stopping": True,
                "validation_fraction": 0.1,
            }
            defaults.update(params)
            if refit_after_tuning:
                tune_train, tune_y, tune_valid, tune_valid_y = tuning_data
                tune_train_array, tune_valid_array, _ = _frequency_encode(
                    tune_train,
                    tune_valid,
                    categorical_features,
                )
                tune_defaults = dict(defaults)
                tune_defaults["early_stopping"] = False
                tune_model = HistGradientBoostingClassifier(**tune_defaults)
                tune_model.fit(tune_train_array, tune_y.astype("int8"))
                staged_brier = [
                    float(
                        np.mean(
                            np.square(
                                probability[:, 1]
                                - tune_valid_y.to_numpy(dtype="float64")
                            )
                        )
                    )
                    for probability in tune_model.staged_predict_proba(tune_valid_array)
                ]
                best_iteration = int(np.argmin(staged_brier) + 1)
                del tune_model, tune_train_array, tune_valid_array, staged_brier
                gc.collect()
                x_train_array, x_valid_array, encoding = _frequency_encode(
                    x_train,
                    x_valid,
                    categorical_features,
                )
                final_defaults = dict(defaults)
                final_defaults["max_iter"] = best_iteration
                final_defaults["early_stopping"] = False
                model = HistGradientBoostingClassifier(**final_defaults)
                model.fit(x_train_array, y_train.astype("int8"))
            else:
                model = HistGradientBoostingClassifier(**defaults)
                model.fit(x_train_array, y_train.astype("int8"))
                best_iteration = int(getattr(model, "n_iter_", defaults["max_iter"]))
            probability = model.predict_proba(x_valid_array)[:, 1]
            return ModelResult(probability, _empty_importance(), best_iteration, model, encoding)

        if model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier

            defaults = {
                "n_estimators": 350,
                "max_depth": 18,
                "min_samples_leaf": 40,
                "max_features": 0.7,
                "bootstrap": True,
                "max_samples": 0.5,
                "random_state": seed,
                "n_jobs": n_jobs,
                "class_weight": None,
            }
            defaults.update(params)
            model = ExtraTreesClassifier(**defaults)
            model.fit(x_train_array, y_train.astype("int8"))
            probability = model.predict_proba(x_valid_array)[:, 1]
            importance = _generic_importance(
                x_train.columns.tolist(), model.feature_importances_, "impurity"
            )
            return ModelResult(probability, importance, defaults["n_estimators"], model, encoding)

        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost") from exc
        early_stopping = int(params.pop("early_stopping_rounds", 100))
        defaults = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_estimators": 1400,
            "learning_rate": 0.03,
            "max_depth": 7,
            "min_child_weight": 50,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.2,
            "reg_lambda": 3.0,
            "random_state": seed,
            "n_jobs": n_jobs,
        }
        defaults.update(params)
        if refit_after_tuning:
            tune_train, tune_y, tune_valid, tune_valid_y = tuning_data
            tune_train_array, tune_valid_array, _ = _frequency_encode(
                tune_train,
                tune_valid,
                categorical_features,
            )
            tune_model = xgb.XGBClassifier(
                **defaults,
                early_stopping_rounds=early_stopping,
            )
            tune_model.fit(
                tune_train_array,
                tune_y.astype("int8"),
                eval_set=[(tune_valid_array, tune_valid_y.astype("int8"))],
                verbose=False,
            )
            best_iteration = max(1, int(tune_model.best_iteration) + 1)
            del tune_model, tune_train_array, tune_valid_array
            gc.collect()
            x_train_array, x_valid_array, encoding = _frequency_encode(
                x_train,
                x_valid,
                categorical_features,
            )
            final_defaults = dict(defaults)
            final_defaults["n_estimators"] = best_iteration
            model = xgb.XGBClassifier(**final_defaults)
            model.fit(x_train_array, y_train.astype("int8"), verbose=False)
        elif early_stopping > 0:
            model = xgb.XGBClassifier(**defaults, early_stopping_rounds=early_stopping)
            model.fit(
                x_train_array,
                y_train.astype("int8"),
                eval_set=[(x_valid_array, y_valid.astype("int8"))],
                verbose=False,
            )
            best_iteration = int(
                getattr(model, "best_iteration", defaults["n_estimators"])
            )
        else:
            # Fixed-iteration path used by the team comparison.  It deliberately
            # avoids using the 2024 holdout for early stopping.
            model = xgb.XGBClassifier(**defaults)
            model.fit(x_train_array, y_train.astype("int8"), verbose=False)
            best_iteration = int(defaults["n_estimators"])
        probability = model.predict_proba(x_valid_array)[:, 1]
        importance = _generic_importance(
            x_train.columns.tolist(), model.feature_importances_, "gain"
        )
        return ModelResult(probability, importance, best_iteration, model, encoding)

    if model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("CatBoost is not installed. Run: pip install catboost") from exc

        defaults = {
            "loss_function": "Logloss",
            "eval_metric": "BrierScore",
            "iterations": 1400,
            "learning_rate": 0.035,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "random_seed": seed,
            "thread_count": n_jobs,
            "verbose": 100,
            "allow_writing_files": False,
        }
        defaults.update(params)
        early_stopping_rounds = int(model_config.get("early_stopping_rounds", 100))
        if refit_after_tuning:
            tune_train, tune_y, tune_valid, tune_valid_y = tuning_data
            tune_train_copy = tune_train.copy()
            tune_valid_copy = tune_valid.copy()
            for column in categorical_features:
                tune_train_copy[column] = (
                    tune_train_copy[column].astype("string").fillna("__MISSING__")
                )
                tune_valid_copy[column] = (
                    tune_valid_copy[column].astype("string").fillna("__MISSING__")
                )
            tune_cat_indices = [
                tune_train_copy.columns.get_loc(column)
                for column in categorical_features
            ]
            tune_model = CatBoostClassifier(**defaults)
            tune_model.fit(
                tune_train_copy,
                tune_y.astype("int8"),
                cat_features=tune_cat_indices,
                eval_set=(tune_valid_copy, tune_valid_y.astype("int8")),
                early_stopping_rounds=early_stopping_rounds,
            )
            tuned_iteration = int(tune_model.get_best_iteration())
            best_iteration = (
                tuned_iteration + 1
                if tuned_iteration >= 0
                else int(defaults["iterations"])
            )
            del tune_model, tune_train_copy, tune_valid_copy
            gc.collect()
            final_defaults = dict(defaults)
            final_defaults["iterations"] = best_iteration
            train_copy = x_train.copy()
            valid_copy = x_valid.copy()
            for column in categorical_features:
                train_copy[column] = (
                    train_copy[column].astype("string").fillna("__MISSING__")
                )
                valid_copy[column] = (
                    valid_copy[column].astype("string").fillna("__MISSING__")
                )
            cat_indices = [
                train_copy.columns.get_loc(column)
                for column in categorical_features
            ]
            model = CatBoostClassifier(**final_defaults)
            model.fit(
                train_copy,
                y_train.astype("int8"),
                cat_features=cat_indices,
            )
        else:
            train_copy = x_train.copy()
            valid_copy = x_valid.copy()
            for column in categorical_features:
                train_copy[column] = (
                    train_copy[column].astype("string").fillna("__MISSING__")
                )
                valid_copy[column] = (
                    valid_copy[column].astype("string").fillna("__MISSING__")
                )
            cat_indices = [
                train_copy.columns.get_loc(column)
                for column in categorical_features
            ]
            model = CatBoostClassifier(**defaults)
            if early_stopping_rounds > 0:
                model.fit(
                    train_copy,
                    y_train.astype("int8"),
                    cat_features=cat_indices,
                    eval_set=(valid_copy, y_valid.astype("int8")),
                    early_stopping_rounds=early_stopping_rounds,
                )
                best_iteration = int(model.get_best_iteration())
            else:
                # Fixed-iteration path for fair LOFO/add-back comparisons.  The
                # 2024 holdout is never used to choose the iteration count.
                model.fit(
                    train_copy,
                    y_train.astype("int8"),
                    cat_features=cat_indices,
                )
                best_iteration = int(defaults["iterations"])
        probability = model.predict_proba(valid_copy)[:, 1]
        importance = _generic_importance(
            x_train.columns.tolist(), model.feature_importances_, "prediction_values_change"
        )
        return ModelResult(
            probability, importance, best_iteration, model, cat_indices
        )

    raise ValueError(f"Unknown model type: {model_type}")


def save_model_artifact(result: ModelResult, model_type: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.model is None:
        (output_dir / "constant_model.json").write_text(
            json.dumps({"note": "constant baseline has no fitted estimator"}, indent=2),
            encoding="utf-8",
        )
        return
    if model_type == "lightgbm":
        result.model.booster_.save_model(output_dir / "model.txt")
    elif model_type == "xgboost":
        result.model.save_model(output_dir / "model.json")
    elif model_type == "catboost":
        result.model.save_model(output_dir / "model.cbm")
    else:
        import joblib

        joblib.dump(
            {"model": result.model, "preprocessing": result.preprocessing},
            output_dir / "model.joblib",
        )
