from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.first_model_features import (
    ID_COLUMN,
    TARGET,
    FeatureBundle,
    aggregate_trackman,
    assemble_features,
    load_mapping,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe first model for LG Aimers pitch control prediction."
    )
    parser.add_argument("--train", required=True, help="Official train.csv path")
    parser.add_argument("--test", help="Official test.csv path; optional for CV-only runs")
    parser.add_argument("--trackman", help="Official trackman_history.csv path")
    parser.add_argument("--mapping", help="pitcher_id to pitcher_trackman_id CSV path")
    parser.add_argument("--output-dir", default="results/first_model")
    parser.add_argument(
        "--cv-seasons",
        nargs="+",
        type=int,
        default=[2024],
        help="Validation seasons. Team standard: train through 2023 and validate on 2024.",
    )
    parser.add_argument("--accepted-mapping-grades", nargs="+", default=["확정", "높음"])
    parser.add_argument("--smoothing", type=float, default=50.0)
    parser.add_argument(
        "--asof-trends",
        action="store_true",
        help="Add the 13 official asof trend/contrast features for the V1.1 ablation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--max-rows-per-season",
        type=int,
        help="Development-only stratified row cap; do not use for final scoring.",
    )
    parser.add_argument(
        "--max-trackman-rows",
        type=int,
        help="Development-only Trackman row cap; do not use for final scoring.",
    )
    return parser.parse_args()


def sample_by_season(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame
    LOGGER.warning("Development row cap is active: max_rows_per_season=%d", limit)
    parts = [
        group.sample(min(len(group), limit), random_state=seed)
        for _, group in frame.groupby("season", observed=True, sort=True)
    ]
    sampled = pd.concat(parts).sort_index().reset_index(drop=True)
    return sampled


def brier_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    return "brier", float(np.mean(np.square(y_pred - y_true))), False


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y_true, probability))
    rate = float(np.mean(y_true))
    baseline = rate * (1 - rate)
    brier_skill_score = max(0.0, 100_000.0 * (1.0 - brier / baseline)) if baseline else 0.0
    return {
        "n": int(len(y_true)),
        "target_rate": rate,
        "prediction_mean": float(np.mean(probability)),
        "brier": brier,
        "brier_skill_score": brier_skill_score,
        "logloss": float(log_loss(y_true, probability, labels=[0, 1])),
        "auc": float(roc_auc_score(y_true, probability)),
    }


def model_parameters(args: argparse.Namespace, n_estimators: int | None = None) -> dict:
    return {
        "objective": "binary",
        "metric": "None",
        "n_estimators": n_estimators or args.n_estimators,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 300,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
        "max_bin": 255,
        "verbosity": -1,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "deterministic": True,
        "force_col_wise": True,
    }


def fit_fold(
    bundle: FeatureBundle,
    season: int,
    args: argparse.Namespace,
) -> tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray, dict[str, float]]:
    is_labeled = np.zeros(len(bundle.features), dtype=bool)
    is_labeled[bundle.train_rows] = True
    train_rows = np.flatnonzero(is_labeled & (bundle.seasons.to_numpy() < season))
    valid_rows = np.flatnonzero(is_labeled & (bundle.seasons.to_numpy() == season))
    if not len(train_rows) or not len(valid_rows):
        raise ValueError(
            f"Cannot build fold for season={season}: "
            f"train={len(train_rows)}, valid={len(valid_rows)}"
        )

    model = lgb.LGBMClassifier(**model_parameters(args))
    model.fit(
        bundle.features.iloc[train_rows],
        bundle.target.iloc[train_rows].astype("int8"),
        eval_set=[
            (
                bundle.features.iloc[valid_rows],
                bundle.target.iloc[valid_rows].astype("int8"),
            )
        ],
        eval_metric=brier_eval,
        categorical_feature=bundle.categorical_features,
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    probability = model.predict_proba(
        bundle.features.iloc[valid_rows], num_iteration=model.best_iteration_
    )[:, 1]
    fold_metrics = metrics(bundle.target.iloc[valid_rows].to_numpy(), probability)
    fold_metrics["season"] = season
    fold_metrics["train_n"] = int(len(train_rows))
    fold_metrics["best_iteration"] = int(model.best_iteration_)
    return model, valid_rows, probability, fold_metrics


def write_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    path: Path,
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(path, index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading train data: %s", args.train)
    train = pd.read_csv(args.train, encoding="utf-8-sig", low_memory=False)
    train = sample_by_season(train, args.max_rows_per_season, args.seed)
    test = None
    if args.test:
        LOGGER.info("Reading test data: %s", args.test)
        test = pd.read_csv(args.test, encoding="utf-8-sig", low_memory=False)

    trackman_summary = None
    mapping = None
    if bool(args.trackman) != bool(args.mapping):
        raise ValueError("--trackman and --mapping must be supplied together")
    if args.trackman:
        LOGGER.info("Aggregating Trackman data: %s", args.trackman)
        trackman_summary = aggregate_trackman(
            args.trackman,
            nrows=args.max_trackman_rows,
        )
        mapping = load_mapping(args.mapping, args.accepted_mapping_grades)
        LOGGER.info(
            "Trackman summary rows=%d, accepted mappings=%d/%d",
            len(trackman_summary),
            int(mapping["mapping_accepted"].sum()),
            len(mapping),
        )

    LOGGER.info("Building current, history, and Trackman features")
    bundle = assemble_features(
        train,
        test,
        trackman_summary=trackman_summary,
        mapping=mapping,
        smoothing=args.smoothing,
        include_asof_trends=args.asof_trends,
    )
    LOGGER.info(
        "Feature matrix rows=%d columns=%d categorical=%d",
        len(bundle.features),
        bundle.features.shape[1],
        len(bundle.categorical_features),
    )

    fold_rows: list[dict[str, float]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_models: list[lgb.LGBMClassifier] = []
    for season in args.cv_seasons:
        if season not in set(bundle.seasons.iloc[bundle.train_rows].astype(int)):
            LOGGER.warning("Skipping unavailable validation season=%d", season)
            continue
        LOGGER.info("Training season-forward fold: validation=%d", season)
        model, valid_rows, probability, fold_metric = fit_fold(bundle, season, args)
        fold_models.append(model)
        fold_rows.append(fold_metric)
        oof_parts.append(
            pd.DataFrame(
                {
                    ID_COLUMN: bundle.row_ids.iloc[valid_rows].to_numpy(),
                    "season": season,
                    TARGET: bundle.target.iloc[valid_rows].astype("int8").to_numpy(),
                    "prediction": probability,
                }
            )
        )
        LOGGER.info(
            "season=%d brier=%.6f BSS=%.2f auc=%.5f best_iteration=%d",
            season,
            fold_metric["brier"],
            fold_metric["brier_skill_score"],
            fold_metric["auc"],
            fold_metric["best_iteration"],
        )

    if not fold_models:
        raise ValueError("No CV folds were trained")

    metrics_frame = pd.DataFrame(fold_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    write_feature_importance(
        fold_models[-1],
        bundle.features.columns.tolist(),
        output_dir / "feature_importance.csv",
    )

    run_summary = {
        "train_path": str(Path(args.train)),
        "test_path": str(Path(args.test)) if args.test else None,
        "trackman_path": str(Path(args.trackman)) if args.trackman else None,
        "mapping_path": str(Path(args.mapping)) if args.mapping else None,
        "feature_count": int(bundle.features.shape[1]),
        "asof_trends": bool(args.asof_trends),
        "categorical_features": bundle.categorical_features,
        "cv_seasons": [int(row["season"]) for row in fold_rows],
        "validation_rule": "Train on season < 2024 and validate on season == 2024.",
        "mean_brier": float(metrics_frame["brier"].mean()),
        "mean_brier_skill_score": float(metrics_frame["brier_skill_score"].mean()),
    }

    if len(bundle.test_rows):
        best_rounds = max(
            50,
            int(np.median([model.best_iteration_ for model in fold_models])),
        )
        LOGGER.info("Training final model on all labeled rows: n_estimators=%d", best_rounds)
        final_model = lgb.LGBMClassifier(**model_parameters(args, n_estimators=best_rounds))
        final_model.fit(
            bundle.features.iloc[bundle.train_rows],
            bundle.target.iloc[bundle.train_rows].astype("int8"),
            categorical_feature=bundle.categorical_features,
        )
        probability = final_model.predict_proba(bundle.features.iloc[bundle.test_rows])[:, 1]
        submission = pd.DataFrame(
            {
                ID_COLUMN: bundle.row_ids.iloc[bundle.test_rows].to_numpy(),
                TARGET: np.clip(probability, 1e-5, 1 - 1e-5),
            }
        )
        submission.to_csv(output_dir / "submission.csv", index=False)
        final_model.booster_.save_model(output_dir / "first_model.txt")
        write_feature_importance(
            final_model,
            bundle.features.columns.tolist(),
            output_dir / "feature_importance_final.csv",
        )
        run_summary["final_n_estimators"] = best_rounds
        run_summary["test_rows"] = int(len(bundle.test_rows))
        run_summary["test_prediction_mean"] = float(probability.mean())
    else:
        LOGGER.warning("No test data supplied; submission.csv was not created")

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
