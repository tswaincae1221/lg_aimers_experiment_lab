from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, roc_auc_score

from .hgb_feature_selection_v2_pkg.common import (
    ID_COLUMN,
    TARGET,
    _read_csv,
    build_main_features,
    resolve_input_path,
    sample_by_season,
)
from .hgb_feature_selection_v2_pkg.season_trend import add_season_trend_features
from .hgb_feature_selection_v2_pkg.trackman import (
    build_trackman_features,
    load_mapping,
    load_or_aggregate_trackman,
    merge_trackman,
)

LOGGER = logging.getLogger(__name__)

RAW_PLAYER_CATEGORICAL = ("pitcher_id", "batter_id")


def official_score(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(probability, dtype="float64")
    target_rate = float(y.mean())
    baseline_brier = float(target_rate * (1.0 - target_rate))
    brier = float(brier_score_loss(y, p))
    score = max(0.0, 100000.0 * (1.0 - brier / baseline_brier)) if baseline_brier > 0 else 0.0
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return {
        "n": int(len(y)),
        "target_rate": target_rate,
        "prediction_mean": float(p.mean()),
        "mean_prediction_bias": float(p.mean() - target_rate),
        "baseline_brier": baseline_brier,
        "brier": brier,
        "official_score": score,
        "auc": auc,
    }


def _prepare_catboost_frame(
    engineered,
    raw_train: pd.DataFrame,
    features: list[str],
    *,
    add_player_ids: bool,
) -> tuple[pd.DataFrame, list[str]]:
    frame = engineered.features.loc[:, features].copy()
    categorical = [f for f in engineered.categorical_features if f in frame.columns]

    if add_player_ids:
        for column in RAW_PLAYER_CATEGORICAL:
            if column not in raw_train.columns:
                raise ValueError(f"raw train is missing {column}")
            frame[column] = (
                raw_train[column]
                .astype("string")
                .str.replace(r"\\.0$", "", regex=True)
                .fillna("__MISSING__")
                .astype(str)
            )
            categorical.append(column)

    categorical = list(dict.fromkeys(categorical))
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    for column in frame.columns:
        if column not in categorical:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return frame, categorical


def _pool(frame: pd.DataFrame, target: pd.Series | np.ndarray, categorical: list[str]) -> Pool:
    cat_idx = [frame.columns.get_loc(column) for column in categorical]
    return Pool(frame, label=np.asarray(target, dtype="int8"), cat_features=cat_idx)


def _base_params(args: argparse.Namespace, *, iterations: int, max_ctr_complexity: int) -> dict:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "iterations": int(iterations),
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "random_seed": args.seed,
        "max_ctr_complexity": int(max_ctr_complexity),
        "border_count": args.border_count,
        "allow_writing_files": False,
        "verbose": args.verbose,
        "thread_count": args.thread_count,
    }
    if args.task_type.upper() == "GPU":
        params.update(
            {
                "task_type": "GPU",
                "devices": args.devices,
                "gpu_ram_part": args.gpu_ram_part,
            }
        )
    else:
        params["task_type"] = "CPU"
    return params


def _select_iterations(
    frame: pd.DataFrame,
    categorical: list[str],
    target: pd.Series,
    seasons: pd.Series,
    args: argparse.Namespace,
) -> tuple[int, pd.DataFrame]:
    train_mask = seasons.astype(int) < args.selection_season
    valid_mask = seasons.astype(int) == args.selection_season
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("selection split is empty")

    x_train = frame.loc[train_mask].reset_index(drop=True)
    x_valid = frame.loc[valid_mask].reset_index(drop=True)
    y_train = target.loc[train_mask].to_numpy("int8")
    y_valid = target.loc[valid_mask].to_numpy("int8")

    model = CatBoostClassifier(
        **_base_params(
            args,
            iterations=args.max_tune_iterations,
            max_ctr_complexity=args.anchor_max_ctr_complexity,
        )
    )
    LOGGER.info(
        "Selecting CatBoost iterations on seasons < %d -> %d using %d features",
        args.selection_season,
        args.selection_season,
        frame.shape[1],
    )
    start = time.perf_counter()
    model.fit(
        _pool(x_train, y_train, categorical),
        eval_set=_pool(x_valid, y_valid, categorical),
        use_best_model=False,
    )
    elapsed = time.perf_counter() - start

    evals = model.get_evals_result()
    validation = evals.get("validation") or evals.get("validation_0")
    if not validation or "BrierScore" not in validation:
        raise RuntimeError(f"CatBoost eval history does not contain BrierScore: {evals.keys()}")
    values = np.asarray(validation["BrierScore"], dtype="float64")
    iterations = np.arange(1, len(values) + 1, dtype=int)
    curve = pd.DataFrame({"iteration": iterations, "selection_brier": values})
    curve["eligible"] = curve["iteration"] >= args.min_select_iterations
    eligible = curve.loc[curve["eligible"]]
    if eligible.empty:
        raise ValueError("min_select_iterations exceeds max_tune_iterations")
    best_row = eligible.loc[eligible["selection_brier"].idxmin()]
    selected = int(best_row["iteration"])
    LOGGER.info(
        "Selected %d iterations; 2023 Brier %.9f; tuning %.1fs",
        selected,
        float(best_row["selection_brier"]),
        elapsed,
    )
    curve["selected"] = curve["iteration"].eq(selected)
    curve.attrs["elapsed_seconds"] = elapsed
    return selected, curve


def _fit_evaluate_variant(
    engineered,
    raw_train: pd.DataFrame,
    features: list[str],
    *,
    add_player_ids: bool,
    max_ctr_complexity: int,
    iterations: int,
    validation_season: int,
    label: str,
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    frame, categorical = _prepare_catboost_frame(
        engineered,
        raw_train,
        features,
        add_player_ids=add_player_ids,
    )
    train_mask = engineered.seasons.astype(int) < validation_season
    valid_mask = engineered.seasons.astype(int) == validation_season

    x_train = frame.loc[train_mask].reset_index(drop=True)
    x_valid = frame.loc[valid_mask].reset_index(drop=True)
    y_train = engineered.target.loc[train_mask].to_numpy("int8")
    y_valid = engineered.target.loc[valid_mask].to_numpy("int8")

    model = CatBoostClassifier(
        **_base_params(
            args,
            iterations=iterations,
            max_ctr_complexity=max_ctr_complexity,
        )
    )
    LOGGER.info(
        "Fitting %s: train<%d valid=%d features=%d cats=%d ctr=%d iters=%d",
        label,
        validation_season,
        validation_season,
        frame.shape[1],
        len(categorical),
        max_ctr_complexity,
        iterations,
    )
    start = time.perf_counter()
    model.fit(_pool(x_train, y_train, categorical))
    probability = model.predict_proba(x_valid)[:, 1]
    elapsed = time.perf_counter() - start

    score = official_score(y_valid, probability)
    score.update(
        {
            "model": label,
            "feature_count": int(frame.shape[1]),
            "categorical_count": int(len(categorical)),
            "add_player_ids": bool(add_player_ids),
            "max_ctr_complexity": int(max_ctr_complexity),
            "iterations": int(iterations),
            "elapsed_seconds": float(elapsed),
        }
    )

    importance = pd.DataFrame(
        {
            "feature": frame.columns,
            "importance": model.get_feature_importance(type="PredictionValuesChange"),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    del model, x_train, x_valid, frame
    gc.collect()
    return score, probability, importance


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_input_path(args.train, "train.csv")
    trackman_path = resolve_input_path(args.trackman, "trackman_history.csv")
    mapping_path = Path(args.mapping)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    train = _read_csv(train_path)
    train = sample_by_season(train, args.max_rows_per_season, args.seed)

    main = build_main_features(train, shrinkage=args.shrinkage)
    main = add_season_trend_features(
        main,
        train,
        shrinkage=args.shrinkage,
        min_history_seasons=args.season_trend_min_history,
    )

    mapping = load_mapping(mapping_path, tuple(args.accepted_mapping_grades))
    cache_base = Path(args.trackman_cache_dir) if args.trackman_cache_dir else output_dir
    group_stats, context_stats, drift_stats = load_or_aggregate_trackman(
        trackman_path,
        cache_base,
        max_rows=args.max_trackman_rows,
        chunksize=args.trackman_chunksize,
        rerun=args.rerun_trackman,
    )
    tm_features, tm_blocks, tm_categorical = build_trackman_features(
        main,
        mapping,
        group_stats,
        context_stats,
        drift_stats,
        shrinkage_pitches=args.trackman_shrinkage,
        context_smoothing=args.context_smoothing,
    )
    engineered = merge_trackman(main, tm_features, tm_blocks, tm_categorical)
    if "asof_pitcher_pitchmix_n" in engineered.features.columns:
        engineered.features.drop(columns=["asof_pitcher_pitchmix_n"], inplace=True)

    recent = set(engineered.blocks.get("recent_form", []))
    features_104 = [column for column in engineered.features.columns if column not in recent]
    features_118 = list(engineered.features.columns)
    if len(features_104) != args.expected_104_count:
        raise ValueError(f"Expected 104 features, got {len(features_104)}")
    if len(features_118) != args.expected_118_count:
        raise ValueError(f"Expected 118 features, got {len(features_118)}")

    trackman_features = {
        feature
        for block, columns in engineered.blocks.items()
        if block.startswith("trackman_")
        for feature in columns
    }
    features_104_no_tm = [f for f in features_104 if f not in trackman_features]

    anchor_frame, anchor_categorical = _prepare_catboost_frame(
        engineered,
        train,
        features_104,
        add_player_ids=True,
    )
    selected_iterations, curve = _select_iterations(
        anchor_frame,
        anchor_categorical,
        engineered.target,
        engineered.seasons,
        args,
    )
    curve.to_csv(output_dir / "catboost_iteration_selection_2023.csv", index=False)
    del anchor_frame
    gc.collect()

    variants = [
        {
            "label": "cat104_no_player_ids_ctr1",
            "features": features_104,
            "add_player_ids": False,
            "max_ctr_complexity": 1,
        },
        {
            "label": "cat104_player_ids_ctr1",
            "features": features_104,
            "add_player_ids": True,
            "max_ctr_complexity": 1,
        },
        {
            "label": "cat104_player_ids_ctr2",
            "features": features_104,
            "add_player_ids": True,
            "max_ctr_complexity": 2,
        },
        {
            "label": "cat118_player_ids_ctr2",
            "features": features_118,
            "add_player_ids": True,
            "max_ctr_complexity": 2,
        },
        {
            "label": "cat104_player_ids_no_trackman_ctr2",
            "features": features_104_no_tm,
            "add_player_ids": True,
            "max_ctr_complexity": 2,
        },
    ]

    valid_mask = engineered.seasons.astype(int) == args.validation_season
    prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: engineered.row_ids.loc[valid_mask].to_numpy(),
            "season": args.validation_season,
            TARGET: engineered.target.loc[valid_mask].to_numpy("int8"),
        }
    )

    scores: list[dict] = []
    for variant in variants:
        score, probability, importance = _fit_evaluate_variant(
            engineered,
            train,
            variant["features"],
            add_player_ids=variant["add_player_ids"],
            max_ctr_complexity=variant["max_ctr_complexity"],
            iterations=selected_iterations,
            validation_season=args.validation_season,
            label=variant["label"],
            args=args,
        )
        scores.append(score)
        prediction_frame[variant["label"]] = probability
        importance.to_csv(output_dir / f"importance_{variant['label']}.csv", index=False)
        with (output_dir / f"features_{variant['label']}.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(variant["features"]) + "\n")
        del probability, importance
        gc.collect()

    scores_df = pd.DataFrame(scores).sort_values("brier", ignore_index=True)
    scores_df["delta_brier_vs_hgb104"] = scores_df["brier"] - args.hgb104_reference_brier
    # HGB reference score is computed with the same validation target-rate baseline.
    baseline_brier = float(scores_df.iloc[0]["baseline_brier"])
    hgb_reference_score = max(
        0.0,
        100000.0 * (1.0 - args.hgb104_reference_brier / baseline_brier),
    )
    scores_df["delta_score_vs_hgb104"] = scores_df["official_score"] - hgb_reference_score
    scores_df.to_csv(output_dir / "catboost_model_scores.csv", index=False)
    prediction_frame.to_csv(
        output_dir / "catboost_validation_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = {
        "design": {
            "selection_train_seasons": f"< {args.selection_season}",
            "selection_validation_season": args.selection_season,
            "selection_anchor": "104 features + raw pitcher_id/batter_id + CTR complexity 1",
            "selected_iterations": selected_iterations,
            "min_select_iterations": args.min_select_iterations,
            "max_tune_iterations": args.max_tune_iterations,
            "final_train_seasons": f"< {args.validation_season}",
            "final_validation_season": args.validation_season,
            "catboost_native_categorical": True,
            "hgb104_reference_brier": args.hgb104_reference_brier,
            "hgb104_reference_score_on_same_baseline": hgb_reference_score,
            "variants": [v["label"] for v in variants],
        },
        "scores": scores_df.to_dict(orient="records"),
    }
    with (output_dir / "catboost_experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    LOGGER.info("\n%s", scores_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native-categorical CatBoost comparison for LG Aimers control success.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--trackman", required=True)
    parser.add_argument("--mapping", default="resources/pitcher_trackman_mapping.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trackman-cache-dir")
    parser.add_argument("--selection-season", type=int, default=2023)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--max-tune-iterations", type=int, default=1600)
    parser.add_argument("--min-select-iterations", type=int, default=200)
    parser.add_argument("--anchor-max-ctr-complexity", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--random-strength", type=float, default=0.5)
    parser.add_argument("--border-count", type=int, default=128)
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--gpu-ram-part", type=float, default=0.85)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shrinkage", type=float, default=50.0)
    parser.add_argument("--season-trend-min-history", type=int, default=4)
    parser.add_argument("--trackman-shrinkage", type=float, default=100.0)
    parser.add_argument("--context-smoothing", type=float, default=50.0)
    parser.add_argument("--trackman-chunksize", type=int, default=250000)
    parser.add_argument("--accepted-mapping-grades", nargs="+", default=["확정", "높음"])
    parser.add_argument("--max-rows-per-season", type=int)
    parser.add_argument("--max-trackman-rows", type=int)
    parser.add_argument("--rerun-trackman", action="store_true")
    parser.add_argument("--expected-104-count", type=int, default=104)
    parser.add_argument("--expected-118-count", type=int, default=118)
    parser.add_argument("--hgb104-reference-brier", type=float, default=0.24801181765968383)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
