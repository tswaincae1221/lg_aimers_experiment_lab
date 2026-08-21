from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import catboost_native_experiment as base
from .catboost_native_experiment_v2 import _gpu_safe_base_params
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


def _score(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(p, dtype="float64")
    rate = float(y.mean())
    baseline = float(rate * (1.0 - rate))
    brier = float(brier_score_loss(y, p))
    score = max(0.0, 100000.0 * (1.0 - brier / baseline)) if baseline > 0 else 0.0
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return {
        "n": int(len(y)),
        "target_rate": rate,
        "prediction_mean": float(p.mean()),
        "prediction_bias": float(p.mean() - rate),
        "baseline_brier": baseline,
        "brier": brier,
        "official_score_like": score,
        "auc": auc,
    }


def _select_iterations_r_only(
    frame: pd.DataFrame,
    categorical: list[str],
    target: pd.Series,
    seasons: pd.Series,
    game_type: pd.Series,
    args: argparse.Namespace,
) -> tuple[int, pd.DataFrame]:
    train_mask = (seasons.astype(int) < args.selection_season) & game_type.eq(args.first_team_value)
    valid_mask = (seasons.astype(int) == args.selection_season) & game_type.eq(args.first_team_value)
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("R-only selection split is empty")

    x_train = frame.loc[train_mask].reset_index(drop=True)
    x_valid = frame.loc[valid_mask].reset_index(drop=True)
    y_train = target.loc[train_mask].to_numpy("int8")
    y_valid = target.loc[valid_mask].to_numpy("int8")

    train_pool = base._pool(x_train, y_train, categorical)
    valid_pool = base._pool(x_valid, y_valid, categorical)
    model = CatBoostClassifier(
        **_gpu_safe_base_params(
            args,
            iterations=args.max_tune_iterations,
            max_ctr_complexity=1,
        )
    )

    LOGGER.info(
        "R-only iteration tuning: train<%d %s, validate %d %s, n_train=%d n_valid=%d features=%d",
        args.selection_season,
        args.first_team_value,
        args.selection_season,
        args.first_team_value,
        int(train_mask.sum()),
        int(valid_mask.sum()),
        frame.shape[1],
    )
    start = time.perf_counter()
    model.fit(train_pool, eval_set=valid_pool, use_best_model=False)
    elapsed = time.perf_counter() - start

    step = args.coarse_step
    coarse = list(range(args.min_select_iterations, args.max_tune_iterations + 1, step))
    if coarse[-1] != args.max_tune_iterations:
        coarse.append(args.max_tune_iterations)

    rows: list[dict] = []
    for it in coarse:
        p = model.predict_proba(valid_pool, ntree_end=int(it))[:, 1]
        rows.append({"iteration": int(it), "selection_brier": float(brier_score_loss(y_valid, p)), "stage": "coarse"})

    cdf = pd.DataFrame(rows)
    coarse_best = int(cdf.loc[cdf["selection_brier"].idxmin(), "iteration"])
    lo = max(args.min_select_iterations, coarse_best - step + 1)
    hi = min(args.max_tune_iterations, coarse_best + step - 1)
    existing = set(cdf["iteration"].astype(int))
    for it in range(lo, hi + 1):
        if it in existing:
            continue
        p = model.predict_proba(valid_pool, ntree_end=int(it))[:, 1]
        rows.append({"iteration": int(it), "selection_brier": float(brier_score_loss(y_valid, p)), "stage": "refine"})

    curve = pd.DataFrame(rows).sort_values("iteration", ignore_index=True)
    best_idx = curve["selection_brier"].idxmin()
    selected = int(curve.loc[best_idx, "iteration"])
    curve["selected"] = curve["iteration"].eq(selected)
    curve["train_elapsed_seconds"] = float(elapsed)
    LOGGER.info("R-only selected iterations=%d, 2023 R Brier=%.9f", selected, float(curve.loc[best_idx, "selection_brier"]))
    return selected, curve


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = resolve_input_path(args.train, "train.csv")
    trackman_path = resolve_input_path(args.trackman, "trackman_history.csv")
    mapping_path = Path(args.mapping)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    train = _read_csv(train_path)
    train = sample_by_season(train, args.max_rows_per_season, args.seed)

    if "game_type" not in train.columns:
        raise ValueError("train.csv has no game_type")
    raw_game_type_counts = train["game_type"].astype("string").value_counts(dropna=False).to_dict()
    LOGGER.info("game_type counts: %s", raw_game_type_counts)

    main = build_main_features(train, shrinkage=args.shrinkage)
    main = add_season_trend_features(
        main,
        train,
        shrinkage=args.shrinkage,
        min_history_seasons=args.season_trend_min_history,
    )
    mapping = load_mapping(mapping_path, tuple(args.accepted_mapping_grades))
    cache_base = Path(args.trackman_cache_dir) if args.trackman_cache_dir else out
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
    features = [c for c in engineered.features.columns if c not in recent]
    if len(features) != args.expected_104_count:
        raise ValueError(f"Expected {args.expected_104_count} features, got {len(features)}")

    game_type = pd.Series(
        train["game_type"].astype("string").to_numpy(),
        index=engineered.features.index,
        dtype="string",
    )

    frame, categorical = base._prepare_catboost_frame(
        engineered,
        train,
        features,
        add_player_ids=False,
    )

    if args.drop_game_type_feature and "game_type" in frame.columns:
        frame = frame.drop(columns=["game_type"])
        categorical = [c for c in categorical if c != "game_type"]

    selected_iterations, curve = _select_iterations_r_only(
        frame,
        categorical,
        engineered.target,
        engineered.seasons,
        game_type,
        args,
    )
    curve.to_csv(out / "r_only_iteration_selection_2023.csv", index=False)

    final_train_mask = (engineered.seasons.astype(int) < args.validation_season) & game_type.eq(args.first_team_value)
    valid_all = engineered.seasons.astype(int) == args.validation_season

    x_train = frame.loc[final_train_mask].reset_index(drop=True)
    y_train = engineered.target.loc[final_train_mask].to_numpy("int8")
    x_valid = frame.loc[valid_all].reset_index(drop=True)
    y_valid = engineered.target.loc[valid_all].to_numpy("int8")

    model = CatBoostClassifier(
        **_gpu_safe_base_params(
            args,
            iterations=selected_iterations,
            max_ctr_complexity=1,
        )
    )
    LOGGER.info(
        "Fit R-only final model: train seasons<%d & game_type=%s, n=%d; validate all %d n=%d",
        args.validation_season,
        args.first_team_value,
        int(final_train_mask.sum()),
        args.validation_season,
        int(valid_all.sum()),
    )
    start = time.perf_counter()
    model.fit(base._pool(x_train, y_train, categorical))
    p_all = model.predict_proba(x_valid)[:, 1]
    elapsed = time.perf_counter() - start

    result = pd.DataFrame({
        ID_COLUMN: engineered.row_ids.loc[valid_all].to_numpy(),
        "season": args.validation_season,
        "game_type": game_type.loc[valid_all].astype(str).to_numpy(),
        TARGET: y_valid,
        "prediction_r_only": p_all,
    })
    result.to_csv(out / "r_only_validation_predictions.csv.gz", index=False, compression="gzip")

    scores = []
    for slice_name, mask in [
        ("2024_all", np.ones(len(result), dtype=bool)),
        (f"2024_{args.first_team_value}", result["game_type"].eq(args.first_team_value).to_numpy()),
        (f"2024_{args.second_team_value}", result["game_type"].eq(args.second_team_value).to_numpy()),
    ]:
        if not mask.any():
            continue
        row = _score(result.loc[mask, TARGET].to_numpy(), result.loc[mask, "prediction_r_only"].to_numpy())
        row.update({"model": "cat104_r_only", "slice": slice_name, "iterations": selected_iterations, "elapsed_seconds": elapsed})
        scores.append(row)

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(out / "r_only_scores.csv", index=False)

    importance = pd.DataFrame({
        "feature": frame.columns,
        "importance": model.get_feature_importance(type="PredictionValuesChange"),
    }).sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(out / "r_only_feature_importance.csv", index=False)

    baseline_path = Path(args.baseline_predictions) if args.baseline_predictions else None
    comparison = None
    if baseline_path is not None and baseline_path.is_file():
        base_pred = pd.read_csv(baseline_path)
        baseline_col = args.baseline_prediction_column
        if baseline_col not in base_pred.columns:
            raise ValueError(f"Baseline column {baseline_col} not found in {baseline_path}")
        merged = result.merge(base_pred[[ID_COLUMN, baseline_col]], on=ID_COLUMN, how="inner", validate="one_to_one")
        comp_rows = []
        for slice_name, mask in [
            ("2024_all", np.ones(len(merged), dtype=bool)),
            (f"2024_{args.first_team_value}", merged["game_type"].eq(args.first_team_value).to_numpy()),
            (f"2024_{args.second_team_value}", merged["game_type"].eq(args.second_team_value).to_numpy()),
        ]:
            if not mask.any():
                continue
            y = merged.loc[mask, TARGET].to_numpy()
            new = _score(y, merged.loc[mask, "prediction_r_only"].to_numpy())
            old = _score(y, merged.loc[mask, baseline_col].to_numpy())
            comp_rows.append({
                "slice": slice_name,
                "n": int(mask.sum()),
                "baseline_brier": old["brier"],
                "r_only_brier": new["brier"],
                "delta_brier_r_only_minus_baseline": new["brier"] - old["brier"],
                "baseline_score_like": old["official_score_like"],
                "r_only_score_like": new["official_score_like"],
                "delta_score_like": new["official_score_like"] - old["official_score_like"],
                "baseline_auc": old["auc"],
                "r_only_auc": new["auc"],
                "baseline_pred_mean": old["prediction_mean"],
                "r_only_pred_mean": new["prediction_mean"],
                "target_rate": new["target_rate"],
            })
        comparison = pd.DataFrame(comp_rows)
        comparison.to_csv(out / "r_only_vs_full_cat104_comparison.csv", index=False)

    summary = {
        "design": {
            "first_team_value": args.first_team_value,
            "second_team_value": args.second_team_value,
            "game_type_counts": {str(k): int(v) for k, v in raw_game_type_counts.items()},
            "feature_count": int(frame.shape[1]),
            "categorical_count": int(len(categorical)),
            "player_ids": False,
            "max_ctr_complexity": 1,
            "selection": f"train < {args.selection_season} & {args.first_team_value}; validate {args.selection_season} & {args.first_team_value}",
            "selected_iterations": selected_iterations,
            "final_train": f"season < {args.validation_season} & game_type={args.first_team_value}",
            "final_validation": f"season={args.validation_season}, scored all/{args.first_team_value}/{args.second_team_value}",
            "important_note": "Official asof_* features are kept as provided; this experiment removes F rows from model fitting but does not recompute official asof histories by game_type.",
        },
        "scores": scores_df.to_dict(orient="records"),
        "comparison": None if comparison is None else comparison.to_dict(orient="records"),
    }
    with (out / "r_only_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("\n%s", scores_df.to_string(index=False))
    if comparison is not None:
        LOGGER.info("\nComparison with full-data Cat104:\n%s", comparison.to_string(index=False))

    del model, frame, x_train, x_valid
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CatBoost-104 on first-team (R) rows only and compare against full-data CatBoost.")
    p.add_argument("--train", required=True)
    p.add_argument("--trackman", required=True)
    p.add_argument("--mapping", default="resources/pitcher_trackman_mapping.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trackman-cache-dir")
    p.add_argument("--baseline-predictions")
    p.add_argument("--baseline-prediction-column", default="cat104_no_player_ids_ctr1")
    p.add_argument("--first-team-value", default="R")
    p.add_argument("--second-team-value", default="F")
    p.add_argument("--drop-game-type-feature", action="store_true")
    p.add_argument("--selection-season", type=int, default=2023)
    p.add_argument("--validation-season", type=int, default=2024)
    p.add_argument("--max-tune-iterations", type=int, default=1000)
    p.add_argument("--min-select-iterations", type=int, default=100)
    p.add_argument("--coarse-step", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--l2-leaf-reg", type=float, default=5.0)
    p.add_argument("--random-strength", type=float, default=0.5)
    p.add_argument("--border-count", type=int, default=128)
    p.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--gpu-ram-part", type=float, default=0.85)
    p.add_argument("--thread-count", type=int, default=-1)
    p.add_argument("--verbose", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shrinkage", type=float, default=50.0)
    p.add_argument("--season-trend-min-history", type=int, default=4)
    p.add_argument("--trackman-shrinkage", type=float, default=100.0)
    p.add_argument("--context-smoothing", type=float, default=50.0)
    p.add_argument("--trackman-chunksize", type=int, default=250000)
    p.add_argument("--accepted-mapping-grades", nargs="+", default=["확정", "높음"])
    p.add_argument("--max-rows-per-season", type=int)
    p.add_argument("--max-trackman-rows", type=int)
    p.add_argument("--rerun-trackman", action="store_true")
    p.add_argument("--expected-104-count", type=int, default=104)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
