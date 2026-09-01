from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .hgb_feature_selection_v2_pkg.common import (
    ID_COLUMN,
    TARGET,
    _file_signature,
    _read_csv,
    _stable_hash,
    build_main_features,
    resolve_input_path,
    sample_by_season,
)
from .hgb_feature_selection_v2_pkg.season_trend import add_season_trend_features
from .hgb_feature_selection_v2_pkg.selection import (
    fit_subset,
    frequency_encode,
    score_row,
)
from .hgb_feature_selection_v2_pkg.trackman import (
    build_trackman_features,
    load_mapping,
    load_or_aggregate_trackman,
    merge_trackman,
)

LOGGER = logging.getLogger(__name__)

TOP20_FEATURES = [
    "hand_x_fastball_rate",
    "game_type",
    "success_interact",
    "threat_x_reverse",
    "batter_team_id",
    "hand_pair",
    "tm_drift_game_n_log1p",
    "asof_pitcher_ball_rate",
    "pitcher_success_shrunk",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "pitcher_team_id",
    "asof_pitcher_prev1_game_success_rate",
    "asof_batter_n",
    "three_ball_x_middle",
    "asof_pitcher_reverse_rate",
    "tm_expected_velocity_dispersion",
    "tm_expected_movement_dispersion",
    "count_diff_x_success",
    "asof_pitcher_n",
]


def _read_reference_full(reference_dir: Path | None) -> pd.DataFrame:
    if reference_dir is None:
        return pd.DataFrame()
    path = reference_dir / "model_scores.csv"
    if not path.is_file():
        LOGGER.warning("Reference model_scores.csv not found: %s", path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "model" not in frame.columns:
        return pd.DataFrame()
    full = frame.loc[frame["model"].astype("string").eq("full")].copy()
    if full.empty:
        return pd.DataFrame()
    full["model"] = "reference_full_118"
    return full.head(1)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_input_path(args.train, "train.csv")
    trackman_path = resolve_input_path(args.trackman, "trackman_history.csv")
    mapping_path = Path(args.mapping)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    LOGGER.info("Reading train: %s", train_path)
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
    LOGGER.info("Trackman cache base: %s", cache_base)
    group_stats, context_stats, drift_stats = load_or_aggregate_trackman(
        trackman_path,
        cache_base,
        max_rows=args.max_trackman_rows,
        chunksize=args.trackman_chunksize,
        rerun=args.rerun_trackman,
    )
    trackman_features, tm_blocks, tm_categorical = build_trackman_features(
        main,
        mapping,
        group_stats,
        context_stats,
        drift_stats,
        shrinkage_pitches=args.trackman_shrinkage,
        context_smoothing=args.context_smoothing,
    )
    engineered = merge_trackman(main, trackman_features, tm_blocks, tm_categorical)

    missing = [name for name in TOP20_FEATURES if name not in engineered.features.columns]
    if missing:
        raise ValueError(f"Top20 feature(s) missing from engineered data: {missing}")

    feature_frame = engineered.features.loc[:, TOP20_FEATURES].copy()
    categorical = [
        name for name in engineered.categorical_features if name in TOP20_FEATURES
    ]

    training_mask = engineered.seasons.astype(int) < args.validation_season
    validation_mask = engineered.seasons.astype(int) == args.validation_season
    if training_mask.sum() == 0 or validation_mask.sum() == 0:
        raise ValueError(
            f"Invalid split: train={int(training_mask.sum())}, "
            f"valid={int(validation_mask.sum())}, "
            f"validation_season={args.validation_season}"
        )

    encoded = frequency_encode(
        feature_frame.loc[training_mask],
        feature_frame.loc[validation_mask],
        categorical,
    )
    y_train = engineered.target.loc[training_mask].to_numpy("int8")
    y_valid = engineered.target.loc[validation_mask].to_numpy("int8")
    indices = np.arange(len(TOP20_FEATURES), dtype="int32")

    LOGGER.info(
        "Training Top20 HGB: rows=%d features=%d iter=350",
        len(y_train),
        len(indices),
    )
    start = time.perf_counter()
    model, probability, brier = fit_subset(
        encoded.x_train,
        y_train,
        encoded.x_valid,
        y_valid,
        indices,
        max_iter=350,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start

    score = score_row(y_valid, probability, "top20_perm_fixed350", len(TOP20_FEATURES))
    score["elapsed_seconds"] = elapsed
    top20_score = pd.DataFrame([score])

    reference = _read_reference_full(
        Path(args.reference_results_dir) if args.reference_results_dir else None
    )
    comparison = pd.concat([reference, top20_score], ignore_index=True, sort=False)
    comparison.to_csv(output_dir / "model_scores_top20.csv", index=False)

    with (output_dir / "top20_features.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(TOP20_FEATURES) + "\n")

    predictions = pd.DataFrame(
        {
            ID_COLUMN: engineered.row_ids.loc[validation_mask].to_numpy(),
            "season": args.validation_season,
            TARGET: y_valid,
            "prediction_top20": probability,
        }
    )
    predictions.to_csv(
        output_dir / "validation_predictions_top20.csv.gz",
        index=False,
        compression="gzip",
    )

    joblib.dump(
        {
            "model": model,
            "features": TOP20_FEATURES,
            "encoder_state": encoded.encoder_state,
            "fixed_iteration": 350,
            "validation_season": args.validation_season,
            "ranking_source": "2024 holdout permutation importance from fixed-350 full model",
        },
        output_dir / "hgb_top20_fixed350.joblib",
    )

    summary = {
        "run_signature": _stable_hash(
            {
                "pipeline_version": "top20-v1",
                "train": _file_signature(train_path),
                "trackman": _file_signature(trackman_path),
                "mapping": _file_signature(mapping_path),
                "features": TOP20_FEATURES,
                "validation_season": args.validation_season,
                "fixed_iteration": 350,
            }
        ),
        "train_rows": int(training_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_season": args.validation_season,
        "fixed_iteration": 350,
        "feature_count": 20,
        "features": TOP20_FEATURES,
        "brier": float(brier),
        "ranking_source": "fixed-350 full-model permutation importance, 75k-row 2024 sample",
        "season_trend_note": (
            "season_trend_prior is not a direct Top20 column because it is constant "
            "within the 2024 holdout, but it still affects pitcher_success_shrunk."
        ),
    }
    with (output_dir / "run_summary_top20.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    LOGGER.info("Top20 done: Brier=%.8f", brier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-350 HGB using the exact Top20 permutation-importance features."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--trackman", required=True)
    parser.add_argument("--mapping", default="resources/pitcher_trackman_mapping.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-results-dir")
    parser.add_argument(
        "--trackman-cache-dir",
        help=(
            "Optional existing V2 full-results directory whose cache/ folder should "
            "be reused, avoiding Trackman re-aggregation."
        ),
    )
    parser.add_argument("--validation-season", type=int, default=2024)
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
