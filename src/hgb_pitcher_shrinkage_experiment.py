from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .hgb_top20_experiment import TOP20_FEATURES
from .hgb_feature_selection_v2_pkg.common import (
    ID_COLUMN,
    TARGET,
    _read_csv,
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

DEFAULT_STRENGTHS = (5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
N_BUCKET_EDGES = (-np.inf, 20, 50, 100, 300, np.inf)
N_BUCKET_LABELS = ("n<20", "20<=n<50", "50<=n<100", "100<=n<300", "n>=300")


def _variant_features(variant: str) -> list[str]:
    if variant == "raw_top20":
        return [
            "asof_pitcher_success_rate" if name == "pitcher_success_shrunk" else name
            for name in TOP20_FEATURES
        ]
    if variant == "c_shrunk_top20":
        return list(TOP20_FEATURES)
    if variant == "c_shrunk_plus_reliability":
        return [*TOP20_FEATURES, "pitcher_asof_reliability"]
    raise ValueError(f"Unknown variant: {variant}")


def _segment_scores(
    *,
    y_true: np.ndarray,
    probability: np.ndarray,
    pitcher_n: pd.Series,
    variant: str,
    shrinkage: float | None,
) -> list[dict]:
    counts = pd.to_numeric(pitcher_n, errors="coerce").fillna(0.0).clip(lower=0.0)
    bucket = pd.cut(
        counts,
        bins=N_BUCKET_EDGES,
        labels=N_BUCKET_LABELS,
        right=False,
    )
    rows: list[dict] = []
    for label in N_BUCKET_LABELS:
        mask = bucket.astype("string").eq(label).to_numpy()
        if not mask.any():
            continue
        p = np.clip(probability[mask], 1e-6, 1 - 1e-6)
        rows.append(
            {
                "variant": variant,
                "shrinkage": shrinkage,
                "pitcher_n_bucket": label,
                "n": int(mask.sum()),
                "target_rate": float(np.mean(y_true[mask])),
                "prediction_mean": float(np.mean(p)),
                "brier": float(brier_score_loss(y_true[mask], p)),
            }
        )
    return rows


def _fit_variant(
    *,
    engineered,
    training_mask: pd.Series,
    validation_mask: pd.Series,
    feature_names: list[str],
    variant: str,
    shrinkage: float | None,
    max_iter: int,
    seed: int,
) -> tuple[dict, np.ndarray, list[dict]]:
    missing = [name for name in feature_names if name not in engineered.features.columns]
    if missing:
        raise ValueError(f"{variant} missing feature(s): {missing}")

    categorical = [
        name for name in engineered.categorical_features if name in feature_names
    ]
    frame = engineered.features.loc[:, feature_names].copy()
    encoded = frequency_encode(
        frame.loc[training_mask],
        frame.loc[validation_mask],
        categorical,
    )
    y_train = engineered.target.loc[training_mask].to_numpy("int8")
    y_valid = engineered.target.loc[validation_mask].to_numpy("int8")
    indices = np.arange(len(feature_names), dtype="int32")

    started = time.perf_counter()
    _, probability, _ = fit_subset(
        encoded.x_train,
        y_train,
        encoded.x_valid,
        y_valid,
        indices,
        max_iter=max_iter,
        seed=seed,
    )
    elapsed = time.perf_counter() - started

    row = score_row(y_valid, probability, variant, len(feature_names))
    row["shrinkage"] = shrinkage
    row["elapsed_seconds"] = elapsed

    segment_rows = _segment_scores(
        y_true=y_valid,
        probability=probability,
        pitcher_n=engineered.features.loc[validation_mask, "asof_pitcher_n"],
        variant=variant,
        shrinkage=shrinkage,
    )
    return row, probability, segment_rows


def _build_engineered(
    *,
    train: pd.DataFrame,
    mapping: pd.DataFrame,
    group_stats,
    context_stats,
    drift_stats,
    shrinkage: float,
    season_trend_min_history: int,
    trackman_shrinkage: float,
    context_smoothing: float,
):
    main = build_main_features(train, shrinkage=shrinkage)
    main = add_season_trend_features(
        main,
        train,
        shrinkage=shrinkage,
        min_history_seasons=season_trend_min_history,
    )
    trackman_features, tm_blocks, tm_categorical = build_trackman_features(
        main,
        mapping,
        group_stats,
        context_stats,
        drift_stats,
        shrinkage_pitches=trackman_shrinkage,
        context_smoothing=context_smoothing,
    )
    return merge_trackman(main, trackman_features, tm_blocks, tm_categorical)


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

    mapping = load_mapping(mapping_path, tuple(args.accepted_mapping_grades))
    cache_base = Path(args.trackman_cache_dir) if args.trackman_cache_dir else output_dir
    group_stats, context_stats, drift_stats = load_or_aggregate_trackman(
        trackman_path,
        cache_base,
        max_rows=args.max_trackman_rows,
        chunksize=args.trackman_chunksize,
        rerun=args.rerun_trackman,
    )

    score_rows: list[dict] = []
    segment_rows: list[dict] = []
    best: dict | None = None
    best_probability: np.ndarray | None = None
    best_engineered = None
    best_validation_mask: pd.Series | None = None

    raw_done = False
    strengths = sorted(set(float(value) for value in args.strengths))
    if any(value <= 0 for value in strengths):
        raise ValueError("All shrinkage strengths must be > 0.")

    for shrinkage in strengths:
        LOGGER.info("Building features with shrinkage m=%g", shrinkage)
        engineered = _build_engineered(
            train=train,
            mapping=mapping,
            group_stats=group_stats,
            context_stats=context_stats,
            drift_stats=drift_stats,
            shrinkage=shrinkage,
            season_trend_min_history=args.season_trend_min_history,
            trackman_shrinkage=args.trackman_shrinkage,
            context_smoothing=args.context_smoothing,
        )

        training_mask = engineered.seasons.astype(int) < args.validation_season
        validation_mask = engineered.seasons.astype(int) == args.validation_season
        if training_mask.sum() == 0 or validation_mask.sum() == 0:
            raise ValueError(
                f"Invalid split: train={int(training_mask.sum())}, "
                f"valid={int(validation_mask.sum())}, "
                f"validation_season={args.validation_season}"
            )

        variants = ["c_shrunk_top20", "c_shrunk_plus_reliability"]
        if not raw_done:
            variants.insert(0, "raw_top20")

        for variant in variants:
            effective_shrinkage = None if variant == "raw_top20" else shrinkage
            LOGGER.info("Training %s (m=%s)", variant, effective_shrinkage)
            row, probability, segments = _fit_variant(
                engineered=engineered,
                training_mask=training_mask,
                validation_mask=validation_mask,
                feature_names=_variant_features(variant),
                variant=variant,
                shrinkage=effective_shrinkage,
                max_iter=args.max_iter,
                seed=args.seed,
            )
            score_rows.append(row)
            segment_rows.extend(segments)

            if best is None or row["brier"] < best["brier"]:
                best = dict(row)
                best_probability = probability.copy()
                best_engineered = engineered
                best_validation_mask = validation_mask.copy()

        raw_done = True

    scores = pd.DataFrame(score_rows).sort_values(
        ["brier", "variant", "shrinkage"],
        na_position="first",
        ignore_index=True,
    )
    segments = pd.DataFrame(segment_rows).sort_values(
        ["pitcher_n_bucket", "brier", "variant", "shrinkage"],
        na_position="first",
        ignore_index=True,
    )
    scores.to_csv(output_dir / "pitcher_shrinkage_scores.csv", index=False)
    segments.to_csv(output_dir / "pitcher_shrinkage_segment_scores.csv", index=False)

    assert best is not None
    assert best_probability is not None
    assert best_engineered is not None
    assert best_validation_mask is not None

    y_valid = best_engineered.target.loc[best_validation_mask].to_numpy("int8")
    predictions = pd.DataFrame(
        {
            ID_COLUMN: best_engineered.row_ids.loc[best_validation_mask].to_numpy(),
            "season": args.validation_season,
            TARGET: y_valid,
            "asof_pitcher_n": best_engineered.features.loc[
                best_validation_mask, "asof_pitcher_n"
            ].to_numpy(),
            "prediction_best": best_probability,
        }
    )
    predictions.to_csv(
        output_dir / "validation_predictions_pitcher_shrinkage_best.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = {
        "validation_season": args.validation_season,
        "fixed_iteration": args.max_iter,
        "strengths": strengths,
        "prior": (
            "leakage-safe season_trend_prior: only seasons strictly before the "
            "prediction season are used"
        ),
        "formula": "(raw_rate * asof_n + prior * m) / (asof_n + m)",
        "variants": {
            "raw_top20": (
                "Top20 with pitcher_success_shrunk replaced by "
                "asof_pitcher_success_rate"
            ),
            "c_shrunk_top20": "Existing Top20 with pitcher_success_shrunk",
            "c_shrunk_plus_reliability": (
                "Existing Top20 plus pitcher_asof_reliability = n / (n + m)"
            ),
        },
        "best": best,
        "n_buckets": list(N_BUCKET_LABELS),
    }
    with (output_dir / "pitcher_shrinkage_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Done. Best=%s m=%s Brier=%.10f",
        best["model"],
        best["shrinkage"],
        best["brier"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate C-style Bayesian pitcher-control shrinkage on the fixed-350 "
            "Top20 HGB model."
        )
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--trackman", required=True)
    parser.add_argument("--mapping", default="resources/pitcher_trackman_mapping.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trackman-cache-dir")
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=list(DEFAULT_STRENGTHS),
    )
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
