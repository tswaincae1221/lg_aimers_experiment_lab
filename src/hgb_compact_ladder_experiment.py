from __future__ import annotations

import argparse
import gc
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
    _read_csv,
    build_main_features,
    resolve_input_path,
    sample_by_season,
)
from .hgb_feature_selection_v2_pkg.season_trend import add_season_trend_features
from .hgb_feature_selection_v2_pkg.selection import brier, fit_subset, frequency_encode, score_row
from .hgb_feature_selection_v2_pkg.trackman import (
    build_trackman_features,
    load_mapping,
    load_or_aggregate_trackman,
    merge_trackman,
)

LOGGER = logging.getLogger(__name__)

TARGET_SIZES = (80, 60, 40)
PROTECTED_FEATURES = {"season", "season_trend_prior"}


def _feature_to_block(engineered, features: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    feature_set = set(features)
    for block, columns in engineered.blocks.items():
        for feature in columns:
            if feature in feature_set and feature not in result:
                result[feature] = block
    return {feature: result.get(feature, "unassigned") for feature in features}


def _permutation_importance(
    model,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    *,
    sample_size: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if sample_size < len(y_valid):
        sample_idx = np.sort(rng.choice(len(y_valid), size=sample_size, replace=False))
        x = x_valid[sample_idx].copy()
        y = y_valid[sample_idx]
    else:
        x = x_valid.copy()
        y = y_valid

    baseline = brier(y, model.predict_proba(x)[:, 1])
    rows: list[dict] = []
    for j, feature in enumerate(feature_names):
        original = x[:, j].copy()
        deltas: list[float] = []
        for _ in range(repeats):
            x[:, j] = rng.permutation(original)
            deltas.append(brier(y, model.predict_proba(x)[:, 1]) - baseline)
        x[:, j] = original
        mean = float(np.mean(deltas))
        std = float(np.std(deltas, ddof=0))
        rows.append(
            {
                "feature": feature,
                "permutation_delta_brier_mean": mean,
                "permutation_delta_brier_std": std,
                "robust_importance": mean - std,
                "sample_n": len(y),
                "repeats": repeats,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["robust_importance", "permutation_delta_brier_mean", "feature"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def _fit_selection_stage(
    engineered,
    features: list[str],
    *,
    selection_season: int,
    max_iter: int,
    sample_size: int,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    train_mask = engineered.seasons.astype(int) < selection_season
    valid_mask = engineered.seasons.astype(int) == selection_season
    categorical = [f for f in engineered.categorical_features if f in features]
    encoded = frequency_encode(
        engineered.features.loc[train_mask, features],
        engineered.features.loc[valid_mask, features],
        categorical,
    )
    y_train = engineered.target.loc[train_mask].to_numpy("int8")
    y_valid = engineered.target.loc[valid_mask].to_numpy("int8")
    idx = np.arange(len(features), dtype="int32")
    model, _, stage_brier = fit_subset(
        encoded.x_train,
        y_train,
        encoded.x_valid,
        y_valid,
        idx,
        max_iter=max_iter,
        seed=seed,
    )
    importance = _permutation_importance(
        model,
        encoded.x_valid,
        y_valid,
        features,
        sample_size=sample_size,
        repeats=repeats,
        seed=seed,
    )
    del encoded, model, y_train, y_valid
    gc.collect()
    return importance, stage_brier


def _prune_to_target(
    current: list[str],
    importance: pd.DataFrame,
    feature_block: dict[str, str],
    target_size: int,
) -> tuple[list[str], pd.DataFrame]:
    if target_size >= len(current):
        return list(current), pd.DataFrame()

    counts = pd.Series([feature_block[f] for f in current]).value_counts().to_dict()
    floors = {block: 1 for block in counts}

    imp = importance.set_index("feature")
    removal_order = sorted(
        current,
        key=lambda f: (
            float(imp.loc[f, "robust_importance"]),
            float(imp.loc[f, "permutation_delta_brier_mean"]),
            f,
        ),
    )

    keep = set(current)
    removed_rows: list[dict] = []
    for feature in removal_order:
        if len(keep) <= target_size:
            break
        if feature in PROTECTED_FEATURES:
            continue
        block = feature_block[feature]
        if counts[block] <= floors[block]:
            continue
        keep.remove(feature)
        counts[block] -= 1
        row = imp.loc[feature].to_dict()
        row.update({"feature": feature, "block": block, "reason": "lowest robust importance"})
        removed_rows.append(row)

    if len(keep) != target_size:
        raise RuntimeError(
            f"Could not prune from {len(current)} to {target_size} with block floors/protection; "
            f"stopped at {len(keep)}."
        )
    selected = [feature for feature in current if feature in keep]
    return selected, pd.DataFrame(removed_rows)


def _evaluate_on_2024(
    engineered,
    features: list[str],
    *,
    validation_season: int,
    max_iter: int,
    seed: int,
    label: str,
) -> tuple[dict, np.ndarray, dict, object]:
    train_mask = engineered.seasons.astype(int) < validation_season
    valid_mask = engineered.seasons.astype(int) == validation_season
    categorical = [f for f in engineered.categorical_features if f in features]
    encoded = frequency_encode(
        engineered.features.loc[train_mask, features],
        engineered.features.loc[valid_mask, features],
        categorical,
    )
    y_train = engineered.target.loc[train_mask].to_numpy("int8")
    y_valid = engineered.target.loc[valid_mask].to_numpy("int8")
    idx = np.arange(len(features), dtype="int32")
    start = time.perf_counter()
    model, probability, _ = fit_subset(
        encoded.x_train,
        y_train,
        encoded.x_valid,
        y_valid,
        idx,
        max_iter=max_iter,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    score = score_row(y_valid, probability, label, len(features))
    score["elapsed_seconds"] = elapsed
    return score, probability, encoded.encoder_state, model


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
    baseline104 = [f for f in engineered.features.columns if f not in recent]
    if len(baseline104) != args.expected_baseline_count:
        raise ValueError(
            f"Expected {args.expected_baseline_count} no-recent-form features, got {len(baseline104)}. "
            "Feature schema changed; inspect before comparing compact ladders."
        )

    feature_block = _feature_to_block(engineered, baseline104)
    current = list(baseline104)
    ladder: dict[int, list[str]] = {len(current): list(current)}
    stage_summaries: list[dict] = []

    for target_size in TARGET_SIZES:
        LOGGER.info(
            "Selection stage %d -> %d using seasons < %d / validate %d",
            len(current), target_size, args.selection_season, args.selection_season,
        )
        importance, stage_brier = _fit_selection_stage(
            engineered,
            current,
            selection_season=args.selection_season,
            max_iter=350,
            sample_size=args.selection_permutation_sample,
            repeats=args.selection_permutation_repeats,
            seed=args.seed,
        )
        importance["block"] = importance["feature"].map(feature_block)
        importance.to_csv(
            output_dir / f"selection_importance_{len(current)}_to_{target_size}.csv",
            index=False,
        )
        next_features, removed = _prune_to_target(
            current,
            importance,
            feature_block,
            target_size,
        )
        removed.to_csv(
            output_dir / f"removed_{len(current)}_to_{target_size}.csv",
            index=False,
        )
        stage_summaries.append(
            {
                "from_features": len(current),
                "to_features": target_size,
                "selection_2023_brier": stage_brier,
                "removed": len(current) - target_size,
            }
        )
        current = next_features
        ladder[target_size] = list(current)

    scores: list[dict] = []
    validation_mask = engineered.seasons.astype(int) == args.validation_season
    prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: engineered.row_ids.loc[validation_mask].to_numpy(),
            "season": args.validation_season,
            TARGET: engineered.target.loc[validation_mask].to_numpy("int8"),
        }
    )

    for size in (args.expected_baseline_count, 80, 60, 40):
        features = ladder[size]
        LOGGER.info("Final evaluation: %d features on %d holdout", size, args.validation_season)
        score, probability, encoder_state, model = _evaluate_on_2024(
            engineered,
            features,
            validation_season=args.validation_season,
            max_iter=350,
            seed=args.seed,
            label=f"compact_{size}",
        )
        scores.append(score)
        prediction_frame[f"prediction_{size}"] = probability
        with (output_dir / f"features_{size}.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(features) + "\n")
        joblib.dump(
            {
                "model": model,
                "features": features,
                "encoder_state": encoder_state,
                "fixed_iteration": 350,
                "selection_season": args.selection_season,
                "validation_season": args.validation_season,
            },
            output_dir / f"hgb_compact_{size}.joblib",
        )
        del model, encoder_state, probability
        gc.collect()

    scores_df = pd.DataFrame(scores).sort_values("brier", ignore_index=True)
    scores_df.to_csv(output_dir / "compact_model_scores.csv", index=False)
    prediction_frame.to_csv(
        output_dir / "compact_validation_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(stage_summaries).to_csv(output_dir / "selection_stage_summary.csv", index=False)

    summary = {
        "selection_design": {
            "candidate_pool": "118 full features minus 14-feature recent_form block = 104",
            "selection_train_seasons": f"< {args.selection_season}",
            "selection_validation_season": args.selection_season,
            "final_train_seasons": f"< {args.validation_season}",
            "final_validation_season": args.validation_season,
            "fixed_hgb_iterations": 350,
            "ranking_metric": "permutation Brier delta; robust_importance = mean - std",
            "recursive_targets": list(TARGET_SIZES),
            "block_floor": "at least one feature from every non-recent semantic block",
            "protected_features": sorted(PROTECTED_FEATURES),
        },
        "scores": scores_df.to_dict(orient="records"),
        "stages": stage_summaries,
    }
    with (output_dir / "compact_ladder_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 104/80/60/40 fixed-350 HGB models using 2023-only recursive feature selection."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--trackman", required=True)
    parser.add_argument("--mapping", default="resources/pitcher_trackman_mapping.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trackman-cache-dir")
    parser.add_argument("--selection-season", type=int, default=2023)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--selection-permutation-sample", type=int, default=50000)
    parser.add_argument("--selection-permutation-repeats", type=int, default=3)
    parser.add_argument("--expected-baseline-count", type=int, default=104)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
