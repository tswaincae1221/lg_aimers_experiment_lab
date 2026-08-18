from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import (
    ID_COLUMN, TARGET, _atomic_to_csv, _file_signature, _read_csv, _stable_hash,
    build_main_features, resolve_input_path, sample_by_season,
)
from .trackman import build_trackman_features, load_mapping, load_or_aggregate_trackman, merge_trackman
from .selection import (
    _feature_catalog, brier, choose_selected_features, fit_subset, frequency_encode,
    run_block_ablation, run_lofo, run_permutation_importance, score_row, tune_iterations,
)

LOGGER = logging.getLogger(__name__)

def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_input_path(args.train, "train.csv")
    trackman_path = resolve_input_path(args.trackman, "trackman_history.csv")
    mapping_path = Path(args.mapping)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}. Run from the repository root.")

    LOGGER.info("Reading train: %s", train_path)
    train = _read_csv(train_path)
    train = sample_by_season(train, args.max_rows_per_season, args.seed)
    main = build_main_features(train, shrinkage=args.shrinkage)

    LOGGER.info("Reading mapping: %s", mapping_path)
    mapping = load_mapping(mapping_path, tuple(args.accepted_mapping_grades))
    LOGGER.info("Accepted mappings: %d", len(mapping))

    LOGGER.info("Aggregating Trackman: %s", trackman_path)
    group_stats, context_stats, drift_stats = load_or_aggregate_trackman(
        trackman_path,
        output_dir,
        max_rows=args.max_trackman_rows,
        chunksize=args.trackman_chunksize,
        rerun=args.rerun,
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

    # Remove duplicate perfect-information column if both official counts are identical in this data.
    if "asof_pitcher_pitchmix_n" in engineered.features.columns:
        engineered.features.drop(columns=["asof_pitcher_pitchmix_n"], inplace=True)

    validation_mask = engineered.seasons.astype(int) == args.validation_season
    training_mask = engineered.seasons.astype(int) < args.validation_season
    if training_mask.sum() == 0 or validation_mask.sum() == 0:
        raise ValueError(
            f"Invalid split: train={int(training_mask.sum())}, valid={int(validation_mask.sum())}, "
            f"validation_season={args.validation_season}"
        )

    best_iter = tune_iterations(
        engineered,
        tuning_season=args.tuning_season,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    encoded = frequency_encode(
        engineered.features.loc[training_mask],
        engineered.features.loc[validation_mask],
        engineered.categorical_features,
    )
    y_train = engineered.target.loc[training_mask].to_numpy("int8")
    y_valid = engineered.target.loc[validation_mask].to_numpy("int8")
    all_idx = np.arange(len(encoded.feature_names), dtype="int32")
    run_signature = _stable_hash(
        {
            "pipeline_version": 2,
            "train": _file_signature(train_path),
            "trackman": _file_signature(trackman_path),
            "mapping": _file_signature(mapping_path),
            "validation_season": args.validation_season,
            "tuning_season": args.tuning_season,
            "max_rows_per_season": args.max_rows_per_season,
            "max_trackman_rows": args.max_trackman_rows,
            "best_iter": best_iter,
            "features": encoded.feature_names,
            "permutation_repeats": args.permutation_repeats,
            "permutation_sample": args.permutation_sample,
            "lofo_candidates": args.lofo_candidates,
        }
    )

    LOGGER.info("Training full HGB: rows=%d features=%d iter=%d", len(y_train), len(all_idx), best_iter)
    start = time.perf_counter()
    full_model, full_probability, full_brier = fit_subset(
        encoded.x_train,
        y_train,
        encoded.x_valid,
        y_valid,
        all_idx,
        max_iter=best_iter,
        seed=args.seed,
    )
    full_elapsed = time.perf_counter() - start
    full_score = score_row(y_valid, full_probability, "full", len(all_idx))
    full_score["elapsed_seconds"] = full_elapsed

    block_ablation = run_block_ablation(
        engineered,
        encoded,
        y_train,
        y_valid,
        max_iter=best_iter,
        seed=args.seed,
        full_brier=full_brier,
        checkpoint_path=output_dir / "checkpoints" / "block_ablation.csv",
        run_signature=run_signature,
        rerun=args.rerun,
    )
    _atomic_to_csv(block_ablation, output_dir / "block_ablation.csv")

    permutation = run_permutation_importance(
        full_model,
        encoded.x_valid,
        y_valid,
        encoded.feature_names,
        repeats=args.permutation_repeats,
        sample_size=args.permutation_sample,
        seed=args.seed,
        checkpoint_path=output_dir / "checkpoints" / "permutation_importance.csv",
        run_signature=run_signature,
        rerun=args.rerun,
    )
    _atomic_to_csv(permutation, output_dir / "permutation_importance.csv")

    lofo = run_lofo(
        encoded,
        y_train,
        y_valid,
        permutation,
        max_candidates=args.lofo_candidates,
        max_iter=best_iter,
        seed=args.seed,
        full_brier=full_brier,
        checkpoint_path=output_dir / "checkpoints" / "lofo_results.csv",
        run_signature=run_signature,
        rerun=args.rerun,
    )
    _atomic_to_csv(lofo, output_dir / "lofo_results.csv")

    selected_features, dropped_features = choose_selected_features(
        encoded.feature_names,
        lofo,
        drop_tolerance=args.lofo_drop_tolerance,
    )
    selected_idx = np.array([encoded.feature_names.index(name) for name in selected_features], dtype="int32")
    selected_model, selected_probability, selected_brier = fit_subset(
        encoded.x_train,
        y_train,
        encoded.x_valid,
        y_valid,
        selected_idx,
        max_iter=best_iter,
        seed=args.seed,
    )

    # Conservative safeguard: feature selection must not materially hurt the 2024 holdout.
    selection_accepted = selected_brier <= full_brier + args.max_selected_regret
    if not selection_accepted:
        LOGGER.warning(
            "Combined selection worsened Brier by %.8f (> %.8f); falling back to full feature set.",
            selected_brier - full_brier,
            args.max_selected_regret,
        )
        selected_features = list(encoded.feature_names)
        dropped_features = []
        selected_model = full_model
        selected_probability = full_probability
        selected_brier = full_brier
        selected_idx = all_idx

    selected_score = score_row(y_valid, selected_probability, "selected", len(selected_features))
    selected_score["elapsed_seconds"] = np.nan
    pd.DataFrame([full_score, selected_score]).to_csv(output_dir / "model_scores.csv", index=False)

    permutation_lofo = permutation.merge(lofo, on="feature", how="left")
    block_map = _feature_catalog(engineered)[["feature", "block"]]
    final_selection = permutation_lofo.merge(block_map, on="feature", how="left")
    final_selection["selected"] = final_selection["feature"].isin(selected_features)
    final_selection.to_csv(output_dir / "feature_selection.csv", index=False)
    _feature_catalog(engineered).to_csv(output_dir / "feature_catalog.csv", index=False)

    with (output_dir / "selected_features.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(selected_features) + "\n")
    with (output_dir / "dropped_features.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(dropped_features) + ("\n" if dropped_features else ""))

    validation_predictions = pd.DataFrame(
        {
            ID_COLUMN: engineered.row_ids.loc[validation_mask].to_numpy(),
            "season": args.validation_season,
            TARGET: y_valid,
            "prediction_full": full_probability,
            "prediction_selected": selected_probability,
        }
    )
    validation_predictions.to_csv(
        output_dir / "validation_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    joblib.dump(
        {
            "model": selected_model,
            "selected_features": selected_features,
            "encoder_state": encoded.encoder_state,
            "best_iteration": best_iter,
            "validation_season": args.validation_season,
        },
        output_dir / "hgb_selected_model.joblib",
    )

    summary = {
        "run_signature": run_signature,
        "train_path": str(train_path),
        "trackman_path": str(trackman_path),
        "mapping_path": str(mapping_path),
        "train_rows": int(training_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_season": args.validation_season,
        "tuning_season": args.tuning_season,
        "best_iteration": best_iter,
        "full_feature_count": len(encoded.feature_names),
        "selected_feature_count": len(selected_features),
        "dropped_feature_count": len(dropped_features),
        "selection_accepted": bool(selection_accepted),
        "full_brier": full_brier,
        "selected_brier": selected_brier,
        "delta_selected_minus_full": selected_brier - full_brier,
        "rules": [
            "Train seasons < 2024, validate 2024.",
            "HGB iterations tuned on 2023 using seasons < 2023, then held fixed for selection.",
            "Trackman for a row uses only Trackman seasons < that row season.",
            "No current-pitch actual location, result, or actual pitch type is used.",
            "No test-row rolling/expanding statistics are created.",
        ],
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Done. full_brier=%.8f selected_brier=%.8f selected_features=%d/%d",
        full_brier,
        selected_brier,
        len(selected_features),
        len(encoded.feature_names),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab-ready literature-grounded HGB feature-selection pipeline (V2)."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--trackman", required=True)
    parser.add_argument("--mapping", default="resources/pitcher_trackman_mapping.csv")
    parser.add_argument("--output-dir", default="results/hgb_feature_selection_v2")
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--tuning-season", type=int, default=2023)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--shrinkage", type=float, default=50.0)
    parser.add_argument("--trackman-shrinkage", type=float, default=100.0)
    parser.add_argument("--context-smoothing", type=float, default=50.0)
    parser.add_argument("--trackman-chunksize", type=int, default=250000)
    parser.add_argument("--accepted-mapping-grades", nargs="+", default=["확정", "높음"])
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--permutation-sample", type=int, default=75000)
    parser.add_argument("--lofo-candidates", type=int, default=20)
    parser.add_argument(
        "--lofo-drop-tolerance",
        type=float,
        default=1e-5,
        help="Drop candidate if removing it worsens Brier by no more than this amount.",
    )
    parser.add_argument("--max-selected-regret", type=float, default=5e-5)
    parser.add_argument("--max-rows-per-season", type=int)
    parser.add_argument("--max-trackman-rows", type=int)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore Trackman/selection checkpoints and recompute from scratch.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="quick sets safe development caps unless explicit caps were supplied.",
    )
    args = parser.parse_args()
    if args.mode == "quick":
        if args.max_rows_per_season is None:
            args.max_rows_per_season = 5000
        if args.max_trackman_rows is None:
            args.max_trackman_rows = 300000
        args.permutation_sample = min(args.permutation_sample, 15000)
        args.permutation_repeats = min(args.permutation_repeats, 2)
        args.lofo_candidates = min(args.lofo_candidates, 8)
        args.max_iter = min(args.max_iter, 120)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
