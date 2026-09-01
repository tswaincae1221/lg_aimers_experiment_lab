from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_features import apply_feature_set, feature_catalog
from src.experiment_models import fit_validation_model, save_model_artifact
from src.experiment_reporting import (
    add_baseline_deltas,
    add_comparison_deltas,
    add_preprocessing_deltas,
    compute_metrics,
    write_global_artifacts,
    write_json,
    write_markdown_report,
    write_run_artifacts,
)
from src.first_model_features import (
    ID_COLUMN,
    TARGET,
    aggregate_trackman,
    assemble_features,
    assemble_raw_features,
    load_mapping,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configuration-driven LG Aimers model and feature experiment lab."
    )
    parser.add_argument("--config", default="config/experiments.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test")
    parser.add_argument("--trackman")
    parser.add_argument("--mapping")
    parser.add_argument("--output-dir", default="results/experiment_lab")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--preset",
        choices=["starter", "hist_compare", "four_models", "extended", "all"],
        default="starter",
        help=(
            "Experiment group. 'hist_compare' runs only HistGradientBoosting; "
            "'four_models' compares HistGB, XGBoost, CatBoost, and LightGBM."
        ),
    )
    parser.add_argument("--only", nargs="+", help="Run only these experiment names")
    parser.add_argument("--validation-season", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = ["feature_sets", "models", "experiments", "baseline_experiment"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Experiment config is missing keys: {missing}")
    return config


def sample_by_season(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame
    parts = [
        group.sample(min(len(group), limit), random_state=seed)
        for _, group in frame.groupby("season", observed=True, sort=True)
    ]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _file_signature(path: str | None) -> dict | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return {"path": str(target.resolve()), "missing": True}
    stat = target.stat()
    return {
        "path": str(target.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def stable_hash(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def select_experiments(config: dict, preset: str, only: list[str] | None) -> list[dict]:
    experiments = config["experiments"]
    if only:
        lookup = {experiment["name"]: experiment for experiment in experiments}
        missing = [name for name in only if name not in lookup]
        if missing:
            raise ValueError(f"Unknown --only experiments: {missing}")
        return [lookup[name] for name in only]

    selected = []
    for experiment in experiments:
        if not experiment.get("enabled", True):
            continue
        if preset == "all" or preset in experiment.get("presets", ["starter"]):
            selected.append(experiment)
    if not selected:
        raise ValueError(f"No experiments selected for preset={preset}")
    return selected


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-")


def _read_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _append_history(path: Path, row: dict) -> pd.DataFrame:
    history = _read_history(path)
    updated = pd.concat([history, pd.DataFrame([row])], ignore_index=True, sort=False)
    updated.to_csv(path, index=False)
    return updated


def _latest_leaderboard(
    history: pd.DataFrame,
    data_signature: str,
    mode: str,
    validation_season: int,
    baseline_experiment: str,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    comparable = history.loc[
        (history["status"] == "success")
        & (history["data_signature"] == data_signature)
        & (history["mode"] == mode)
        & (pd.to_numeric(history["validation_season"], errors="coerce") == validation_season)
    ].copy()
    if comparable.empty:
        return comparable
    comparable = comparable.sort_values("finished_at").groupby(
        "experiment", observed=True, as_index=False
    ).tail(1)
    comparable = add_baseline_deltas(comparable, baseline_experiment)
    comparable = add_comparison_deltas(comparable)
    comparable = add_preprocessing_deltas(comparable)
    return comparable.sort_values("brier", ascending=True).reset_index(drop=True)


def _already_completed(
    history: pd.DataFrame,
    data_signature: str,
    config_hash: str,
    mode: str,
) -> bool:
    if history.empty:
        return False
    return bool(
        (
            (history["status"] == "success")
            & (history["data_signature"] == data_signature)
            & (history["config_hash"] == config_hash)
            & (history["mode"] == mode)
        ).any()
    )


def _load_trackman_summary(
    trackman_path: str,
    output_dir: Path,
    max_rows: int | None,
) -> pd.DataFrame:
    signature_payload = {
        "file": _file_signature(trackman_path),
        "max_rows": max_rows,
    }
    signature = stable_hash(signature_payload)[:12]
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"trackman_summary_{signature}.pkl"
    metadata_path = cache_dir / f"trackman_summary_{signature}.json"
    if cache_path.is_file():
        LOGGER.info("Loading cached Trackman summary: %s", cache_path)
        return pd.read_pickle(cache_path)

    LOGGER.info("Aggregating Trackman rows (first run only)")
    summary = aggregate_trackman(trackman_path, nrows=max_rows)
    summary.to_pickle(cache_path)
    write_json(
        metadata_path,
        {**signature_payload, "summary_rows": len(summary), "cache_path": str(cache_path)},
    )
    return summary


def _runtime_info() -> dict:
    packages = {}
    for package in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost"]:
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _refresh_outputs(
    output_dir: Path,
    history: pd.DataFrame,
    data_signature: str,
    mode: str,
    validation_season: int,
    baseline_experiment: str,
) -> pd.DataFrame:
    leaderboard = _latest_leaderboard(
        history,
        data_signature,
        mode,
        validation_season,
        baseline_experiment,
    )
    if leaderboard.empty:
        return leaderboard
    write_global_artifacts(output_dir, leaderboard, baseline_experiment)
    write_markdown_report(
        output_dir,
        leaderboard,
        baseline_experiment,
        validation_season,
    )
    return leaderboard


def run(args: argparse.Namespace) -> pd.DataFrame:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    history_path = output_dir / "experiment_history.csv"
    seed = int(config.get("seed", 42))
    n_jobs = int(args.n_jobs or config.get("n_jobs", 4))
    validation_season = int(
        args.validation_season or config.get("validation_season", 2024)
    )
    mode_config = config.get("modes", {}).get(args.mode, {})
    max_rows = mode_config.get("max_rows_per_season")
    max_trackman_rows = mode_config.get("max_trackman_rows")

    data_payload = {
        "train": _file_signature(args.train),
        "test": _file_signature(args.test),
        "trackman": _file_signature(args.trackman),
        "mapping": _file_signature(args.mapping),
        "validation_season": validation_season,
        "mode": args.mode,
        "max_rows_per_season": max_rows,
        "max_trackman_rows": max_trackman_rows,
    }
    data_signature = stable_hash(data_payload)
    write_json(output_dir / "data_signature.json", {**data_payload, "hash": data_signature})
    write_json(output_dir / "runtime.json", _runtime_info())
    feature_catalog().to_csv(output_dir / "feature_catalog.csv", index=False)

    selected = select_experiments(config, args.preset, args.only)
    pd.DataFrame(selected).to_json(
        output_dir / "selected_experiments.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    LOGGER.info("Selected %d experiments: %s", len(selected), [x["name"] for x in selected])
    if args.mode == "quick":
        LOGGER.warning(
            "QUICK mode is active (%s rows/season). Scores are not for model selection.",
            max_rows,
        )

    train = pd.read_csv(args.train, encoding="utf-8-sig", low_memory=False)
    train = sample_by_season(train, max_rows, seed)
    test = (
        pd.read_csv(args.test, encoding="utf-8-sig", low_memory=False)
        if args.test and Path(args.test).is_file()
        else None
    )

    if bool(args.trackman) != bool(args.mapping):
        raise ValueError("--trackman and --mapping must be supplied together")
    trackman_summary = None
    mapping = None
    if args.trackman:
        trackman_summary = _load_trackman_summary(
            args.trackman,
            output_dir,
            max_trackman_rows,
        )
        mapping = load_mapping(
            args.mapping,
            config.get("accepted_mapping_grades", ["확정", "높음"]),
        )

    requested_views = {
        config["feature_sets"][item["feature_set"]].get("input_view", "v1")
        for item in selected
    }
    bundles = {}
    if "v1" in requested_views:
        LOGGER.info("Building leakage-safe V1 features once for engineered experiments")
        bundles["v1"] = assemble_features(
            train,
            test,
            trackman_summary=trackman_summary,
            mapping=mapping,
            smoothing=float(config.get("smoothing", 50.0)),
            include_asof_trends=False,
        )
    if "raw_minimal" in requested_views:
        LOGGER.info("Building official-column raw-minimal feature view")
        bundles["raw_minimal"] = assemble_raw_features(train, test)
    unknown_views = requested_views.difference(bundles)
    if unknown_views:
        raise ValueError(f"Unknown feature input views: {sorted(unknown_views)}")

    split_rows = {}
    expected_row_ids = None
    for input_view, view_bundle in bundles.items():
        labeled = np.zeros(len(view_bundle.features), dtype=bool)
        labeled[view_bundle.train_rows] = True
        seasons = view_bundle.seasons.to_numpy()
        train_rows = np.flatnonzero(labeled & (seasons < validation_season))
        valid_rows = np.flatnonzero(labeled & (seasons == validation_season))
        if not len(train_rows) or not len(valid_rows):
            raise ValueError(
                f"Invalid split for {input_view}: train season < {validation_season} "
                f"has {len(train_rows)} rows, validation season == {validation_season} "
                f"has {len(valid_rows)} rows"
            )
        validation_ids = view_bundle.row_ids.iloc[valid_rows].reset_index(drop=True)
        if expected_row_ids is None:
            expected_row_ids = validation_ids
        elif not validation_ids.equals(expected_row_ids):
            raise ValueError("Raw and engineered feature views do not align on validation rows")
        split_rows[input_view] = (train_rows, valid_rows)
        LOGGER.info(
            "Team split fixed [%s]: season < %d train=%d, season == %d valid=%d",
            input_view,
            validation_season,
            len(train_rows),
            validation_season,
            len(valid_rows),
        )

    history = _read_history(history_path)
    for experiment in selected:
        name = experiment["name"]
        feature_set_name = experiment["feature_set"]
        model_name = experiment["model"]
        if feature_set_name not in config["feature_sets"]:
            raise ValueError(f"Unknown feature set in {name}: {feature_set_name}")
        if model_name not in config["models"]:
            raise ValueError(f"Unknown model in {name}: {model_name}")
        feature_set_config = config["feature_sets"][feature_set_name]
        model_config = config["models"][model_name]
        input_view = feature_set_config.get("input_view", "v1")
        bundle = bundles[input_view]
        train_rows, valid_rows = split_rows[input_view]
        tuning_data = None
        tuning_train_rows = np.array([], dtype="int64")
        tuning_valid_rows = np.array([], dtype="int64")
        tuning_season = validation_season - 1
        if model_config.get("refit_after_tuning", False):
            view_seasons = bundle.seasons.to_numpy()
            labeled = np.zeros(len(bundle.features), dtype=bool)
            labeled[bundle.train_rows] = True
            tuning_train_rows = np.flatnonzero(labeled & (view_seasons < tuning_season))
            tuning_valid_rows = np.flatnonzero(labeled & (view_seasons == tuning_season))
            if not len(tuning_train_rows) or not len(tuning_valid_rows):
                raise ValueError(
                    f"Cannot build tuning split for {name}: season < {tuning_season} "
                    f"train={len(tuning_train_rows)}, season == {tuning_season} "
                    f"valid={len(tuning_valid_rows)}"
                )
            tuning_data = (tuning_train_rows, tuning_valid_rows)
        resolved = {
            "experiment": experiment,
            "feature_set": feature_set_config,
            "model": model_config,
            "seed": seed,
            "n_jobs": n_jobs,
            "validation_season": validation_season,
        }
        config_hash = stable_hash(resolved)
        if not args.rerun and _already_completed(
            history, data_signature, config_hash, args.mode
        ):
            LOGGER.info("Skipping completed experiment: %s", name)
            continue

        started_at = datetime.now(timezone.utc)
        timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = output_dir / "runs" / f"{timestamp}__{_slug(name)}__{config_hash[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "resolved_config.json", resolved)
        LOGGER.info("Starting experiment: %s", name)
        start_clock = time.perf_counter()

        base_row = {
            "run_id": run_dir.name,
            "experiment": name,
            "feature_set": feature_set_name,
            "model": model_name,
            "model_type": model_config["type"],
            "input_view": input_view,
            "preprocessing_stage": feature_set_config.get(
                "preprocessing_stage", input_view
            ),
            "comparison_experiment": experiment.get("comparison_experiment"),
            "preprocessing_comparison_experiment": experiment.get(
                "preprocessing_comparison_experiment"
            ),
            "model_family": model_config.get("family", model_config["type"]),
            "tuning_season": tuning_season
            if model_config.get("refit_after_tuning", False)
            else None,
            "tuning_train_n": int(len(tuning_train_rows)),
            "tuning_valid_n": int(len(tuning_valid_rows)),
            "mode": args.mode,
            "validation_season": validation_season,
            "data_signature": data_signature,
            "config_hash": config_hash,
            "started_at": started_at.isoformat(),
            "run_dir": str(run_dir),
            "train_n": int(len(train_rows)),
            "valid_n": int(len(valid_rows)),
        }
        try:
            features, categorical, feature_report = apply_feature_set(
                bundle.features,
                bundle.categorical_features,
                feature_set_config,
            )
            write_json(run_dir / "feature_report.json", feature_report)
            resolved_tuning_data = None
            if tuning_data is not None:
                tuning_train_rows, tuning_valid_rows = tuning_data
                resolved_tuning_data = (
                    features.iloc[tuning_train_rows],
                    bundle.target.iloc[tuning_train_rows],
                    features.iloc[tuning_valid_rows],
                    bundle.target.iloc[tuning_valid_rows],
                )
            model_result = fit_validation_model(
                model_config,
                features.iloc[train_rows],
                bundle.target.iloc[train_rows],
                features.iloc[valid_rows],
                bundle.target.iloc[valid_rows],
                categorical,
                seed=seed,
                n_jobs=n_jobs,
                tuning_data=resolved_tuning_data,
            )
            elapsed = time.perf_counter() - start_clock
            metric_values = compute_metrics(
                bundle.target.iloc[valid_rows], model_result.probability
            )
            predictions = pd.DataFrame(
                {
                    ID_COLUMN: bundle.row_ids.iloc[valid_rows].to_numpy(),
                    "season": validation_season,
                    TARGET: bundle.target.iloc[valid_rows].astype("int8").to_numpy(),
                    "prediction": model_result.probability,
                }
            )
            predictions.to_csv(
                run_dir / "validation_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
            write_run_artifacts(
                run_dir,
                bundle.target.iloc[valid_rows],
                model_result.probability,
                model_result.importance,
            )
            if config.get("save_validation_models", False):
                save_model_artifact(model_result, model_config["type"], run_dir)

            finished_at = datetime.now(timezone.utc).isoformat()
            record = {
                **base_row,
                "status": "success",
                "finished_at": finished_at,
                "elapsed_seconds": round(elapsed, 3),
                "feature_count": int(features.shape[1]),
                "categorical_count": len(categorical),
                "best_iteration": model_result.best_iteration,
                **metric_values,
            }
            write_json(run_dir / "metrics.json", record)
            history = _append_history(history_path, record)
            LOGGER.info(
                "Finished %s | brier=%.8f auc=%.6f time=%.1fs",
                name,
                metric_values["brier"],
                metric_values["auc"],
                elapsed,
            )
        except Exception as exc:  # continue the remaining experiment matrix
            elapsed = time.perf_counter() - start_clock
            error_text = traceback.format_exc()
            (run_dir / "error.txt").write_text(error_text, encoding="utf-8")
            record = {
                **base_row,
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_json(run_dir / "metrics.json", record)
            history = _append_history(history_path, record)
            LOGGER.error("Experiment failed: %s | %s: %s", name, type(exc).__name__, exc)

        _refresh_outputs(
            output_dir,
            history,
            data_signature,
            args.mode,
            validation_season,
            config["baseline_experiment"],
        )

    leaderboard = _refresh_outputs(
        output_dir,
        history,
        data_signature,
        args.mode,
        validation_season,
        config["baseline_experiment"],
    )
    if leaderboard.empty:
        raise RuntimeError("All selected experiments failed. Check runs/*/error.txt")
    LOGGER.info("Best experiment: %s", leaderboard.iloc[0]["experiment"])
    LOGGER.info("Best Brier Score: %.8f", leaderboard.iloc[0]["brier"])
    return leaderboard


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
