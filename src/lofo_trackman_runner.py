from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_features import (
    BATTER_THREAT_INTERACTION_FEATURES,
    apply_feature_set,
)
from src.experiment_models import fit_validation_model
from src.experiment_reporting import compute_metrics
from src.first_model_features import (
    ASOF_TREND_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET,
    aggregate_trackman,
    assemble_raw_features,
    build_trackman_features,
    load_mapping,
)
from src.trackman_sabermetrics import (
    GROUP_METADATA,
    build_trackman_catalog,
    group_trackman_features,
)


LOGGER = logging.getLogger(__name__)
BASE_BLOCKS = ["batter_threat_interactions", "asof_trend"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe HGB/CatBoost LOFO on 47 basic + 6 teammate + "
            "13 trend features, then Trackman add-back experiments."
        )
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--trackman")
    parser.add_argument("--mapping", default="resources/pitcher_trackman_mapping.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--phase", choices=["all", "lofo", "trackman"], default="all"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["hist_gbdt", "catboost"],
        default=["hist_gbdt", "catboost"],
    )
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-rows-per-season", type=int)
    parser.add_argument("--max-trackman-rows", type=int)
    parser.add_argument(
        "--trackman-scope",
        choices=["none", "groups", "all"],
        default="all",
        help="'all' runs both six group add-backs and every individual Trackman add-back.",
    )
    parser.add_argument(
        "--catboost-task-type", choices=["CPU", "GPU"], default="CPU"
    )
    parser.add_argument(
        "--selected-trackman-top-k",
        type=int,
        default=8,
        help="Number of consensus-positive individual Trackman features in the bundle check.",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _stable_hash(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_signature(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return {"path": str(target), "missing": True}
    stat = target.stat()
    return {
        "path": str(target.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sample_by_season(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame.reset_index(drop=True)
    parts = [
        group.sample(min(len(group), limit), random_state=seed)
        for _, group in frame.groupby("season", observed=True, sort=True)
    ]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _feature_catalog(feature_names: list[str]) -> pd.DataFrame:
    basic = set(RAW_NUMERIC_FEATURES) | set(RAW_CATEGORICAL_FEATURES)
    teammate = set(BATTER_THREAT_INTERACTION_FEATURES)
    trend = set(ASOF_TREND_FEATURES)
    rows = []
    for feature in feature_names:
        if feature in basic:
            group = "basic"
            label = "공식 기본 피처"
        elif feature in teammate:
            group = "teammate_important"
            label = "팀원 중요도 피처"
        elif feature in trend:
            group = "trend"
            label = "과거 추세 피처"
        else:
            raise ValueError(f"Unexpected base feature: {feature}")
        rows.append({"feature": feature, "feature_group": group, "group_label_ko": label})
    return pd.DataFrame(rows)


def _model_configs(args: argparse.Namespace) -> dict[str, dict]:
    quick = args.mode == "quick"
    return {
        "hist_gbdt": {
            "type": "hist_gbdt",
            "params": {
                "learning_rate": 0.06,
                "max_iter": 90 if quick else 350,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 80 if quick else 200,
                "l2_regularization": 2.0,
                "early_stopping": False,
            },
        },
        "catboost": {
            "type": "catboost",
            "params": {
                "iterations": 120 if quick else 400,
                "learning_rate": 0.035,
                "depth": 8,
                "l2_leaf_reg": 5.0,
                "task_type": args.catboost_task_type,
                "verbose": False,
            },
            # A fixed count prevents the 2024 holdout from selecting iterations.
            "early_stopping_rounds": 0,
        },
    }


def _load_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _append_history(path: Path, history: pd.DataFrame, row: dict) -> pd.DataFrame:
    updated = pd.concat([history, pd.DataFrame([row])], ignore_index=True, sort=False)
    _atomic_to_csv(updated, path)
    return updated


def _completed(
    history: pd.DataFrame,
    run_signature: str,
    model: str,
    experiment_type: str,
    item: str,
) -> bool:
    if history.empty:
        return False
    return bool(
        (
            (history["run_signature"] == run_signature)
            & (history["model"] == model)
            & (history["experiment_type"] == experiment_type)
            & (history["item"] == item)
            & (history["status"] == "success")
        ).any()
    )


def _latest_success(history: pd.DataFrame, run_signature: str) -> pd.DataFrame:
    if history.empty:
        return history
    selected = history.loc[
        (history["run_signature"] == run_signature) & (history["status"] == "success")
    ].copy()
    if selected.empty:
        return selected
    return selected.groupby(
        ["model", "experiment_type", "item"], observed=True, as_index=False
    ).tail(1)


def _fit_experiment(
    *,
    model_name: str,
    model_config: dict,
    features: pd.DataFrame,
    categorical: list[str],
    target: pd.Series,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    experiment_type: str,
    item: str,
    run_signature: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    started = time.perf_counter()
    selected_categorical = [column for column in categorical if column in features.columns]
    result = fit_validation_model(
        model_config,
        features.loc[train_mask],
        target.loc[train_mask],
        features.loc[valid_mask],
        target.loc[valid_mask],
        selected_categorical,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    metrics = compute_metrics(target.loc[valid_mask], result.probability)
    if experiment_type == "baseline" and not result.importance.empty:
        importance_path = output_dir / f"baseline_intrinsic_importance_{model_name}.csv"
        _atomic_to_csv(result.importance, importance_path)
    row = {
        "run_signature": run_signature,
        "status": "success",
        "mode": args.mode,
        "validation_season": args.validation_season,
        "model": model_name,
        "experiment_type": experiment_type,
        "item": item,
        "feature_count": int(features.shape[1]),
        "train_n": int(train_mask.sum()),
        "valid_n": int(valid_mask.sum()),
        "best_iteration": result.best_iteration,
        "elapsed_seconds": float(time.perf_counter() - started),
        **metrics,
    }
    del result
    gc.collect()
    return row


def _load_trackman_summary(
    trackman_path: str | Path,
    output_dir: Path,
    max_rows: int | None,
) -> pd.DataFrame:
    signature = _stable_hash(
        {"file": _file_signature(trackman_path), "max_rows": max_rows}
    )[:12]
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"trackman_summary_{signature}.pkl"
    if cache_path.is_file():
        LOGGER.info("Trackman cache hit: %s", cache_path)
        return pd.read_pickle(cache_path)
    LOGGER.info("Aggregating Trackman history. This is done once per data signature.")
    summary = aggregate_trackman(trackman_path, nrows=max_rows)
    summary.to_pickle(cache_path)
    return summary


def _build_trackman_frame(
    train: pd.DataFrame,
    trackman_path: str | Path,
    mapping_path: str | Path,
    output_dir: Path,
    max_rows: int | None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    summary = _load_trackman_summary(trackman_path, output_dir, max_rows)
    mapping = load_mapping(mapping_path)
    current = train[["pitcher_id", "season"]].copy()
    current["pitcher_id"] = (
        current["pitcher_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    current["season"] = pd.to_numeric(current["season"], errors="raise").astype("int16")
    frame = build_trackman_features(current, summary, mapping)
    categorical = ["tm_mapping_grade"] if "tm_mapping_grade" in frame.columns else []
    catalog = build_trackman_catalog(frame.columns)
    return frame, categorical, catalog


def _write_rankings(
    history: pd.DataFrame,
    run_signature: str,
    base_catalog: pd.DataFrame,
    trackman_catalog: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    current = _latest_success(history, run_signature)
    if current.empty:
        return
    baselines = current.loc[current["experiment_type"] == "baseline", ["model", "brier"]]
    baseline_map = baselines.drop_duplicates("model", keep="last").set_index("model")["brier"]
    _atomic_to_csv(baselines.sort_values("brier"), output_dir / "baseline_scores.csv")

    lofo = current.loc[current["experiment_type"] == "lofo"].copy()
    if not lofo.empty:
        lofo["baseline_brier"] = lofo["model"].map(baseline_map)
        lofo["delta_brier"] = lofo["brier"] - lofo["baseline_brier"]
        lofo["importance_pct_of_baseline"] = (
            100.0 * lofo["delta_brier"] / lofo["baseline_brier"]
        )
        lofo = lofo.merge(base_catalog, left_on="item", right_on="feature", how="left")
        lofo["rank_within_model"] = lofo.groupby("model")["delta_brier"].rank(
            method="min", ascending=False
        )
        lofo = lofo.sort_values(["model", "delta_brier"], ascending=[True, False])
        _atomic_to_csv(lofo, output_dir / "lofo_importance_by_model.csv")

        pivot = lofo.pivot(index=["item", "feature_group"], columns="model", values="delta_brier")
        combined = pivot.reset_index().rename(columns={"item": "feature"})
        model_columns = [column for column in ["hist_gbdt", "catboost"] if column in combined]
        combined["mean_delta_brier"] = combined[model_columns].mean(axis=1)
        combined["positive_in_all_run_models"] = combined[model_columns].gt(0).all(axis=1)
        combined["combined_rank"] = combined["mean_delta_brier"].rank(
            method="min", ascending=False
        )
        combined = combined.sort_values("mean_delta_brier", ascending=False)
        _atomic_to_csv(combined, output_dir / "lofo_importance_combined.csv")

    trackman = current.loc[
        current["experiment_type"].isin(
            ["trackman_group", "trackman_individual", "trackman_selected_bundle"]
        )
    ].copy()
    if not trackman.empty:
        trackman["baseline_brier"] = trackman["model"].map(baseline_map)
        trackman["brier_improvement"] = trackman["baseline_brier"] - trackman["brier"]
        trackman["improvement_pct_of_baseline"] = (
            100.0 * trackman["brier_improvement"] / trackman["baseline_brier"]
        )
        trackman["rank_within_model"] = trackman.groupby(
            ["model", "experiment_type"]
        )["brier_improvement"].rank(method="min", ascending=False)
        if trackman_catalog is not None:
            lookup = trackman_catalog[["feature", "group", "group_label_ko"]]
            individual = trackman["experiment_type"] == "trackman_individual"
            matched = trackman.loc[individual, ["item"]].merge(
                lookup, left_on="item", right_on="feature", how="left"
            )
            trackman.loc[individual, "trackman_group"] = matched["group"].to_numpy()
            trackman.loc[individual, "trackman_group_label_ko"] = matched[
                "group_label_ko"
            ].to_numpy()
        trackman = trackman.sort_values(
            ["experiment_type", "model", "brier_improvement"],
            ascending=[True, True, False],
        )
        _atomic_to_csv(trackman, output_dir / "trackman_addback_by_model.csv")

        pivot = trackman.pivot(
            index=["experiment_type", "item"],
            columns="model",
            values="brier_improvement",
        ).reset_index()
        model_columns = [column for column in ["hist_gbdt", "catboost"] if column in pivot]
        pivot["mean_brier_improvement"] = pivot[model_columns].mean(axis=1)
        pivot["positive_in_all_run_models"] = pivot[model_columns].gt(0).all(axis=1)
        pivot["combined_rank"] = pivot.groupby("experiment_type")[
            "mean_brier_improvement"
        ].rank(method="min", ascending=False)
        pivot = pivot.sort_values(
            ["experiment_type", "mean_brier_improvement"], ascending=[True, False]
        )
        _atomic_to_csv(pivot, output_dir / "trackman_addback_combined.csv")


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"experiment_history_{args.mode}.csv"

    if args.max_rows_per_season is None and args.mode == "quick":
        args.max_rows_per_season = 12_000
    if args.max_trackman_rows is None and args.mode == "quick":
        args.max_trackman_rows = 400_000
    if args.mode == "full":
        args.max_rows_per_season = None
        args.max_trackman_rows = None

    LOGGER.info("Reading train.csv directly from: %s", args.train)
    train = pd.read_csv(args.train, low_memory=False)
    train = _sample_by_season(train, args.max_rows_per_season, args.seed)
    bundle = assemble_raw_features(train)
    base_features, base_categorical, feature_report = apply_feature_set(
        bundle.features,
        bundle.categorical_features,
        {"blocks": BASE_BLOCKS},
    )
    expected = len(RAW_NUMERIC_FEATURES) + len(RAW_CATEGORICAL_FEATURES) + 6 + 13
    if base_features.shape[1] != expected:
        raise AssertionError(
            f"Expected {expected} base/team/trend features, got {base_features.shape[1]}"
        )
    base_catalog = _feature_catalog(base_features.columns.tolist())
    _atomic_to_csv(base_catalog, output_dir / "base_feature_catalog.csv")

    seasons = bundle.seasons.to_numpy()
    target = bundle.target.astype("int8")
    train_mask = seasons < args.validation_season
    valid_mask = seasons == args.validation_season
    if not train_mask.any() or not valid_mask.any():
        raise ValueError(
            f"Invalid split: train={train_mask.sum()}, valid={valid_mask.sum()}, "
            f"validation_season={args.validation_season}"
        )

    configs = _model_configs(args)
    run_payload = {
        "train": _file_signature(args.train),
        "trackman": _file_signature(args.trackman),
        "mapping": _file_signature(args.mapping),
        "mode": args.mode,
        "validation_season": args.validation_season,
        "max_rows_per_season": args.max_rows_per_season,
        "max_trackman_rows": args.max_trackman_rows,
        "features": base_features.columns.tolist(),
        "models": {name: configs[name] for name in args.models},
    }
    run_signature = _stable_hash(run_payload)[:16]
    _write_json(
        {
            **run_payload,
            "run_signature": run_signature,
            "feature_report": feature_report,
            "train_n": int(train_mask.sum()),
            "valid_n": int(valid_mask.sum()),
        },
        output_dir / f"run_metadata_{args.mode}.json",
    )
    history = _load_history(history_path)

    for model_name in args.models:
        if args.rerun or not _completed(
            history, run_signature, model_name, "baseline", "base_66"
        ):
            LOGGER.info("[%s] baseline (66 features)", model_name)
            row = _fit_experiment(
                model_name=model_name,
                model_config=configs[model_name],
                features=base_features,
                categorical=base_categorical,
                target=target,
                train_mask=train_mask,
                valid_mask=valid_mask,
                experiment_type="baseline",
                item="base_66",
                run_signature=run_signature,
                args=args,
                output_dir=output_dir,
            )
            history = _append_history(history_path, history, row)

    if args.phase in {"all", "lofo"}:
        for model_name in args.models:
            for number, feature in enumerate(base_features.columns, start=1):
                if not args.rerun and _completed(
                    history, run_signature, model_name, "lofo", feature
                ):
                    continue
                LOGGER.info(
                    "[%s] LOFO %d/%d: remove %s",
                    model_name,
                    number,
                    base_features.shape[1],
                    feature,
                )
                subset = base_features.drop(columns=feature)
                row = _fit_experiment(
                    model_name=model_name,
                    model_config=configs[model_name],
                    features=subset,
                    categorical=base_categorical,
                    target=target,
                    train_mask=train_mask,
                    valid_mask=valid_mask,
                    experiment_type="lofo",
                    item=feature,
                    run_signature=run_signature,
                    args=args,
                    output_dir=output_dir,
                )
                history = _append_history(history_path, history, row)
                del subset
                gc.collect()
        _write_rankings(history, run_signature, base_catalog, None, output_dir)

    trackman_catalog = None
    if args.phase in {"all", "trackman"} and args.trackman_scope != "none":
        if not args.trackman:
            raise ValueError("Trackman phase requires --trackman")
        trackman, trackman_categorical, trackman_catalog = _build_trackman_frame(
            train,
            args.trackman,
            args.mapping,
            output_dir,
            args.max_trackman_rows,
        )
        _atomic_to_csv(trackman_catalog, output_dir / "trackman_feature_catalog.csv")
        trackman_groups = group_trackman_features(trackman_catalog)
        group_rows = []
        for group, columns in trackman_groups.items():
            metadata = GROUP_METADATA[group]
            group_rows.append(
                {
                    "group": group,
                    "group_label_ko": metadata["label_ko"],
                    "feature_count": len(columns),
                    "sabermetric_role": metadata["sabermetric_role"],
                    "rationale": metadata["rationale"],
                }
            )
        _atomic_to_csv(pd.DataFrame(group_rows), output_dir / "trackman_group_catalog.csv")

        for model_name in args.models:
            for group, columns in trackman_groups.items():
                if not args.rerun and _completed(
                    history, run_signature, model_name, "trackman_group", group
                ):
                    continue
                LOGGER.info("[%s] Trackman group add-back: %s", model_name, group)
                combined = pd.concat([base_features, trackman[columns]], axis=1)
                row = _fit_experiment(
                    model_name=model_name,
                    model_config=configs[model_name],
                    features=combined,
                    categorical=[*base_categorical, *trackman_categorical],
                    target=target,
                    train_mask=train_mask,
                    valid_mask=valid_mask,
                    experiment_type="trackman_group",
                    item=group,
                    run_signature=run_signature,
                    args=args,
                    output_dir=output_dir,
                )
                history = _append_history(history_path, history, row)
                del combined
                gc.collect()

            if args.trackman_scope == "all":
                for number, feature in enumerate(trackman.columns, start=1):
                    if not args.rerun and _completed(
                        history,
                        run_signature,
                        model_name,
                        "trackman_individual",
                        feature,
                    ):
                        continue
                    LOGGER.info(
                        "[%s] Trackman individual %d/%d: add %s",
                        model_name,
                        number,
                        trackman.shape[1],
                        feature,
                    )
                    combined = pd.concat([base_features, trackman[[feature]]], axis=1)
                    row = _fit_experiment(
                        model_name=model_name,
                        model_config=configs[model_name],
                        features=combined,
                        categorical=[*base_categorical, *trackman_categorical],
                        target=target,
                        train_mask=train_mask,
                        valid_mask=valid_mask,
                        experiment_type="trackman_individual",
                        item=feature,
                        run_signature=run_signature,
                        args=args,
                        output_dir=output_dir,
                    )
                    history = _append_history(history_path, history, row)
                    del combined
                    gc.collect()

        if args.trackman_scope == "all" and args.selected_trackman_top_k > 0:
            current = _latest_success(history, run_signature)
            individual = current.loc[
                current["experiment_type"] == "trackman_individual"
            ].copy()
            baseline_map = (
                current.loc[current["experiment_type"] == "baseline"]
                .drop_duplicates("model", keep="last")
                .set_index("model")["brier"]
            )
            individual["brier_improvement"] = (
                individual["model"].map(baseline_map) - individual["brier"]
            )
            selection = individual.pivot(
                index="item", columns="model", values="brier_improvement"
            )
            model_columns = [
                column for column in args.models if column in selection.columns
            ]
            selection["mean_brier_improvement"] = selection[model_columns].mean(axis=1)
            selection["positive_in_all_run_models"] = selection[model_columns].gt(0).all(
                axis=1
            )
            robust = selection.loc[selection["positive_in_all_run_models"]].copy()
            if robust.empty:
                robust = selection.loc[selection["mean_brier_improvement"] > 0].copy()
            robust = robust.sort_values("mean_brier_improvement", ascending=False).head(
                args.selected_trackman_top_k
            )
            selected_features = robust.index.tolist()
            selected_catalog = trackman_catalog.loc[
                trackman_catalog["feature"].isin(selected_features)
            ].merge(robust.reset_index(), left_on="feature", right_on="item", how="left")
            selected_catalog = selected_catalog.sort_values(
                "mean_brier_improvement", ascending=False
            )
            _atomic_to_csv(
                selected_catalog,
                output_dir / "selected_trackman_features.csv",
            )
            if selected_features:
                item = f"consensus_top_{len(selected_features)}"
                for model_name in args.models:
                    if not args.rerun and _completed(
                        history,
                        run_signature,
                        model_name,
                        "trackman_selected_bundle",
                        item,
                    ):
                        continue
                    LOGGER.info(
                        "[%s] exploratory selected Trackman bundle: %s",
                        model_name,
                        ", ".join(selected_features),
                    )
                    combined = pd.concat(
                        [base_features, trackman[selected_features]], axis=1
                    )
                    row = _fit_experiment(
                        model_name=model_name,
                        model_config=configs[model_name],
                        features=combined,
                        categorical=[*base_categorical, *trackman_categorical],
                        target=target,
                        train_mask=train_mask,
                        valid_mask=valid_mask,
                        experiment_type="trackman_selected_bundle",
                        item=item,
                        run_signature=run_signature,
                        args=args,
                        output_dir=output_dir,
                    )
                    history = _append_history(history_path, history, row)
                    del combined
                    gc.collect()

    _write_rankings(history, run_signature, base_catalog, trackman_catalog, output_dir)
    LOGGER.info("Done. Results and checkpoints: %s", output_dir)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
