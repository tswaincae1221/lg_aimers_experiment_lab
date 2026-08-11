from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
        description="Export the exact leakage-safe feature matrix used by the first model."
    )
    parser.add_argument("--train", required=True, help="Official train.csv path")
    parser.add_argument("--test", help="Official test.csv path")
    parser.add_argument("--trackman", help="Official trackman_history.csv path")
    parser.add_argument("--mapping", help="pitcher_id to pitcher_trackman_id CSV path")
    parser.add_argument("--output-dir", default="results/preprocessed_first_model")
    parser.add_argument("--accepted-mapping-grades", nargs="+", default=["확정", "높음"])
    parser.add_argument("--smoothing", type=float, default=50.0)
    parser.add_argument(
        "--asof-trends",
        action="store_true",
        help="Include the 13 V1.1 asof trend/contrast features.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "pickle"],
        default="parquet",
        help="Parquet is recommended. Pickle is a dependency-light fallback.",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--row-group-size", type=int, default=100_000)
    parser.add_argument("--preview-rows", type=int, default=100)
    parser.add_argument(
        "--max-rows-per-season",
        type=int,
        help="Development-only row cap. Omit this option for the complete result.",
    )
    parser.add_argument(
        "--max-trackman-rows",
        type=int,
        help="Development-only Trackman cap. Omit this option for the complete result.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_by_season(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame
    LOGGER.warning("Development row cap is active: max_rows_per_season=%d", limit)
    parts = [
        group.sample(min(len(group), limit), random_state=seed)
        for _, group in frame.groupby("season", observed=True, sort=True)
    ]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def build_export_frame(
    bundle: FeatureBundle,
    rows: np.ndarray,
    include_target: bool,
) -> pd.DataFrame:
    """Return row identifiers plus the exact model matrix for the requested rows."""
    features = bundle.features.iloc[rows].reset_index(drop=True)
    identifiers = pd.DataFrame({ID_COLUMN: bundle.row_ids.iloc[rows].to_numpy()})
    if include_target:
        target = bundle.target.iloc[rows]
        if target.isna().any():
            raise ValueError("Training target contains missing values")
        identifiers[TARGET] = target.astype("int8").to_numpy()
    return pd.concat([identifiers, features], axis=1)


def _logical_missing_count(series: pd.Series) -> int:
    count = int(series.isna().sum())
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(
        series.dtype
    ):
        count += int(series.astype("string").eq("__MISSING__").sum())
    return count


def build_feature_schema(bundle: FeatureBundle) -> pd.DataFrame:
    train_features = bundle.features.iloc[bundle.train_rows]
    test_features = bundle.features.iloc[bundle.test_rows]
    rows: list[dict[str, object]] = []
    categorical = set(bundle.categorical_features)
    for position, feature in enumerate(bundle.features.columns, start=1):
        train_missing_n = _logical_missing_count(train_features[feature])
        test_missing_n = (
            _logical_missing_count(test_features[feature]) if len(test_features) else 0
        )
        rows.append(
            {
                "position": position,
                "feature": feature,
                "dtype": str(bundle.features[feature].dtype),
                "is_categorical": feature in categorical,
                "train_missing_n": train_missing_n,
                "train_missing_rate": float(train_missing_n / len(train_features)),
                "test_missing_n": test_missing_n,
                "test_missing_rate": (
                    float(test_missing_n / len(test_features))
                    if len(test_features)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def write_feature_table(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    args: argparse.Namespace,
) -> Path:
    if args.format == "pickle":
        path = output_dir / f"{stem}.pkl.gz"
        LOGGER.info("Writing %s: rows=%d columns=%d", path, *frame.shape)
        frame.to_pickle(path, compression={"method": "gzip", "compresslevel": 1})
        return path

    path = output_dir / f"{stem}.parquet"
    LOGGER.info("Writing %s: rows=%d columns=%d", path, *frame.shape)
    frame.to_parquet(
        path,
        engine="pyarrow",
        compression=args.compression,
        index=False,
        row_group_size=args.row_group_size,
    )
    return path


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

    if bool(args.trackman) != bool(args.mapping):
        raise ValueError("--trackman and --mapping must be supplied together")

    trackman_summary = None
    mapping = None
    if args.trackman:
        LOGGER.info("Aggregating Trackman data: %s", args.trackman)
        trackman_summary = aggregate_trackman(args.trackman, nrows=args.max_trackman_rows)
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

    train_output = build_export_frame(bundle, bundle.train_rows, include_target=True)
    train_output_path = write_feature_table(train_output, output_dir, "train_features", args)
    del train_output

    test_output_path = None
    if len(bundle.test_rows):
        test_output = build_export_frame(bundle, bundle.test_rows, include_target=False)
        test_output_path = write_feature_table(test_output, output_dir, "test_features", args)
        test_output.head(args.preview_rows).to_csv(
            output_dir / "test_features_preview.csv", index=False, encoding="utf-8-sig"
        )
        del test_output

    train_preview = build_export_frame(
        bundle,
        bundle.train_rows[: args.preview_rows],
        include_target=True,
    )
    train_preview.to_csv(
        output_dir / "train_features_preview.csv", index=False, encoding="utf-8-sig"
    )

    schema = build_feature_schema(bundle)
    schema.to_csv(output_dir / "feature_schema.csv", index=False, encoding="utf-8-sig")

    summary = {
        "train_source": str(Path(args.train)),
        "test_source": str(Path(args.test)) if args.test else None,
        "trackman_source": str(Path(args.trackman)) if args.trackman else None,
        "mapping_source": str(Path(args.mapping)) if args.mapping else None,
        "train_rows": int(len(bundle.train_rows)),
        "test_rows": int(len(bundle.test_rows)),
        "model_feature_count": int(bundle.features.shape[1]),
        "categorical_feature_count": int(len(bundle.categorical_features)),
        "numeric_feature_count": int(
            bundle.features.shape[1] - len(bundle.categorical_features)
        ),
        "categorical_features": bundle.categorical_features,
        "smoothing": args.smoothing,
        "asof_trends": bool(args.asof_trends),
        "accepted_mapping_grades": args.accepted_mapping_grades,
        "history_rule": "For season s, use labeled pitch and Trackman rows with season < s.",
        "output_format": args.format,
        "train_output": str(train_output_path),
        "test_output": str(test_output_path) if test_output_path else None,
        "parquet_compression": args.compression if args.format == "parquet" else None,
        "row_group_size": args.row_group_size,
    }
    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Done: train=%d test=%d model_features=%d",
        len(bundle.train_rows),
        len(bundle.test_rows),
        bundle.features.shape[1],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
