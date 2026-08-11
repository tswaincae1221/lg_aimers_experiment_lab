from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "brier",
    "brier_skill_score",
    "logloss",
    "auc",
    "prediction_mean",
    "target_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V1 baseline and V1.1 asof-trend season-forward CV results."
    )
    parser.add_argument("--baseline", required=True, help="V1 metrics.csv path")
    parser.add_argument("--candidate", required=True, help="V1.1 metrics.csv path")
    parser.add_argument(
        "--output",
        default="results/asof_trend_comparison.csv",
        help="Comparison CSV path",
    )
    return parser.parse_args()


def compare_metrics(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    required = {"season", *METRICS}
    for label, frame in [("baseline", baseline), ("candidate", candidate)]:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} metrics are missing columns: {missing}")

    left = baseline[["season", *METRICS]].copy()
    right = candidate[["season", *METRICS]].copy()
    merged = left.merge(right, on="season", suffixes=("_v1", "_v1_1"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Baseline and candidate must contain exactly the same CV seasons")

    for metric in METRICS:
        merged[f"{metric}_delta"] = merged[f"{metric}_v1_1"] - merged[f"{metric}_v1"]

    mean_row: dict[str, float | str] = {"season": "mean"}
    for column in merged.columns:
        if column != "season":
            mean_row[column] = float(merged[column].mean())
    return pd.concat([merged, pd.DataFrame([mean_row])], ignore_index=True)


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    comparison = compare_metrics(baseline, candidate)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)

    display_columns = [
        "season",
        "brier_v1",
        "brier_v1_1",
        "brier_delta",
        "brier_skill_score_delta",
        "auc_delta",
    ]
    print(comparison[display_columns].to_string(index=False))
    mean_brier_delta = float(comparison.loc[comparison["season"] == "mean", "brier_delta"].iloc[0])
    if mean_brier_delta < 0:
        print(f"\nV1.1 improved mean Brier Score by {-mean_brier_delta:.8f}.")
    elif mean_brier_delta > 0:
        print(f"\nV1.1 worsened mean Brier Score by {mean_brier_delta:.8f}.")
    else:
        print("\nV1.1 and V1 have the same mean Brier Score.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
