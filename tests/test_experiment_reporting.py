import numpy as np
import pandas as pd

from src.experiment_reporting import (
    add_baseline_deltas,
    add_comparison_deltas,
    add_preprocessing_deltas,
    compute_metrics,
)


def test_metrics_and_baseline_delta_direction() -> None:
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert metrics["brier"] < 0.25
    assert 0 <= metrics["ece_10bin"] <= 1

    board = pd.DataFrame(
        {
            "experiment": ["v1", "candidate"],
            "brier": [0.249, 0.247],
        }
    )
    result = add_baseline_deltas(board, "v1")
    candidate = result.loc[result["experiment"] == "candidate"].iloc[0]
    assert np.isclose(candidate["brier_delta_vs_baseline"], -0.002)
    assert candidate["brier_improvement_pct"] > 0


def test_pairwise_comparison_delta_uses_declared_experiment() -> None:
    board = pd.DataFrame(
        {
            "experiment": ["asof_logistic", "asof_threat_logistic"],
            "comparison_experiment": [None, "asof_logistic"],
            "brier": [0.2490, 0.2485],
        }
    )
    result = add_comparison_deltas(board)
    candidate = result.loc[result["experiment"] == "asof_threat_logistic"].iloc[0]
    assert np.isclose(candidate["comparison_brier"], 0.2490)
    assert np.isclose(candidate["brier_delta_vs_comparison"], -0.0005)
    assert candidate["brier_improvement_vs_comparison_pct"] > 0


def test_preprocessing_delta_uses_declared_raw_experiment() -> None:
    board = pd.DataFrame(
        {
            "experiment": ["raw_hist", "engineered_hist"],
            "preprocessing_comparison_experiment": [None, "raw_hist"],
            "brier": [0.2500, 0.2475],
        }
    )
    result = add_preprocessing_deltas(board)
    candidate = result.loc[result["experiment"] == "engineered_hist"].iloc[0]
    assert np.isclose(candidate["preprocessing_comparison_brier"], 0.2500)
    assert np.isclose(candidate["brier_delta_vs_preprocessing"], -0.0025)
    assert candidate["brier_improvement_vs_preprocessing_pct"] > 0
