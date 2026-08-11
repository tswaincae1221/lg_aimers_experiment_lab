import numpy as np
import pandas as pd

from src.compare_experiments import METRICS, compare_metrics


def test_compare_metrics_uses_candidate_minus_baseline_delta() -> None:
    baseline = pd.DataFrame(
        {
            "season": [2022, 2023],
            **{metric: [1.0, 2.0] for metric in METRICS},
        }
    )
    candidate = pd.DataFrame(
        {
            "season": [2022, 2023],
            **{metric: [0.9, 1.8] for metric in METRICS},
        }
    )

    result = compare_metrics(baseline, candidate)

    assert np.isclose(result.loc[0, "brier_delta"], -0.1)
    assert np.isclose(result.loc[1, "brier_delta"], -0.2)
    assert result.loc[2, "season"] == "mean"
    assert np.isclose(result.loc[2, "brier_delta"], -0.15)
