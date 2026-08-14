import numpy as np
import pandas as pd

from src.first_model_features import (
    ASOF_TREND_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
)
from src.lofo_trackman_runner import _feature_catalog, _write_rankings
from src.experiment_features import BATTER_THREAT_INTERACTION_FEATURES


def test_requested_feature_catalog_has_47_plus_6_plus_13() -> None:
    features = [
        *RAW_NUMERIC_FEATURES,
        *RAW_CATEGORICAL_FEATURES,
        *BATTER_THREAT_INTERACTION_FEATURES,
        *ASOF_TREND_FEATURES,
    ]
    catalog = _feature_catalog(features)

    assert len(catalog) == 66
    assert catalog["feature_group"].value_counts().to_dict() == {
        "basic": 47,
        "trend": 13,
        "teammate_important": 6,
    }


def test_lofo_and_addback_directions_are_written_correctly(tmp_path) -> None:
    history = pd.DataFrame(
        [
            {
                "run_signature": "sig",
                "status": "success",
                "model": "hist_gbdt",
                "experiment_type": "baseline",
                "item": "base_66",
                "brier": 0.200,
            },
            {
                "run_signature": "sig",
                "status": "success",
                "model": "hist_gbdt",
                "experiment_type": "lofo",
                "item": "season",
                "brier": 0.205,
            },
            {
                "run_signature": "sig",
                "status": "success",
                "model": "hist_gbdt",
                "experiment_type": "trackman_individual",
                "item": "tm_prev_rel_height_std",
                "brier": 0.198,
            },
        ]
    )
    base_catalog = pd.DataFrame(
        [{"feature": "season", "feature_group": "basic", "group_label_ko": "기본"}]
    )
    tm_catalog = pd.DataFrame(
        [
            {
                "feature": "tm_prev_rel_height_std",
                "group": "release_repeatability",
                "group_label_ko": "릴리스 재현성",
            }
        ]
    )

    _write_rankings(history, "sig", base_catalog, tm_catalog, tmp_path)

    lofo = pd.read_csv(tmp_path / "lofo_importance_by_model.csv")
    addback = pd.read_csv(tmp_path / "trackman_addback_by_model.csv")
    assert np.isclose(lofo.loc[0, "delta_brier"], 0.005)
    assert np.isclose(addback.loc[0, "brier_improvement"], 0.002)
