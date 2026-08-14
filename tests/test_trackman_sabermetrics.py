import pandas as pd

from src.trackman_sabermetrics import (
    GROUP_ORDER,
    build_trackman_catalog,
    classify_trackman_feature,
    group_trackman_features,
)


def test_trackman_features_are_assigned_to_expected_groups() -> None:
    expected = {
        "tm_prev_rel_height_std": "release_repeatability",
        "tm_delta_rel_speed_mean": "velocity_retention",
        "tm_career_spin_rate_mean": "spin_stability",
        "tm_prev_induced_vert_break_std": "movement_repeatability",
        "tm_career_usage_fastball_rate": "repertoire_mix",
        "tm_history_missing": "evidence_reliability",
    }
    assert {
        feature: classify_trackman_feature(feature) for feature in expected
    } == expected


def test_catalog_groups_each_feature_exactly_once() -> None:
    features = [
        "tm_prev_extension_std",
        "tm_prev_zone_speed_mean",
        "tm_delta_spin_rate_mean",
        "tm_career_horz_break_mean",
        "tm_prev_usage_offspeed_rate",
        "tm_mapping_grade",
    ]
    catalog = build_trackman_catalog(features)
    groups = group_trackman_features(catalog)

    assert catalog["feature"].is_unique
    assert set(catalog["feature"]) == set(features)
    assert list(groups) == GROUP_ORDER
    assert sum(len(group_features) for group_features in groups.values()) == len(features)
    assert isinstance(catalog, pd.DataFrame)
