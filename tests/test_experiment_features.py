import numpy as np
import pandas as pd

from src.experiment_features import (
    ASOF_TREND_FEATURES,
    BATTER_THREAT_INTERACTION_FEATURES,
    PITCHMIX_FEATURES,
    RELIABILITY_FEATURES,
    SITUATION_FEATURES,
    apply_feature_set,
)
from src.compact_feature_profiles import (
    COMPACT_CORE_COLUMNS,
    COMPACT_TRACKMAN_COLUMNS,
)


def make_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "li": [2.5, 0.8],
            "late_inning": [1, 0],
            "close_game": [1, 1],
            "risp": [1, 0],
            "outs_before": [2, 0],
            "bases_loaded": [0, 0],
            "num_runners_on": [2, 0],
            "full_count": [1, 0],
            "base_state": pd.Categorical(["12_", "___"]),
            "count_state": pd.Categorical(["3-2", "0-0"]),
            "asof_pitcher_n": [100, 0],
            "asof_batter_n": [40, 0],
            "asof_pitcher_pitchmix_n": [100, 0],
            "asof_pitcher_success_rate": [0.60, np.nan],
            "asof_pitcher_reverse_rate": [0.20, np.nan],
            "asof_pitcher_middle_rate": [0.40, np.nan],
            "asof_pitcher_strike_rate": [0.58, np.nan],
            "asof_pitcher_ball_rate": [0.42, np.nan],
            "asof_pitcher_prev1_game_success_rate": [0.70, np.nan],
            "asof_pitcher_prev3_game_success_rate": [0.62, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.55, np.nan],
            "asof_pitcher_prev1_game_middle_rate": [0.48, np.nan],
            "asof_pitcher_prev3_game_middle_rate": [0.43, np.nan],
            "asof_pitcher_prev5_game_middle_rate": [0.39, np.nan],
            "asof_batter_success_rate": [0.51, np.nan],
            "asof_batter_middle_rate": [0.36, np.nan],
            "asof_pitcher_fastball_rate": [0.5, np.nan],
            "asof_pitcher_breaking_rate": [0.3, np.nan],
            "asof_pitcher_offspeed_rate": [0.2, np.nan],
            "pitcher_career_n": [1000, np.nan],
            "pitcher_history_missing": [0, 1],
            "tm_career_pitch_n": [500, np.nan],
            "pitcher_id": pd.Categorical(["10", "20"]),
            "batter_id": pd.Categorical(["100", "200"]),
            "tm_history_missing": [0, 1],
            "pitcher_hand": pd.Categorical(["R", "L"]),
            "batter_hand": pd.Categorical(["R", "R"]),
        }
    )


def test_combined_feature_blocks_have_expected_columns_and_values() -> None:
    base = make_base()
    features, categorical, report = apply_feature_set(
        base,
        ["base_state", "count_state", "pitcher_id", "batter_id"],
        {"blocks": ["asof_trend", "situation", "pitchmix", "reliability"]},
    )

    expected = set(
        ASOF_TREND_FEATURES
        + SITUATION_FEATURES
        + PITCHMIX_FEATURES
        + RELIABILITY_FEATURES
    )
    assert expected.issubset(features.columns)
    assert report["added_feature_count"] == len(expected)
    assert np.isclose(features.loc[0, "asof_pitcher_success_trend_1v5"], 0.15)
    assert features.loc[0, "late_close_high_li"] == 1
    assert np.isclose(features.loc[0, "pitchmix_observed_sum"], 1.0)
    assert features.loc[0, "information_source_count"] == 5
    assert "base_out_state" in categorical
    assert "count_base_out_state" in categorical


def test_drop_patterns_support_id_and_trackman_ablations() -> None:
    base = make_base()
    base["trackman_reliability"] = [0.8, 0.0]
    features, categorical, report = apply_feature_set(
        base,
        ["pitcher_id", "batter_id"],
        {
            "blocks": [],
            "drop_patterns": ["pitcher_id", "batter_id", "tm_*", "trackman_*"],
        },
    )
    assert "pitcher_id" not in features
    assert "batter_id" not in features
    assert not any(column.startswith("tm_") for column in features)
    assert categorical == []
    assert set(report["dropped_features"]) == {
        "pitcher_id",
        "batter_id",
        "tm_career_pitch_n",
        "tm_history_missing",
        "trackman_reliability",
    }


def test_batter_threat_interactions_match_proposed_definitions() -> None:
    base = make_base()
    features, categorical, report = apply_feature_set(
        base,
        ["pitcher_hand", "batter_hand"],
        {"blocks": ["batter_threat_interactions"]},
    )

    assert set(BATTER_THREAT_INTERACTION_FEATURES).issubset(features.columns)
    assert report["added_feature_count"] == 6
    assert categorical == ["pitcher_hand", "batter_hand"]
    assert features.loc[0, "hand_match"] == 1.0
    assert features.loc[1, "hand_match"] == 0.0
    assert np.isclose(features.loc[0, "success_interact"], 0.60 * 0.51)
    assert np.isclose(features.loc[0, "reverse_x_middle"], 0.20 * 0.40)
    assert np.isclose(features.loc[0, "threat_x_reverse"], (1 - 0.51) * 0.20)
    assert np.isclose(features.loc[0, "threat_x_middle"], (1 - 0.51) * 0.40)
    assert np.isclose(features.loc[0, "threat_x_ball"], (1 - 0.51) * 0.42)
    assert features.loc[1, "threat_x_ball"] != features.loc[1, "threat_x_ball"]


def test_batter_threat_interactions_preserve_missing_and_reject_invalid_rates() -> None:
    base = make_base()
    base.loc[0, "asof_batter_success_rate"] = 1.2
    base.loc[1, "pitcher_hand"] = np.nan
    features, _, _ = apply_feature_set(
        base,
        ["pitcher_hand", "batter_hand"],
        {"blocks": ["batter_threat_interactions"]},
    )

    assert features.loc[0, "threat_x_ball"] != features.loc[0, "threat_x_ball"]
    assert features.loc[1, "hand_match"] != features.loc[1, "hand_match"]


def test_compact_profiles_are_explicit_and_preserve_source_order() -> None:
    base = make_base()
    for column in COMPACT_CORE_COLUMNS:
        if column not in base:
            base[column] = 0.0

    features, categorical, report = apply_feature_set(
        base,
        ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand"],
        {"blocks": [], "keep_profile": "compact_core"},
    )

    assert features.columns.tolist() == [
        column for column in base.columns if column in COMPACT_CORE_COLUMNS
    ]
    assert "pitcher_id" not in features
    assert "batter_id" not in features
    assert "tm_career_pitch_n" not in features
    assert categorical == ["pitcher_hand", "batter_hand"]
    assert report["final_feature_count"] == len(COMPACT_CORE_COLUMNS)
    assert report["keep_profile"] == "compact_core"


def test_compact_trackman_profile_keeps_only_curated_trackman_columns() -> None:
    base = make_base()
    for column in [*COMPACT_CORE_COLUMNS, *COMPACT_TRACKMAN_COLUMNS]:
        if column not in base:
            base[column] = 0.0
    base["tm_unselected_noise"] = 1.0

    features, _, report = apply_feature_set(
        base,
        ["pitcher_hand", "batter_hand"],
        {"blocks": [], "keep_profile": "compact_core_trackman"},
    )

    assert features.shape[1] == len(COMPACT_CORE_COLUMNS) + len(COMPACT_TRACKMAN_COLUMNS)
    assert set(COMPACT_TRACKMAN_COLUMNS).issubset(features.columns)
    assert "tm_unselected_noise" not in features
    assert report["selection_dropped_feature_count"] > 0


def test_compact_profile_fails_loudly_when_upstream_column_is_missing() -> None:
    base = make_base()
    try:
        apply_feature_set(base, [], {"blocks": [], "keep_profile": "compact_core"})
    except ValueError as exc:
        assert "requires missing columns" in str(exc)
    else:
        raise AssertionError("Incomplete compact profile should fail")
