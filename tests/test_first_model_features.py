from __future__ import annotations

import numpy as np
import pandas as pd

from src.first_model_features import (
    ASOF_TREND_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET,
    TRACKMAN_NUMERIC_COLUMNS,
    assemble_features,
    assemble_raw_features,
    build_current_features,
    build_trackman_features,
)


def make_rows() -> pd.DataFrame:
    rows = []
    for index, (season, pitcher, target) in enumerate(
        [(2019, 10, 1), (2019, 10, 0), (2020, 10, 1), (2020, 20, 0)]
    ):
        rows.append(
            {
                "row_id": f"TRAIN_{index}",
                "season": season,
                "game_month": 4,
                "game_dayofweek": 1,
                "inning": 1,
                "top_bottom": "T",
                "game_type": "R",
                "balls_before": index % 4,
                "strikes_before": index % 3,
                "outs_before": 0,
                "run_top_before": 0,
                "run_bot_before": 0,
                "run_total_before": 0,
                "score_diff_home": 0,
                "score_diff_pitcher_team": 0,
                "runner_on_1b": 0,
                "runner_on_2b": 0,
                "runner_on_3b": 0,
                "num_runners_on": 0,
                "base_state": "___",
                "home_win_expectancy": 50.0,
                "away_win_expectancy": 50.0,
                "li": 1.0,
                "pitcher_id": pitcher,
                "batter_id": 100 + index,
                "pitcher_hand": 1,
                "batter_hand": 2,
                "pitcher_team_id": 1,
                "batter_team_id": 2,
                "asof_pitcher_n": index,
                "asof_pitcher_success_rate": np.nan,
                "asof_pitcher_reverse_rate": np.nan,
                "asof_pitcher_middle_rate": np.nan,
                "asof_pitcher_ball_rate": np.nan,
                "asof_pitcher_strike_rate": np.nan,
                "asof_pitcher_prev1_game_success_rate": np.nan,
                "asof_pitcher_prev3_game_success_rate": np.nan,
                "asof_pitcher_prev5_game_success_rate": np.nan,
                "asof_pitcher_prev1_game_middle_rate": np.nan,
                "asof_pitcher_prev3_game_middle_rate": np.nan,
                "asof_pitcher_prev5_game_middle_rate": np.nan,
                "asof_batter_n": 0,
                "asof_batter_success_rate": np.nan,
                "asof_batter_middle_rate": np.nan,
                "asof_pitcher_pitchmix_n": 0,
                "asof_pitcher_fastball_rate": np.nan,
                "asof_pitcher_breaking_rate": np.nan,
                "asof_pitcher_offspeed_rate": np.nan,
                TARGET: target,
            }
        )
    return pd.DataFrame(rows)


def test_history_features_use_only_prior_seasons() -> None:
    train = make_rows()
    bundle = assemble_features(train, smoothing=0.0)

    first_2020 = bundle.features.loc[2]
    assert first_2020["pitcher_career_n"] == 2
    assert first_2020["pitcher_career_rate"] == 0.5

    new_pitcher_2020 = bundle.features.loc[3]
    assert new_pitcher_2020["pitcher_history_missing"] == 1


def test_raw_minimal_features_keep_only_official_columns() -> None:
    rows = make_rows()
    rows.loc[0, "li"] = np.inf
    rows.loc[1, "pitcher_hand"] = np.nan
    bundle = assemble_raw_features(rows)

    assert bundle.features.columns.tolist() == [
        *RAW_NUMERIC_FEATURES,
        *RAW_CATEGORICAL_FEATURES,
    ]
    assert "count_state" not in bundle.features
    assert "pitcher_career_rate" not in bundle.features
    assert "tm_career_pitch_n" not in bundle.features
    assert np.isnan(bundle.features.loc[0, "li"])
    assert str(bundle.features.loc[1, "pitcher_hand"]) == "__MISSING__"
    assert bundle.categorical_features == RAW_CATEGORICAL_FEATURES


def test_current_situation_derivations() -> None:
    bundle = assemble_features(make_rows(), smoothing=1.0)
    row = bundle.features.loc[3]
    assert row["count_state"] == "3-0"
    assert row["three_ball"] == 1
    assert row["is_tie"] == 1
    assert row["risp"] == 0


def test_asof_trend_features_are_optional_and_use_only_source_columns() -> None:
    rows = make_rows()
    rows.loc[0, [
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_ball_rate",
    ]] = [0.60, 0.70, 0.62, 0.55, 0.40, 0.48, 0.43, 0.39, 0.51, 0.36, 0.58, 0.42]

    baseline = build_current_features(rows)
    augmented = build_current_features(rows, include_asof_trends=True)

    assert not set(ASOF_TREND_FEATURES).intersection(baseline.columns)
    assert set(ASOF_TREND_FEATURES).issubset(augmented.columns)
    assert np.isclose(augmented.loc[0, "asof_pitcher_success_trend_1v5"], 0.15)
    assert np.isclose(augmented.loc[0, "asof_pitcher_middle_trend_3v5"], 0.04)
    assert np.isclose(augmented.loc[0, "asof_pitcher_batter_success_gap"], 0.09)
    assert np.isclose(augmented.loc[0, "asof_pitcher_strike_ball_gap"], 0.16)
    assert augmented.loc[0, "asof_success_trend_source_n"] == 3
    assert augmented.loc[1, "asof_success_trend_source_n"] == 0
    assert np.isnan(augmented.loc[1, "asof_pitcher_success_trend_1v5"])


def test_asof_trend_flag_adds_exactly_13_features() -> None:
    baseline = assemble_features(make_rows(), smoothing=1.0)
    augmented = assemble_features(
        make_rows(),
        smoothing=1.0,
        include_asof_trends=True,
    )

    assert augmented.features.shape[1] == baseline.features.shape[1] + 13
    added = [
        column
        for column in augmented.features.columns
        if column not in baseline.features.columns
    ]
    assert added == ASOF_TREND_FEATURES


def test_trackman_features_use_only_prior_seasons() -> None:
    train = make_rows()
    current = build_current_features(train)
    summary_rows = []
    for season, speed in [(2019, 140.0), (2020, 150.0)]:
        row = {
            "pitcher_trackman_id": "900",
            "season": season,
            "tm_pitch_n": 10,
            "usage_fastball__sum": 10,
            "usage_breaking__sum": 0,
            "usage_offspeed__sum": 0,
            "usage_other__sum": 0,
        }
        for column in TRACKMAN_NUMERIC_COLUMNS:
            mean = speed if column == "rel_speed" else 1.0
            row[f"{column}__n"] = 10
            row[f"{column}__sum"] = 10 * mean
            row[f"{column}__sqsum"] = 10 * mean**2
        summary_rows.append(row)

    mapping = pd.DataFrame(
        {
            "pitcher_id": ["10"],
            "pitcher_trackman_id": ["900"],
            "신뢰등급": ["확정"],
            "매칭순도": [1.0],
            "mapping_accepted": [True],
        }
    )
    features = build_trackman_features(current, pd.DataFrame(summary_rows), mapping)

    assert np.isnan(features.loc[0, "tm_career_rel_speed_mean"])
    assert features.loc[2, "tm_career_rel_speed_mean"] == 140.0
