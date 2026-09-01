from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.hgb_feature_selection_v2 import (
    EngineeredData,
    add_season_trend_features,
    build_main_features,
    build_trackman_features,
    frequency_encode,
    leakage_safe_season_trend_prior,
    parse_args,
)


def _main_rows() -> pd.DataFrame:
    rows = []
    for i, season in enumerate([2022, 2023, 2024]):
        rows.append(
            {
                "row_id": f"r{i}",
                "season": season,
                "game_month": 6,
                "game_dayofweek": 1,
                "inning": 5,
                "top_bottom": "T",
                "game_type": "R",
                "balls_before": 2,
                "strikes_before": 1,
                "outs_before": 1,
                "run_top_before": 1,
                "run_bot_before": 1,
                "run_total_before": 2,
                "score_diff_home": 0,
                "score_diff_pitcher_team": 0,
                "runner_on_1b": 0,
                "runner_on_2b": 1,
                "runner_on_3b": 0,
                "num_runners_on": 1,
                "base_state": "_2_",
                "home_win_expectancy": 50.0,
                "away_win_expectancy": 50.0,
                "li": 1.2,
                "pitcher_id": 10,
                "batter_id": 20,
                "pitcher_hand": "R",
                "batter_hand": "L",
                "pitcher_team_id": 1,
                "batter_team_id": 2,
                "asof_pitcher_n": 200,
                "asof_pitcher_success_rate": 0.55,
                "asof_pitcher_reverse_rate": 0.12,
                "asof_pitcher_middle_rate": 0.20,
                "asof_pitcher_ball_rate": 0.35,
                "asof_pitcher_strike_rate": 0.65,
                "asof_pitcher_prev1_game_success_rate": 0.58,
                "asof_pitcher_prev3_game_success_rate": 0.56,
                "asof_pitcher_prev5_game_success_rate": 0.54,
                "asof_pitcher_prev1_game_middle_rate": 0.18,
                "asof_pitcher_prev3_game_middle_rate": 0.19,
                "asof_pitcher_prev5_game_middle_rate": 0.21,
                "asof_batter_n": 150,
                "asof_batter_success_rate": 0.52,
                "asof_batter_middle_rate": 0.22,
                "asof_pitcher_pitchmix_n": 200,
                "asof_pitcher_fastball_rate": 0.55,
                "asof_pitcher_breaking_rate": 0.30,
                "asof_pitcher_offspeed_rate": 0.15,
                "control_success": i % 2,
            }
        )
    return pd.DataFrame(rows)


def _group_row(season: int, scale: float) -> dict:
    row = {
        "season": season,
        "pitcher_trackman_id": "900",
        "pitch_type_group": "fastball",
        "tm_group_pitch_n": 100,
    }
    for metric in (
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
        "extension",
        "rel_height",
        "rel_side",
        "zone_speed",
    ):
        mean = 100.0 if metric == "spin_rate" else 10.0
        sd = scale
        n = 100.0
        row[f"{metric}__n"] = n
        row[f"{metric}__sum"] = mean * n
        row[f"{metric}__sqsum"] = (n - 1) * sd**2 + n * mean**2
    return row


def test_main_features_match_requested_design() -> None:
    built = build_main_features(_main_rows())
    expected = {
        "pitcher_success_shrunk",
        "success_form_1",
        "success_trend_1v5",
        "middle_form_1",
        "recent_success_volatility",
        "count_state",
        "hand_pair",
        "pitcher_batter_success_gap",
        "threat_x_ball",
        "base_out_state",
        "li_log1p",
        "pitchmix_entropy",
    }
    assert expected.issubset(built.features.columns)
    assert "pitcher_id" not in built.features.columns
    assert "batter_id" not in built.features.columns
    assert built.blocks["recent_form"]
    assert built.blocks["count_intent"]


def test_trackman_features_ignore_same_or_future_season() -> None:
    main = build_main_features(_main_rows())
    mapping = pd.DataFrame(
        {
            "pitcher_id": ["10"],
            "pitcher_trackman_id": ["900"],
            "tm_mapping_grade": ["확정"],
            "tm_mapping_purity": [1.0],
        }
    )
    base_stats = pd.DataFrame([_group_row(2022, 1.0)])
    future_stats = pd.DataFrame([_group_row(2022, 1.0), _group_row(2024, 50.0)])
    context_cols = [
        "season",
        "pitcher_trackman_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "pitch_type_group",
        "context_n",
    ]
    drift_cols = ["season", "pitcher_trackman_id", "trackman_game_id"]
    for metric in ("rel_speed", "rel_height", "rel_side", "extension"):
        drift_cols += [
            f"{metric}__n",
            f"{metric}__sum_x",
            f"{metric}__sum_x2",
            f"{metric}__sum_y",
            f"{metric}__sum_xy",
        ]
    empty_context = pd.DataFrame(columns=context_cols)
    empty_drift = pd.DataFrame(columns=drift_cols)

    base, _, _ = build_trackman_features(main, mapping, base_stats, empty_context, empty_drift)
    future, _, _ = build_trackman_features(main, mapping, future_stats, empty_context, empty_drift)
    row_2024 = main.seasons.eq(2024)
    cols = [column for column in base.columns if "dispersion" in column]
    pd.testing.assert_frame_equal(base.loc[row_2024, cols], future.loc[row_2024, cols])
    assert "tm_expected_release_dispersion" in base.columns
    assert not any(column.startswith("tm_fastball_release") for column in base.columns)


def test_frequency_encoding_is_fit_on_training_only() -> None:
    train = pd.DataFrame({"cat": ["a", "a", "b"], "num": [1.0, np.nan, 3.0]})
    valid = pd.DataFrame({"cat": ["c"], "num": [np.nan]})
    encoded = frequency_encode(train, valid, ["cat"])
    assert np.isclose(encoded.x_train[0, 0], 2 / 3)
    assert encoded.x_valid[0, 0] == 0.0
    assert encoded.x_valid[0, 1] == 2.0


def _season_trend_rows() -> pd.DataFrame:
    rows = []
    rates = {
        2019: 0.56,
        2020: 0.53,
        2021: 0.53,
        2022: 0.52,
        2023: 0.50,
        2024: 0.10,
    }
    for season, rate in rates.items():
        n = 100
        ones = int(round(rate * n))
        for i in range(n):
            rows.append(
                {
                    "season": season,
                    "control_success": int(i < ones),
                    "asof_pitcher_success_rate": 0.55,
                    "asof_pitcher_n": 100,
                    "asof_batter_success_rate": 0.52,
                    "asof_batter_n": 80,
                }
            )
    return pd.DataFrame(rows)


def test_season_trend_prior_uses_only_strictly_previous_seasons() -> None:
    raw = _season_trend_rows()
    trend_a = leakage_safe_season_trend_prior(raw, raw["season"], min_history_seasons=4)

    changed = raw.copy()
    changed.loc[changed["season"].eq(2024), "control_success"] = 1
    trend_b = leakage_safe_season_trend_prior(
        changed,
        changed["season"],
        min_history_seasons=4,
    )

    for season in (2020, 2021, 2022, 2023, 2024):
        a = trend_a.loc[raw["season"].eq(season), "season_trend_prior"].iloc[0]
        b = trend_b.loc[changed["season"].eq(season), "season_trend_prior"].iloc[0]
        assert np.isclose(a, b)

    prior_2024 = trend_a.loc[raw["season"].eq(2024), "season_trend_prior"].iloc[0]
    assert 0.47 < prior_2024 < 0.51


def test_season_trend_features_replace_shrinkage_prior() -> None:
    raw = _season_trend_rows()
    index = raw.index
    engineered = EngineeredData(
        features=pd.DataFrame(
            {
                "pitcher_success_shrunk": 0.5,
                "batter_success_shrunk": 0.5,
            },
            index=index,
        ),
        target=raw["control_success"],
        row_ids=pd.Series(np.arange(len(raw)), index=index),
        seasons=raw["season"],
        blocks={
            "control_prior": ["pitcher_success_shrunk", "batter_success_shrunk"]
        },
        categorical_features=[],
    )
    augmented = add_season_trend_features(
        engineered,
        raw,
        shrinkage=50.0,
        min_history_seasons=4,
    )
    assert {
        "season_trend_prior",
        "season_trend_slope",
        "season_trend_history_n",
    }.issubset(augmented.features.columns)
    assert augmented.blocks["season_trend"] == [
        "season_trend_prior",
        "season_trend_slope",
        "season_trend_history_n",
    ]
    assert not np.allclose(augmented.features["pitcher_success_shrunk"], 0.5)


def test_full_mode_forces_350_fixed_iterations(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--train",
            "train.csv",
            "--trackman",
            "trackman.csv",
            "--mode",
            "full",
            "--max-iter",
            "7",
        ],
    )
    args = parse_args()
    assert args.max_iter == 350


def test_quick_mode_keeps_small_iteration_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--train",
            "train.csv",
            "--trackman",
            "trackman.csv",
            "--mode",
            "quick",
            "--max-iter",
            "350",
        ],
    )
    args = parse_args()
    assert args.max_iter == 120
