from __future__ import annotations

from collections.abc import Iterable

# Compact profiles are intentionally explicit. A changed upstream feature name
# should fail loudly instead of silently changing the experiment population.
COMPACT_CORE_GROUPS: dict[str, list[str]] = {
    "game_context": [
        "season",
        "game_month",
        "inning",
        "top_bottom",
        "game_type",
        "balls_before",
        "strikes_before",
        "outs_before",
        "run_total_before",
        "score_diff_pitcher_team",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
        "home_win_expectancy",
        "li",
        "pitcher_hand",
        "batter_hand",
        "base_state",
        "full_count",
        "risp",
        "bases_loaded",
        "close_game",
        "late_inning",
        "platoon_matchup",
    ],
    "official_asof": [
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "asof_pitcher_n_log1p",
        "asof_batter_n_log1p",
        "asof_pitcher_pitchmix_n_log1p",
    ],
    "leakage_safe_history": [
        "pitcher_career_rate",
        "pitcher_career_log_n",
        "pitcher_prev_season_rate",
        "pitcher_prev_season_log_n",
        "pitcher_count_rate",
        "pitcher_count_log_n",
        "pitcher_runner_rate",
        "pitcher_runner_log_n",
        "pitcher_risp_rate",
        "pitcher_risp_log_n",
        "pitcher_batter_hand_rate",
        "pitcher_batter_hand_log_n",
        "pitcher_recent20_rate",
        "pitcher_recent50_rate",
        "pitcher_recent100_rate",
        "pitcher_season_gap",
        "pitcher_history_missing",
        "prev_vs_career_rate",
        "recent50_vs_career_rate",
    ],
    "rowwise_interactions": [
        "asof_pitcher_success_trend_1v5",
        "asof_pitcher_success_trend_3v5",
        "asof_pitcher_middle_trend_1v5",
        "asof_pitcher_middle_trend_3v5",
        "asof_pitcher_batter_success_gap",
        "asof_pitcher_batter_middle_gap",
        "asof_pitcher_strike_ball_gap",
        "li_log1p",
        "pressure_score",
        "risp_two_out",
        "bases_loaded_less_two_out",
        "runner_out_pressure",
        "count_pressure",
        "pitchmix_entropy",
        "pitchmix_dominant_rate",
        "pitchmix_fastball_breaking_gap",
        "pitchmix_available",
        "pitcher_asof_reliability",
        "batter_asof_reliability",
        "pitchmix_reliability",
        "pitcher_history_reliability",
        "asof_pitcher_batter_weighted_gap",
        "information_source_count",
    ],
}


COMPACT_TRACKMAN_COLUMNS = [
    "tm_mapping_purity",
    "tm_mapping_accepted",
    "tm_career_pitch_n",
    "tm_career_rel_speed_mean",
    "tm_career_spin_rate_mean",
    "tm_career_induced_vert_break_mean",
    "tm_career_horz_break_mean",
    "tm_career_extension_mean",
    "tm_career_zone_speed_mean",
    "tm_prev_rel_speed_mean",
    "tm_prev_spin_rate_mean",
    "tm_prev_induced_vert_break_mean",
    "tm_prev_horz_break_mean",
    "tm_prev_zone_speed_mean",
    "tm_career_usage_fastball_rate",
    "tm_career_usage_breaking_rate",
    "tm_career_usage_offspeed_rate",
    "tm_season_gap",
    "tm_history_missing",
]


def _flatten(groups: Iterable[list[str]]) -> list[str]:
    columns = [column for group in groups for column in group]
    duplicate = sorted({column for column in columns if columns.count(column) > 1})
    if duplicate:
        raise ValueError(f"Compact feature profile contains duplicates: {duplicate}")
    return columns


COMPACT_CORE_COLUMNS = _flatten(COMPACT_CORE_GROUPS.values())

FEATURE_PROFILES = {
    "compact_core": COMPACT_CORE_COLUMNS,
    "compact_core_trackman": [*COMPACT_CORE_COLUMNS, *COMPACT_TRACKMAN_COLUMNS],
}

FEATURE_PROFILE_DESCRIPTIONS = {
    "compact_core": (
        "선수 ID·팀 ID·Trackman 원자료와 중복 파생치를 제거하고 경기 문맥, 공식 asof, "
        "season 이전 이력, 해석 가능한 상호작용만 유지"
    ),
    "compact_core_trackman": (
        "compact_core에 매핑 품질, 구속·회전·무브먼트·구종 구성의 대표 Trackman 19개를 추가"
    ),
}


def resolve_feature_profile(name: str) -> list[str]:
    if name not in FEATURE_PROFILES:
        raise ValueError(
            f"Unknown feature profile '{name}'. Available: {sorted(FEATURE_PROFILES)}"
        )
    return list(FEATURE_PROFILES[name])
