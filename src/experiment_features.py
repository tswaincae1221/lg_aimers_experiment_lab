from __future__ import annotations

from collections.abc import Callable, Iterable
from fnmatch import fnmatch

import numpy as np
import pandas as pd

from src.compact_feature_profiles import (
    FEATURE_PROFILE_DESCRIPTIONS,
    resolve_feature_profile,
)
from src.first_model_features import ASOF_TREND_FEATURES, build_asof_trend_features


FeatureBuilder = Callable[[pd.DataFrame], tuple[pd.DataFrame, list[str]]]


SITUATION_FEATURES = [
    "li_log1p",
    "high_leverage",
    "close_late",
    "late_close_high_li",
    "pressure_score",
    "risp_two_out",
    "bases_loaded_less_two_out",
    "runner_out_pressure",
    "count_pressure",
    "base_out_state",
    "count_base_out_state",
]

PITCHMIX_FEATURES = [
    "pitchmix_entropy",
    "pitchmix_dominant_rate",
    "pitchmix_balance",
    "pitchmix_fastball_breaking_gap",
    "pitchmix_fastball_offspeed_gap",
    "pitchmix_breaking_offspeed_gap",
    "pitchmix_observed_sum",
    "pitchmix_available",
]

RELIABILITY_FEATURES = [
    "pitcher_asof_reliability",
    "batter_asof_reliability",
    "pitchmix_reliability",
    "pitcher_history_reliability",
    "trackman_reliability",
    "asof_pitcher_batter_weighted_gap",
    "information_source_count",
]

BATTER_THREAT_INTERACTION_FEATURES = [
    "hand_match",
    "success_interact",
    "reverse_x_middle",
    "threat_x_reverse",
    "threat_x_middle",
    "threat_x_ball",
]


def _require(frame: pd.DataFrame, columns: Iterable[str], block: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature block '{block}' requires missing columns: {missing}")


def _as_float(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric = out.select_dtypes(exclude=["category"]).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan).astype("float32")
    return out


def _probability(features: pd.DataFrame, column: str) -> pd.Series:
    """Read an official rate while treating out-of-range values as unavailable."""
    values = pd.to_numeric(features[column], errors="coerce")
    return values.where(values.between(0.0, 1.0))


def build_asof_block(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    _require(
        features,
        [
            "asof_pitcher_success_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_strike_rate",
            "asof_pitcher_ball_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
        ],
        "asof_trend",
    )
    return _as_float(build_asof_trend_features(features)), []


def build_situation_block(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    _require(
        features,
        [
            "li",
            "late_inning",
            "close_game",
            "risp",
            "outs_before",
            "bases_loaded",
            "num_runners_on",
            "full_count",
            "base_state",
            "count_state",
        ],
        "situation",
    )
    out = pd.DataFrame(index=features.index)
    li_log = np.log1p(pd.to_numeric(features["li"], errors="coerce").clip(lower=0))
    late = features["late_inning"].fillna(0)
    close = features["close_game"].fillna(0)
    outs = pd.to_numeric(features["outs_before"], errors="coerce")

    out["li_log1p"] = li_log
    out["high_leverage"] = (features["li"] >= 2.0).astype("int8")
    out["close_late"] = (late * close).astype("int8")
    out["late_close_high_li"] = (
        (late == 1) & (close == 1) & (features["li"] >= 2.0)
    ).astype("int8")
    out["pressure_score"] = li_log * (1.0 + late) * (1.0 + close)
    out["risp_two_out"] = ((features["risp"] == 1) & (outs == 2)).astype("int8")
    out["bases_loaded_less_two_out"] = (
        (features["bases_loaded"] == 1) & (outs < 2)
    ).astype("int8")
    out["runner_out_pressure"] = (
        pd.to_numeric(features["num_runners_on"], errors="coerce")
        * (3.0 - outs).clip(lower=0)
        / 3.0
    )
    out["count_pressure"] = features["full_count"].fillna(0) * li_log
    out["base_out_state"] = (
        features["base_state"].astype("string")
        + "|o"
        + outs.astype("Int64").astype("string")
    )
    out["count_base_out_state"] = (
        features["count_state"].astype("string")
        + "|"
        + out["base_out_state"].astype("string")
    )
    categorical = ["base_out_state", "count_base_out_state"]
    for column in categorical:
        out[column] = out[column].fillna("__MISSING__").astype("category")
    return _as_float(out), categorical


def build_pitchmix_block(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rate_columns = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    _require(features, [*rate_columns, "asof_pitcher_pitchmix_n"], "pitchmix")
    rates = features[rate_columns].apply(pd.to_numeric, errors="coerce").clip(0, 1)
    total = rates.sum(axis=1, min_count=1)
    normalized = rates.div(total.replace(0, np.nan), axis=0)
    entropy = -(normalized * np.log(normalized.clip(lower=1e-8))).sum(
        axis=1, min_count=1
    )

    out = pd.DataFrame(index=features.index)
    out["pitchmix_entropy"] = entropy
    out["pitchmix_dominant_rate"] = rates.max(axis=1, skipna=True)
    out["pitchmix_balance"] = rates.min(axis=1, skipna=True)
    out["pitchmix_fastball_breaking_gap"] = rates[rate_columns[0]] - rates[rate_columns[1]]
    out["pitchmix_fastball_offspeed_gap"] = rates[rate_columns[0]] - rates[rate_columns[2]]
    out["pitchmix_breaking_offspeed_gap"] = rates[rate_columns[1]] - rates[rate_columns[2]]
    out["pitchmix_observed_sum"] = total
    out["pitchmix_available"] = (
        pd.to_numeric(features["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0) > 0
    ).astype("int8")
    return _as_float(out), []


def _reliability(n: pd.Series, strength: float) -> pd.Series:
    count = pd.to_numeric(n, errors="coerce").clip(lower=0)
    return count / (count + strength)


def build_reliability_block(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_success_rate",
        "asof_batter_success_rate",
        "pitcher_career_n",
        "pitcher_history_missing",
    ]
    _require(features, required, "reliability")

    out = pd.DataFrame(index=features.index)
    out["pitcher_asof_reliability"] = _reliability(features["asof_pitcher_n"], 50.0)
    out["batter_asof_reliability"] = _reliability(features["asof_batter_n"], 50.0)
    out["pitchmix_reliability"] = _reliability(
        features["asof_pitcher_pitchmix_n"], 50.0
    )
    out["pitcher_history_reliability"] = _reliability(features["pitcher_career_n"], 100.0)

    if "tm_career_pitch_n" in features:
        out["trackman_reliability"] = _reliability(features["tm_career_pitch_n"], 100.0)
    else:
        out["trackman_reliability"] = np.float32(0.0)

    pitcher_rate = pd.to_numeric(features["asof_pitcher_success_rate"], errors="coerce")
    batter_rate = pd.to_numeric(features["asof_batter_success_rate"], errors="coerce")
    out["asof_pitcher_batter_weighted_gap"] = (
        out["pitcher_asof_reliability"] * pitcher_rate
        - out["batter_asof_reliability"] * batter_rate
    )
    out["information_source_count"] = (
        (out["pitcher_asof_reliability"] > 0).astype("int8")
        + (out["batter_asof_reliability"] > 0).astype("int8")
        + (out["pitchmix_reliability"] > 0).astype("int8")
        + (features["pitcher_history_missing"].fillna(1) == 0).astype("int8")
        + (out["trackman_reliability"] > 0).astype("int8")
    )
    return _as_float(out), []


def build_batter_threat_interaction_block(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build six pre-pitch pitcher-batter interaction features.

    ``batter_control_threat_proxy = 1 - asof_batter_success_rate`` is an
    intermediate proxy, not a direct measure of slugging or home-run threat.
    Missing source values stay missing so models can handle them consistently.
    """
    required = [
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_batter_success_rate",
    ]
    _require(features, required, "batter_threat_interactions")

    pitcher_hand = features["pitcher_hand"].astype("string")
    batter_hand = features["batter_hand"].astype("string")
    hand_available = (
        pitcher_hand.notna()
        & batter_hand.notna()
        & pitcher_hand.ne("__MISSING__")
        & batter_hand.ne("__MISSING__")
    )

    pitcher_success = _probability(features, "asof_pitcher_success_rate")
    batter_success = _probability(features, "asof_batter_success_rate")
    reverse_rate = _probability(features, "asof_pitcher_reverse_rate")
    middle_rate = _probability(features, "asof_pitcher_middle_rate")
    ball_rate = _probability(features, "asof_pitcher_ball_rate")
    batter_control_threat_proxy = 1.0 - batter_success

    out = pd.DataFrame(index=features.index)
    out["hand_match"] = (pitcher_hand == batter_hand).astype("float32").where(
        hand_available
    )
    out["success_interact"] = pitcher_success * batter_success
    out["reverse_x_middle"] = reverse_rate * middle_rate
    out["threat_x_reverse"] = batter_control_threat_proxy * reverse_rate
    out["threat_x_middle"] = batter_control_threat_proxy * middle_rate
    out["threat_x_ball"] = batter_control_threat_proxy * ball_rate
    return _as_float(out[BATTER_THREAT_INTERACTION_FEATURES]), []


FEATURE_BLOCKS: dict[str, FeatureBuilder] = {
    "asof_trend": build_asof_block,
    "situation": build_situation_block,
    "pitchmix": build_pitchmix_block,
    "reliability": build_reliability_block,
    "batter_threat_interactions": build_batter_threat_interaction_block,
}


def apply_feature_set(
    base_features: pd.DataFrame,
    base_categorical: list[str],
    feature_set: dict,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Add row-wise blocks, apply ablations, and optionally keep a strict profile."""
    features = base_features.copy(deep=False)
    categorical = list(base_categorical)
    added: list[str] = []

    for block_name in feature_set.get("blocks", []):
        if block_name not in FEATURE_BLOCKS:
            raise ValueError(
                f"Unknown feature block '{block_name}'. Available: {sorted(FEATURE_BLOCKS)}"
            )
        block, block_categorical = FEATURE_BLOCKS[block_name](features)
        duplicate = sorted(set(block.columns).intersection(features.columns))
        if duplicate:
            raise ValueError(f"Feature block '{block_name}' duplicates columns: {duplicate}")
        features = pd.concat([features, block], axis=1)
        categorical.extend(block_categorical)
        added.extend(block.columns.tolist())

    drop_patterns = feature_set.get("drop_patterns", [])
    dropped = [
        column
        for column in features.columns
        if any(fnmatch(column, pattern) for pattern in drop_patterns)
    ]
    if dropped:
        features = features.drop(columns=dropped)

    keep_profile = feature_set.get("keep_profile")
    requested_keep = resolve_feature_profile(keep_profile) if keep_profile else []
    missing_requested = [
        column for column in requested_keep if column not in features.columns
    ]
    if missing_requested:
        raise ValueError(
            f"Feature profile '{keep_profile}' requires missing columns: "
            f"{missing_requested}"
        )
    selection_dropped = []
    if requested_keep:
        requested_set = set(requested_keep)
        selection_dropped = [
            column for column in features.columns if column not in requested_set
        ]
        features = features.loc[:, [
            column for column in features.columns if column in requested_set
        ]]
    categorical = [column for column in categorical if column in features.columns]

    report = {
        "base_feature_count": int(base_features.shape[1]),
        "added_feature_count": len(added),
        "dropped_feature_count": len(dropped),
        "selection_dropped_feature_count": len(selection_dropped),
        "final_feature_count": int(features.shape[1]),
        "blocks": list(feature_set.get("blocks", [])),
        "keep_profile": keep_profile,
        "keep_profile_description": FEATURE_PROFILE_DESCRIPTIONS.get(keep_profile),
        "added_features": added,
        "dropped_features": dropped,
        "selection_dropped_features": selection_dropped,
        "selected_features": features.columns.tolist(),
        "categorical_features": categorical,
    }
    return features, categorical, report


def feature_catalog() -> pd.DataFrame:
    rows = []
    for block, columns in [
        ("asof_trend", ASOF_TREND_FEATURES),
        ("situation", SITUATION_FEATURES),
        ("pitchmix", PITCHMIX_FEATURES),
        ("reliability", RELIABILITY_FEATURES),
        ("batter_threat_interactions", BATTER_THREAT_INTERACTION_FEATURES),
    ]:
        rows.extend({"block": block, "feature": column} for column in columns)
    return pd.DataFrame(rows)
