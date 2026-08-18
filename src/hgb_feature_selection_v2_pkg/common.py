from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

LOGGER = logging.getLogger(__name__)

def _stable_hash(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_to_pickle(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_pickle(temporary)
    os.replace(temporary, path)


def _checkpoint_rows(path: Path, run_signature: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if "run_signature" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["run_signature"].astype("string") == run_signature].copy()


def _append_checkpoint(path: Path, existing: pd.DataFrame, row: dict) -> pd.DataFrame:
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    _atomic_to_csv(updated, path)
    return updated


TARGET = "control_success"
ID_COLUMN = "row_id"
PITCH_GROUPS = ("fastball", "breaking", "offspeed")
TRACKMAN_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
DRIFT_METRICS = ("rel_speed", "rel_height", "rel_side", "extension")

# Mirrors the official-column handling that already works in first_model_features.py.
REQUIRED_MAIN_COLUMNS = [
    ID_COLUMN,
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
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
]

BASE_NUMERIC = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "li",
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
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

BASE_CATEGORICAL = [
    "top_bottom",
    "game_type",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "base_state",
]

TRACKMAN_REQUIRED_COLUMNS = [
    "season",
    "pitcher_trackman_id",
    "trackman_game_id",
    "pitch_no",
    "balls_before",
    "strikes_before",
    "batter_hand",
    "pitch_type_group",
    *TRACKMAN_METRICS,
]


@dataclass(frozen=True)
class EngineeredData:
    features: pd.DataFrame
    target: pd.Series
    row_ids: pd.Series
    seasons: pd.Series
    blocks: dict[str, list[str]]
    categorical_features: list[str]


@dataclass(frozen=True)
class EncodedData:
    x_train: np.ndarray
    x_valid: np.ndarray
    feature_names: list[str]
    encoder_state: dict


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def _to_probability(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.between(0.0, 1.0))


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_row_std(frame: pd.DataFrame) -> pd.Series:
    return frame.astype("float64").std(axis=1, ddof=0, skipna=True).where(frame.notna().sum(axis=1) >= 2)


def _read_csv(path: Path, *, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=usecols,
        nrows=nrows,
    )


def resolve_input_path(requested: str | Path, expected_name: str) -> Path:
    """Resolve an explicit path, with a conservative Colab MyDrive fallback.

    The fallback is intentionally used only when exactly one file with the expected
    name exists under a mounted MyDrive. It never silently picks between duplicates.
    """
    requested_path = Path(requested).expanduser()
    if requested_path.is_file():
        return requested_path

    roots = [Path("/content/drive/MyDrive"), Path("/content/gdrive_hgb/MyDrive")]
    roots = [root for root in roots if root.is_dir()]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(path for path in root.rglob(expected_name) if path.is_file())
    unique = sorted(set(candidates))
    if len(unique) == 1:
        LOGGER.warning("Requested %s not found; auto-resolved to %s", requested_path, unique[0])
        return unique[0]
    if unique:
        joined = "\n  - ".join(str(path) for path in unique[:20])
        raise FileNotFoundError(
            f"Requested path does not exist: {requested_path}\n"
            f"Multiple {expected_name} files were found; pass the intended one explicitly:\n  - {joined}"
        )
    mount_hint = (
        "No mounted MyDrive was detected. In Colab, mount Drive first. "
        "The provided notebook mounts to /content/gdrive_hgb if /content/drive is unavailable."
    )
    raise FileNotFoundError(f"Input file not found: {requested_path}. {mount_hint}")


def sample_by_season(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame.reset_index(drop=True)
    parts = []
    for _, group in frame.groupby("season", observed=True, sort=True):
        parts.append(group.sample(min(len(group), limit), random_state=seed))
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["season", ID_COLUMN], kind="stable")
        .reset_index(drop=True)
    )


def season_prior(train: pd.DataFrame, seasons: pd.Series, fallback: float = 0.5) -> pd.Series:
    """Target prior using only seasons strictly before each prediction season."""
    target = pd.to_numeric(train[TARGET], errors="raise").astype("float64")
    source_season = pd.to_numeric(train["season"], errors="raise").astype(int)
    priors: dict[int, float] = {}
    for season in sorted(pd.to_numeric(seasons, errors="coerce").dropna().astype(int).unique()):
        mask = source_season < season
        priors[season] = float(target.loc[mask].mean()) if mask.any() else fallback
    return pd.to_numeric(seasons, errors="coerce").map(priors).astype("float64")


def _shrink_rate(rate: pd.Series, n: pd.Series, prior: pd.Series, strength: float) -> pd.Series:
    rate_v = _to_probability(rate)
    n_v = _safe_numeric(n).clip(lower=0)
    prior_v = _to_probability(prior)
    return ((rate_v * n_v) + (prior_v * strength)) / (n_v + strength)


def _register(blocks: dict[str, list[str]], block: str, frame: pd.DataFrame, columns: list[str]) -> None:
    blocks.setdefault(block, []).extend(column for column in columns if column in frame.columns)


def build_main_features(train: pd.DataFrame, *, shrinkage: float = 50.0) -> EngineeredData:
    _require_columns(train, [*REQUIRED_MAIN_COLUMNS, TARGET], "train.csv")
    raw = train.copy()
    raw[TARGET] = pd.to_numeric(raw[TARGET], errors="raise").astype("int8")
    raw["season"] = pd.to_numeric(raw["season"], errors="raise").astype(int)
    raw["pitcher_id"] = _normalize_id(raw["pitcher_id"])
    raw["batter_id"] = _normalize_id(raw["batter_id"])

    features = pd.DataFrame(index=raw.index)
    blocks: dict[str, list[str]] = {}
    categorical: list[str] = []

    # Official pre-pitch baseline. High-cardinality pitcher/batter IDs are kept only
    # as join keys, not as HGB features; this matches the compact prior experiments.
    for column in BASE_NUMERIC:
        features[column] = _safe_numeric(raw[column]).astype("float32")
    for column in BASE_CATEGORICAL:
        features[column] = raw[column].astype("string").fillna("__MISSING__")
        categorical.append(column)
    _register(blocks, "official_baseline", features, [*BASE_NUMERIC, *BASE_CATEGORICAL])

    balls = _safe_numeric(raw["balls_before"])
    strikes = _safe_numeric(raw["strikes_before"])
    outs = _safe_numeric(raw["outs_before"])
    li = _safe_numeric(raw["li"]).clip(lower=0)
    score_diff = _safe_numeric(raw["score_diff_pitcher_team"])

    # Control baseline + reliability/shrinkage.
    prior = season_prior(raw, raw["season"])
    features["asof_pitcher_n_log1p"] = np.log1p(_safe_numeric(raw["asof_pitcher_n"]).clip(lower=0))
    features["asof_batter_n_log1p"] = np.log1p(_safe_numeric(raw["asof_batter_n"]).clip(lower=0))
    features["pitcher_success_shrunk"] = _shrink_rate(
        raw["asof_pitcher_success_rate"], raw["asof_pitcher_n"], prior, shrinkage
    )
    features["batter_success_shrunk"] = _shrink_rate(
        raw["asof_batter_success_rate"], raw["asof_batter_n"], prior, shrinkage
    )
    features["pitcher_asof_reliability"] = (
        _safe_numeric(raw["asof_pitcher_n"]).clip(lower=0)
        / (_safe_numeric(raw["asof_pitcher_n"]).clip(lower=0) + shrinkage)
    )
    features["batter_asof_reliability"] = (
        _safe_numeric(raw["asof_batter_n"]).clip(lower=0)
        / (_safe_numeric(raw["asof_batter_n"]).clip(lower=0) + shrinkage)
    )
    _register(
        blocks,
        "control_prior",
        features,
        [
            "asof_pitcher_n_log1p",
            "asof_batter_n_log1p",
            "pitcher_success_shrunk",
            "batter_success_shrunk",
            "pitcher_asof_reliability",
            "batter_asof_reliability",
        ],
    )

    # Recent form: all row-wise transformations of official asof columns.
    p_success = _to_probability(raw["asof_pitcher_success_rate"])
    p_middle = _to_probability(raw["asof_pitcher_middle_rate"])
    recent_success = {
        w: _to_probability(raw[f"asof_pitcher_prev{w}_game_success_rate"]) for w in (1, 3, 5)
    }
    recent_middle = {
        w: _to_probability(raw[f"asof_pitcher_prev{w}_game_middle_rate"]) for w in (1, 3, 5)
    }
    recent_cols: list[str] = []
    for w in (1, 3, 5):
        name = f"success_form_{w}"
        features[name] = recent_success[w] - p_success
        recent_cols.append(name)
        name = f"middle_form_{w}"
        features[name] = recent_middle[w] - p_middle
        recent_cols.append(name)
    for a, b in ((1, 3), (1, 5), (3, 5)):
        s_name = f"success_trend_{a}v{b}"
        m_name = f"middle_trend_{a}v{b}"
        features[s_name] = recent_success[a] - recent_success[b]
        features[m_name] = recent_middle[a] - recent_middle[b]
        recent_cols.extend([s_name, m_name])
    features["recent_success_volatility"] = _safe_row_std(
        pd.DataFrame(recent_success, index=raw.index)
    )
    features["recent_middle_volatility"] = _safe_row_std(
        pd.DataFrame(recent_middle, index=raw.index)
    )
    recent_cols.extend(["recent_success_volatility", "recent_middle_volatility"])
    _register(blocks, "recent_form", features, recent_cols)

    # Count / intent.
    features["count_state"] = (
        balls.astype("Int64").astype("string") + "-" + strikes.astype("Int64").astype("string")
    ).fillna("__MISSING__")
    categorical.append("count_state")
    features["ball_strike_diff"] = balls - strikes
    features["first_pitch"] = ((balls == 0) & (strikes == 0)).astype("int8")
    features["two_strike"] = (strikes == 2).astype("int8")
    features["three_ball"] = (balls == 3).astype("int8")
    features["full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    features["pitcher_ahead"] = (strikes > balls).astype("int8")
    features["pitcher_behind"] = (balls > strikes).astype("int8")
    features["three_ball_x_middle"] = features["three_ball"] * p_middle
    features["two_strike_x_reverse"] = features["two_strike"] * _to_probability(
        raw["asof_pitcher_reverse_rate"]
    )
    features["count_diff_x_success"] = features["ball_strike_diff"] * p_success
    _register(
        blocks,
        "count_intent",
        features,
        [
            "count_state",
            "ball_strike_diff",
            "first_pitch",
            "two_strike",
            "three_ball",
            "full_count",
            "pitcher_ahead",
            "pitcher_behind",
            "three_ball_x_middle",
            "two_strike_x_reverse",
            "count_diff_x_success",
        ],
    )

    # Matchup / handedness / batter context.
    pitcher_hand = raw["pitcher_hand"].astype("string").fillna("__MISSING__")
    batter_hand = raw["batter_hand"].astype("string").fillna("__MISSING__")
    features["hand_pair"] = (pitcher_hand + "-" + batter_hand).astype("string")
    categorical.append("hand_pair")
    hand_available = pitcher_hand.ne("__MISSING__") & batter_hand.ne("__MISSING__")
    features["hand_match"] = (pitcher_hand == batter_hand).astype("float32").where(hand_available)
    batter_success = _to_probability(raw["asof_batter_success_rate"])
    batter_middle = _to_probability(raw["asof_batter_middle_rate"])
    reverse_rate = _to_probability(raw["asof_pitcher_reverse_rate"])
    ball_rate = _to_probability(raw["asof_pitcher_ball_rate"])
    threat = 1.0 - batter_success
    features["pitcher_batter_success_gap"] = features["pitcher_success_shrunk"] - features[
        "batter_success_shrunk"
    ]
    features["pitcher_batter_middle_gap"] = p_middle - batter_middle
    features["success_interact"] = p_success * batter_success
    features["reverse_x_middle"] = reverse_rate * p_middle
    features["threat_x_reverse"] = threat * reverse_rate
    features["threat_x_middle"] = threat * p_middle
    features["threat_x_ball"] = threat * ball_rate
    features["hand_x_middle"] = features["hand_match"] * p_middle
    for group in PITCH_GROUPS:
        source = f"asof_pitcher_{group}_rate"
        name = f"hand_x_{group}_rate"
        features[name] = features["hand_match"] * _to_probability(raw[source])
    _register(
        blocks,
        "matchup",
        features,
        [
            "hand_pair",
            "hand_match",
            "pitcher_batter_success_gap",
            "pitcher_batter_middle_gap",
            "success_interact",
            "reverse_x_middle",
            "threat_x_reverse",
            "threat_x_middle",
            "threat_x_ball",
            "hand_x_middle",
            "hand_x_fastball_rate",
            "hand_x_breaking_rate",
            "hand_x_offspeed_rate",
        ],
    )

    # Game context.
    features["base_out_state"] = (
        raw["base_state"].astype("string").fillna("__MISSING__")
        + "|o"
        + outs.astype("Int64").astype("string")
    )
    categorical.append("base_out_state")
    features["risp"] = (
        (_safe_numeric(raw["runner_on_2b"]) == 1) | (_safe_numeric(raw["runner_on_3b"]) == 1)
    ).astype("int8")
    features["first_base_open"] = (_safe_numeric(raw["runner_on_1b"]) != 1).astype("int8")
    features["abs_score_diff"] = score_diff.abs()
    features["close_game"] = (score_diff.abs() <= 2).astype("int8")
    features["late_inning"] = (_safe_numeric(raw["inning"]) >= 7).astype("int8")
    features["li_log1p"] = np.log1p(li)
    features["high_leverage"] = (li >= 1.5).astype("int8")
    features["full_count_x_li"] = features["full_count"] * features["li_log1p"]
    features["late_close_high_li"] = (
        (features["late_inning"] == 1) & (features["close_game"] == 1) & (li >= 1.5)
    ).astype("int8")
    features["runner_out_pressure"] = (
        _safe_numeric(raw["num_runners_on"]) * (3.0 - outs).clip(lower=0) / 3.0
    )
    top_bottom = raw["top_bottom"].astype("string")
    home_we = _safe_numeric(raw["home_win_expectancy"])
    away_we = _safe_numeric(raw["away_win_expectancy"])
    features["pitcher_team_we"] = np.where(top_bottom.eq("T"), home_we, away_we)
    features["we_pressure"] = (1.0 - (features["pitcher_team_we"] - 50.0).abs() / 50.0).clip(0, 1)
    _register(
        blocks,
        "game_context",
        features,
        [
            "base_out_state",
            "risp",
            "first_base_open",
            "abs_score_diff",
            "close_game",
            "late_inning",
            "li_log1p",
            "high_leverage",
            "full_count_x_li",
            "late_close_high_li",
            "runner_out_pressure",
            "pitcher_team_we",
            "we_pressure",
        ],
    )

    # Pitch mix composition.
    mix = pd.DataFrame(
        {
            group: _to_probability(raw[f"asof_pitcher_{group}_rate"])
            for group in PITCH_GROUPS
        },
        index=raw.index,
    )
    mix_total = mix.sum(axis=1, min_count=1)
    mix_norm = mix.div(mix_total.replace(0, np.nan), axis=0)
    features["pitchmix_entropy"] = -(mix_norm * np.log(mix_norm.clip(lower=1e-8))).sum(
        axis=1, min_count=1
    )
    features["pitchmix_dominant_rate"] = mix.max(axis=1, skipna=True)
    features["pitchmix_fastball_breaking_gap"] = mix["fastball"] - mix["breaking"]
    features["pitchmix_fastball_offspeed_gap"] = mix["fastball"] - mix["offspeed"]
    features["pitchmix_breaking_offspeed_gap"] = mix["breaking"] - mix["offspeed"]
    _register(
        blocks,
        "pitchmix",
        features,
        [
            "pitchmix_entropy",
            "pitchmix_dominant_rate",
            "pitchmix_fastball_breaking_gap",
            "pitchmix_fastball_offspeed_gap",
            "pitchmix_breaking_offspeed_gap",
        ],
    )

    # Preserve join keys separately in DataFrame attrs for Trackman enrichment.
    features.attrs["pitcher_id"] = raw["pitcher_id"].copy()
    features.attrs["balls_before"] = balls.copy()
    features.attrs["strikes_before"] = strikes.copy()
    features.attrs["batter_hand"] = batter_hand.copy()

    for column in features.columns:
        if column not in categorical:
            features[column] = _safe_numeric(features[column]).astype("float32")
        else:
            features[column] = features[column].astype("string").fillna("__MISSING__")

    return EngineeredData(
        features=features,
        target=raw[TARGET].copy(),
        row_ids=raw[ID_COLUMN].copy(),
        seasons=raw["season"].copy(),
        blocks=blocks,
        categorical_features=categorical,
    )

