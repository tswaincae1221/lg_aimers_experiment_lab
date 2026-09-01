from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


PITCH_GROUPS = ("fastball", "breaking", "offspeed")
TM_METRICS = (
    "rel_speed",
    "zone_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
)
DRIFT_METRICS = ("rel_speed", "extension", "rel_height", "rel_side")

OFFICIAL_ASOF = [
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

BASE_NUMERIC = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_total_before",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
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

PROTECTED_FEATURES = {
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_n",
    "pitcher_success_shrunk",
    "count_state",
}


@dataclass
class RunConfig:
    train_path: str
    trackman_path: str
    mapping_path: str
    output_dir: str
    validation_season: int
    random_state: int
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    permutation_sample: int
    permutation_repeats: int
    lofo_max_features: int
    lofo_improvement_threshold: float
    quick_rows: int | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LG Aimers literature-grounded HGB feature generation and selection pipeline"
    )
    p.add_argument("--train", default="/content/drive/MyDrive/aimers_data/train.csv")
    p.add_argument(
        "--trackman", default="/content/drive/MyDrive/aimers_data/trackman_history.csv"
    )
    p.add_argument(
        "--mapping", default="resources/pitcher_trackman_mapping.csv"
    )
    p.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/aimers_data/results/hgb_literature_feature_selection",
    )
    p.add_argument("--validation-season", type=int, default=2024)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=350)
    p.add_argument("--learning-rate", type=float, default=0.06)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=200)
    p.add_argument("--l2-regularization", type=float, default=2.0)
    p.add_argument("--permutation-sample", type=int, default=60000)
    p.add_argument("--permutation-repeats", type=int, default=3)
    p.add_argument("--lofo-max-features", type=int, default=20)
    p.add_argument(
        "--lofo-improvement-threshold",
        type=float,
        default=1e-5,
        help="Drop a feature only when removing it improves Brier by at least this amount.",
    )
    p.add_argument(
        "--mode", choices=["quick", "full"], default="full",
        help="quick uses a bounded train sample and smaller selection workload for pipeline checks.",
    )
    return p.parse_args()


def require_columns(frame: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")


def safe_numeric(s) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(np.nan, index=getattr(s, "index", None), dtype="float64") if s is None else pd.to_numeric(s, errors="coerce")


def col_or_nan(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return safe_numeric(frame[name])
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def rms_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    arr = frame[columns].apply(pd.to_numeric, errors="coerce")
    return np.sqrt((arr.pow(2)).mean(axis=1, skipna=True))


def brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    brier = brier_score_loss(y, p)
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    if denom <= 0:
        return float("nan")
    return 100000.0 * (1.0 - brier / denom)


def metrics_dict(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {
        "brier": float(brier_score_loss(y, p)),
        "brier_skill_score": float(brier_skill(y, p)),
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p)),
        "target_mean": float(np.mean(y)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    return out


def normalize_pitch_group(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.lower()
    mapping = {
        "fastball": "fastball",
        "fb": "fastball",
        "breaking": "breaking",
        "breakingball": "breaking",
        "offspeed": "offspeed",
        "off_speed": "offspeed",
    }
    return x.map(mapping).where(x.map(mapping).notna(), x)


def load_mapping(path: str) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    require_columns(mapping, ["pitcher_id", "pitcher_trackman_id", "신뢰등급"], "mapping")
    mapping = mapping[mapping["신뢰등급"].isin(["확정", "높음"])].copy()
    mapping["pitcher_id"] = pd.to_numeric(mapping["pitcher_id"], errors="coerce").astype("Int64")
    mapping["pitcher_trackman_id"] = pd.to_numeric(
        mapping["pitcher_trackman_id"], errors="coerce"
    ).astype("Int64")
    mapping = mapping.dropna(subset=["pitcher_id", "pitcher_trackman_id"])
    mapping = mapping.drop_duplicates("pitcher_trackman_id", keep="first")
    return mapping[["pitcher_id", "pitcher_trackman_id", "신뢰등급"]]


def load_trackman(trackman_path: str, mapping: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trackman_game_id",
        "pitcher_trackman_id",
        "season",
        "pitch_no",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "pitch_type_group",
        *TM_METRICS,
    ]
    tm = pd.read_csv(trackman_path, usecols=cols, low_memory=False)
    tm["pitcher_trackman_id"] = pd.to_numeric(
        tm["pitcher_trackman_id"], errors="coerce"
    ).astype("Int64")
    tm = tm.merge(mapping, how="inner", on="pitcher_trackman_id", validate="many_to_one")
    tm["pitch_type_group"] = normalize_pitch_group(tm["pitch_type_group"])
    tm = tm[tm["pitch_type_group"].isin(PITCH_GROUPS)].copy()
    tm["season"] = pd.to_numeric(tm["season"], errors="coerce").astype("Int64")
    for col in ["pitch_no", "balls_before", "strikes_before", *TM_METRICS]:
        tm[col] = pd.to_numeric(tm[col], errors="coerce")
    tm["batter_hand"] = tm["batter_hand"].astype("string")
    return tm


def yearly_group_sufficient_stats(tm: pd.DataFrame) -> pd.DataFrame:
    keys = ["pitcher_id", "season", "pitch_type_group"]
    work = tm[keys].copy()
    for m in TM_METRICS:
        v = safe_numeric(tm[m])
        work[f"{m}__n"] = v.notna().astype("int32")
        work[f"{m}__sum"] = v.fillna(0).astype("float64")
        work[f"{m}__sumsq"] = v.fillna(0).pow(2).astype("float64")
    out = work.groupby(keys, observed=True, sort=False).sum(numeric_only=True).reset_index()
    return out


def yearly_context_counts(tm: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "pitcher_id",
        "season",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "pitch_type_group",
    ]
    x = tm.dropna(subset=keys).groupby(keys, observed=True, sort=False).size()
    return x.rename("n").reset_index()


def yearly_drift_stats(tm: pd.DataFrame) -> pd.DataFrame:
    keys_game = [
        "pitcher_id",
        "season",
        "trackman_game_id",
        "pitch_type_group",
    ]
    x = safe_numeric(tm["pitch_no"])
    x_mean = x.groupby([tm[k] for k in keys_game], observed=True).transform("mean")
    xc = x - x_mean

    out = tm[["pitcher_id", "season", "pitch_type_group"]].copy()
    out["x2"] = xc.pow(2).fillna(0)
    for m in DRIFT_METRICS:
        y = safe_numeric(tm[m])
        y_mean = y.groupby([tm[k] for k in keys_game], observed=True).transform("mean")
        yc = y - y_mean
        valid = xc.notna() & yc.notna()
        out[f"{m}__xy"] = (xc * yc).where(valid, 0.0)
        out[f"{m}__x2"] = xc.pow(2).where(valid, 0.0)
    keys = ["pitcher_id", "season", "pitch_type_group"]
    return out.groupby(keys, observed=True, sort=False).sum(numeric_only=True).reset_index()


def cumulative_group_profile(yearly: pd.DataFrame, target_season: int) -> pd.DataFrame:
    hist = yearly[yearly["season"] < target_season]
    if hist.empty:
        return pd.DataFrame(columns=["pitcher_id"])
    agg_cols = [c for c in hist.columns if c not in {"pitcher_id", "season", "pitch_type_group"}]
    agg = (
        hist.groupby(["pitcher_id", "pitch_type_group"], observed=True, sort=False)[agg_cols]
        .sum()
        .reset_index()
    )
    for m in TM_METRICS:
        n = agg[f"{m}__n"].astype(float)
        s = agg[f"{m}__sum"].astype(float)
        ss = agg[f"{m}__sumsq"].astype(float)
        var = (ss - s.pow(2) / n.replace(0, np.nan)) / (n - 1).replace(0, np.nan)
        agg[f"{m}_sd"] = np.sqrt(var.clip(lower=0))
        agg[f"{m}_mean"] = s / n.replace(0, np.nan)

    agg["tm_release_dispersion"] = rms_columns(
        agg, ["rel_height_sd", "rel_side_sd", "extension_sd"]
    )
    agg["tm_velocity_dispersion"] = rms_columns(
        agg, ["rel_speed_sd", "zone_speed_sd"]
    )
    agg["tm_movement_dispersion"] = rms_columns(
        agg, ["induced_vert_break_sd", "horz_break_sd"]
    )
    agg["tm_spin_dispersion"] = agg["spin_rate_sd"]
    agg["tm_group_pitch_n"] = agg["rel_speed__n"]

    keep = [
        "pitcher_id",
        "pitch_type_group",
        "tm_release_dispersion",
        "tm_velocity_dispersion",
        "tm_movement_dispersion",
        "tm_spin_dispersion",
        "tm_group_pitch_n",
    ]
    return agg[keep]


def cumulative_drift_profile(yearly: pd.DataFrame, target_season: int) -> pd.DataFrame:
    hist = yearly[yearly["season"] < target_season]
    if hist.empty:
        return pd.DataFrame(columns=["pitcher_id"])
    agg_cols = [c for c in hist.columns if c not in {"pitcher_id", "season", "pitch_type_group"}]
    agg = (
        hist.groupby(["pitcher_id", "pitch_type_group"], observed=True, sort=False)[agg_cols]
        .sum()
        .reset_index()
    )
    for m in DRIFT_METRICS:
        denom = agg[f"{m}__x2"].replace(0, np.nan)
        agg[f"tm_{m}_drift_slope"] = agg[f"{m}__xy"] / denom
    keep = ["pitcher_id", "pitch_type_group"] + [
        f"tm_{m}_drift_slope" for m in DRIFT_METRICS
    ]
    return agg[keep]


def cumulative_context_profile(yearly: pd.DataFrame, target_season: int) -> pd.DataFrame:
    hist = yearly[yearly["season"] < target_season]
    if hist.empty:
        return pd.DataFrame(
            columns=["pitcher_id", "balls_before", "strikes_before", "batter_hand"]
        )
    keys = ["pitcher_id", "balls_before", "strikes_before", "batter_hand", "pitch_type_group"]
    agg = hist.groupby(keys, observed=True, sort=False)["n"].sum().reset_index()
    wide = agg.pivot_table(
        index=["pitcher_id", "balls_before", "strikes_before", "batter_hand"],
        columns="pitch_type_group",
        values="n",
        fill_value=0,
        observed=True,
    ).reset_index()
    for g in PITCH_GROUPS:
        if g not in wide:
            wide[g] = 0.0
    total = wide[list(PITCH_GROUPS)].sum(axis=1)
    for g in PITCH_GROUPS:
        wide[f"tm_ctx_p_{g}"] = wide[g] / total.replace(0, np.nan)
    return wide[
        ["pitcher_id", "balls_before", "strikes_before", "batter_hand"]
        + [f"tm_ctx_p_{g}" for g in PITCH_GROUPS]
    ]


def wide_group_profile(profile: pd.DataFrame, prefix_fields: list[str]) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(columns=["pitcher_id"])
    pieces = []
    for g in PITCH_GROUPS:
        sub = profile[profile["pitch_type_group"] == g].copy()
        cols = ["pitcher_id"] + prefix_fields
        sub = sub[cols]
        sub = sub.rename(columns={c: f"{c}__{g}" for c in prefix_fields})
        pieces.append(sub)
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="pitcher_id", how="outer")
    return out


def attach_trackman_features(train: pd.DataFrame, tm: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    yearly = yearly_group_sufficient_stats(tm)
    context_yearly = yearly_context_counts(tm)
    drift_yearly = yearly_drift_stats(tm)

    result_parts = []
    created = set()
    for season in sorted(pd.to_numeric(train["season"], errors="coerce").dropna().astype(int).unique()):
        part = train.loc[train["season"] == season, [
            "pitcher_id", "balls_before", "strikes_before", "batter_hand",
            "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"
        ]].copy()
        part["__index"] = part.index

        gp = cumulative_group_profile(yearly, int(season))
        dp = cumulative_drift_profile(drift_yearly, int(season))
        cp = cumulative_context_profile(context_yearly, int(season))

        gp_wide = wide_group_profile(
            gp,
            [
                "tm_release_dispersion",
                "tm_velocity_dispersion",
                "tm_movement_dispersion",
                "tm_spin_dispersion",
                "tm_group_pitch_n",
            ],
        )
        dp_wide = wide_group_profile(
            dp,
            [f"tm_{m}_drift_slope" for m in DRIFT_METRICS],
        )
        part = part.merge(gp_wide, on="pitcher_id", how="left")
        part = part.merge(dp_wide, on="pitcher_id", how="left")
        if not cp.empty:
            cp = cp.copy()
            cp["batter_hand"] = cp["batter_hand"].astype("string")
            part["batter_hand"] = part["batter_hand"].astype("string")
            part = part.merge(
                cp,
                on=["pitcher_id", "balls_before", "strikes_before", "batter_hand"],
                how="left",
            )
        else:
            for g in PITCH_GROUPS:
                part[f"tm_ctx_p_{g}"] = np.nan

        weights = {
            "fastball": safe_numeric(part["asof_pitcher_fastball_rate"]),
            "breaking": safe_numeric(part["asof_pitcher_breaking_rate"]),
            "offspeed": safe_numeric(part["asof_pitcher_offspeed_rate"]),
        }
        weight_sum = sum(weights.values())
        for g in PITCH_GROUPS:
            weights[g] = weights[g] / weight_sum.replace(0, np.nan)

        for metric in [
            "tm_release_dispersion",
            "tm_velocity_dispersion",
            "tm_movement_dispersion",
            "tm_spin_dispersion",
        ]:
            val = sum(
                weights[g] * col_or_nan(part, f"{metric}__{g}")
                for g in PITCH_GROUPS
            )
            part[f"tm_expected_{metric.removeprefix('tm_')}"] = val
            created.add(f"tm_expected_{metric.removeprefix('tm_')}")

            ctx_probs = {}
            for g in PITCH_GROUPS:
                ctx = safe_numeric(part[f"tm_ctx_p_{g}"])
                ctx_probs[g] = (20.0 * weights[g] + 30.0 * ctx) / 50.0
            ctx_sum = sum(ctx_probs.values())
            for g in PITCH_GROUPS:
                ctx_probs[g] = ctx_probs[g] / ctx_sum.replace(0, np.nan)
            ctx_val = sum(
                ctx_probs[g] * col_or_nan(part, f"{metric}__{g}")
                for g in PITCH_GROUPS
            )
            name = f"tm_context_expected_{metric.removeprefix('tm_')}"
            part[name] = ctx_val
            created.add(name)

        for m in DRIFT_METRICS:
            drift = sum(
                weights[g] * col_or_nan(part, f"tm_{m}_drift_slope__{g}")
                for g in PITCH_GROUPS
            )
            name = f"tm_expected_{m}_drift_slope"
            part[name] = drift
            created.add(name)

        counts = [col_or_nan(part, f"tm_group_pitch_n__{g}") for g in PITCH_GROUPS]
        part["tm_history_pitch_n"] = sum(c.fillna(0) for c in counts)
        part["tm_history_reliability"] = part["tm_history_pitch_n"] / (
            part["tm_history_pitch_n"] + 100.0
        )
        part["tm_history_missing"] = (part["tm_history_pitch_n"] <= 0).astype("int8")
        created.update(["tm_history_pitch_n", "tm_history_reliability", "tm_history_missing"])

        keep = ["__index"] + sorted(created)
        result_parts.append(part[keep])

    tm_features = pd.concat(result_parts, ignore_index=True).set_index("__index")
    tm_features = tm_features.reindex(train.index)
    return tm_features, sorted(created)


def build_features(train: pd.DataFrame, tm_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    require_columns(train, ["pitcher_id", "batter_id", "control_success", *BASE_NUMERIC, *BASE_CATEGORICAL, *OFFICIAL_ASOF], "train")
    x = pd.DataFrame(index=train.index)
    blocks: dict[str, list[str]] = {}
    categorical: list[str] = []

    def add_block(name: str, frame: pd.DataFrame, cats: list[str] | None = None) -> None:
        nonlocal x
        duplicate = set(frame.columns).intersection(x.columns)
        if duplicate:
            raise ValueError(f"Duplicate features in block {name}: {sorted(duplicate)}")
        x = pd.concat([x, frame], axis=1)
        blocks[name] = list(frame.columns)
        if cats:
            categorical.extend(cats)

    base_num = train[BASE_NUMERIC].apply(pd.to_numeric, errors="coerce").astype("float32")
    add_block("base_context", base_num)

    base_cat = train[BASE_CATEGORICAL].copy()
    for c in base_cat:
        base_cat[c] = base_cat[c].astype("string").fillna("__MISSING__")
    add_block("base_categorical", base_cat, BASE_CATEGORICAL)

    asof = train[OFFICIAL_ASOF].apply(pd.to_numeric, errors="coerce").astype("float32")
    add_block("official_asof", asof)

    f = pd.DataFrame(index=train.index)
    f["log1p_asof_pitcher_n"] = np.log1p(safe_numeric(train["asof_pitcher_n"]).clip(lower=0))
    f["log1p_asof_batter_n"] = np.log1p(safe_numeric(train["asof_batter_n"]).clip(lower=0))
    n = safe_numeric(train["asof_pitcher_n"]).clip(lower=0)
    r = safe_numeric(train["asof_pitcher_success_rate"])
    f["pitcher_success_shrunk"] = (r * n + 0.5 * 50.0) / (n + 50.0)
    f["pitcher_asof_reliability"] = n / (n + 50.0)
    f["batter_asof_reliability"] = safe_numeric(train["asof_batter_n"]).clip(lower=0) / (
        safe_numeric(train["asof_batter_n"]).clip(lower=0) + 50.0
    )
    add_block("reliability", f.astype("float32"))

    r = pd.DataFrame(index=train.index)
    long_s = safe_numeric(train["asof_pitcher_success_rate"])
    p1 = safe_numeric(train["asof_pitcher_prev1_game_success_rate"])
    p3 = safe_numeric(train["asof_pitcher_prev3_game_success_rate"])
    p5 = safe_numeric(train["asof_pitcher_prev5_game_success_rate"])
    long_m = safe_numeric(train["asof_pitcher_middle_rate"])
    m1 = safe_numeric(train["asof_pitcher_prev1_game_middle_rate"])
    m3 = safe_numeric(train["asof_pitcher_prev3_game_middle_rate"])
    m5 = safe_numeric(train["asof_pitcher_prev5_game_middle_rate"])
    r["success_form_1"] = p1 - long_s
    r["success_form_3"] = p3 - long_s
    r["success_form_5"] = p5 - long_s
    r["success_trend_1v3"] = p1 - p3
    r["success_trend_1v5"] = p1 - p5
    r["middle_form_1"] = m1 - long_m
    r["middle_form_3"] = m3 - long_m
    r["middle_form_5"] = m5 - long_m
    r["middle_trend_1v3"] = m1 - m3
    r["middle_trend_1v5"] = m1 - m5
    r["recent_success_volatility"] = pd.concat([p1, p3, p5], axis=1).std(axis=1)
    r["recent_middle_volatility"] = pd.concat([m1, m3, m5], axis=1).std(axis=1)
    add_block("recent_form", r.astype("float32"))

    c = pd.DataFrame(index=train.index)
    balls = safe_numeric(train["balls_before"])
    strikes = safe_numeric(train["strikes_before"])
    c["ball_strike_diff"] = balls - strikes
    c["two_strike"] = (strikes == 2).astype("int8")
    c["three_ball"] = (balls == 3).astype("int8")
    c["full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    c["first_pitch"] = ((balls == 0) & (strikes == 0)).astype("int8")
    c["pitcher_ahead"] = (strikes > balls).astype("int8")
    c["pitcher_behind"] = (balls > strikes).astype("int8")
    c["count_state"] = balls.astype("Int64").astype("string") + "-" + strikes.astype("Int64").astype("string")
    c["count_state"] = c["count_state"].fillna("__MISSING__")
    add_block("count_intent", c, ["count_state"])

    m = pd.DataFrame(index=train.index)
    ph = train["pitcher_hand"].astype("string")
    bh = train["batter_hand"].astype("string")
    m["hand_match"] = (ph == bh).astype("int8")
    m["hand_pair"] = (ph.fillna("?") + "-" + bh.fillna("?")).astype("string")
    m["pitcher_batter_success_gap"] = safe_numeric(train["asof_pitcher_success_rate"]) - safe_numeric(train["asof_batter_success_rate"])
    m["pitcher_batter_middle_gap"] = safe_numeric(train["asof_pitcher_middle_rate"]) - safe_numeric(train["asof_batter_middle_rate"])
    add_block("matchup", m, ["hand_pair"])

    g = pd.DataFrame(index=train.index)
    outs = safe_numeric(train["outs_before"])
    g["risp"] = ((safe_numeric(train["runner_on_2b"]) == 1) | (safe_numeric(train["runner_on_3b"]) == 1)).astype("int8")
    g["abs_score_diff"] = safe_numeric(train["score_diff_pitcher_team"]).abs()
    g["li_log1p"] = np.log1p(safe_numeric(train["li"]).clip(lower=0))
    g["high_leverage"] = (safe_numeric(train["li"]) >= 1.5).astype("int8")
    g["late_inning"] = (safe_numeric(train["inning"]) >= 7).astype("int8")
    g["base_out_state"] = train["base_state"].astype("string").fillna("___") + "|o" + outs.astype("Int64").astype("string")
    add_block("game_context", g, ["base_out_state"])

    p = pd.DataFrame(index=train.index)
    rates = train[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]].apply(pd.to_numeric, errors="coerce").clip(0, 1)
    total = rates.sum(axis=1).replace(0, np.nan)
    norm = rates.div(total, axis=0)
    p["pitchmix_entropy"] = -(norm * np.log(norm.clip(lower=1e-8))).sum(axis=1)
    p["pitchmix_dominant_rate"] = rates.max(axis=1)
    p["pitchmix_fastball_breaking_gap"] = rates.iloc[:, 0] - rates.iloc[:, 1]
    p["pitchmix_fastball_offspeed_gap"] = rates.iloc[:, 0] - rates.iloc[:, 2]
    add_block("pitchmix", p.astype("float32"))

    it = pd.DataFrame(index=train.index)
    it["success_interact"] = safe_numeric(train["asof_pitcher_success_rate"]) * safe_numeric(train["asof_batter_success_rate"])
    it["reverse_x_middle"] = safe_numeric(train["asof_pitcher_reverse_rate"]) * safe_numeric(train["asof_pitcher_middle_rate"])
    threat = 1.0 - safe_numeric(train["asof_batter_success_rate"])
    it["threat_x_reverse"] = threat * safe_numeric(train["asof_pitcher_reverse_rate"])
    it["threat_x_middle"] = threat * safe_numeric(train["asof_pitcher_middle_rate"])
    it["threat_x_ball"] = threat * safe_numeric(train["asof_pitcher_ball_rate"])
    it["three_ball_x_middle"] = c["three_ball"] * safe_numeric(train["asof_pitcher_middle_rate"])
    it["two_strike_x_reverse"] = c["two_strike"] * safe_numeric(train["asof_pitcher_reverse_rate"])
    it["count_diff_x_success"] = c["ball_strike_diff"] * safe_numeric(train["asof_pitcher_success_rate"])
    it["hand_match_x_middle"] = m["hand_match"] * safe_numeric(train["asof_pitcher_middle_rate"])
    it["full_count_x_li"] = c["full_count"] * g["li_log1p"]
    add_block("interactions", it.astype("float32"))

    tmf = tm_features.copy()
    add_block("trackman_compact", tmf.astype("float32"))

    ft = pd.DataFrame(index=train.index)
    for mname in DRIFT_METRICS:
        source = f"tm_expected_{mname}_drift_slope"
        if source in tmf:
            ft[f"inning_x_{mname}_drift"] = safe_numeric(train["inning"]) * safe_numeric(tmf[source])
    add_block("fatigue_interactions", ft.astype("float32"))

    for col in x.columns:
        if col not in categorical:
            x[col] = safe_numeric(x[col]).astype("float32")
    for col in categorical:
        x[col] = x[col].astype("string").fillna("__MISSING__")
    return x, blocks, categorical


@dataclass
class EncodedData:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    val_row_id: np.ndarray
    feature_names: list[str]
    categorical_mask: np.ndarray
    encoder: OrdinalEncoder | None
    categorical_columns: list[str]


def encode_split(features: pd.DataFrame, train: pd.DataFrame, categorical: list[str], validation_season: int) -> EncodedData:
    mask_train = safe_numeric(train["season"]) < validation_season
    mask_val = safe_numeric(train["season"]) == validation_season
    if not mask_train.any() or not mask_val.any():
        raise ValueError("Training or validation split is empty. Check validation season.")

    feature_names = list(features.columns)
    cat_set = set(categorical)
    cat_mask = np.array([c in cat_set for c in feature_names], dtype=bool)
    num_cols = [c for c in feature_names if c not in cat_set]
    cat_cols = [c for c in feature_names if c in cat_set]

    Xtr = np.empty((int(mask_train.sum()), len(feature_names)), dtype=np.float32)
    Xva = np.empty((int(mask_val.sum()), len(feature_names)), dtype=np.float32)

    if num_cols:
        num_idx = [feature_names.index(c) for c in num_cols]
        Xtr[:, num_idx] = features.loc[mask_train, num_cols].to_numpy(dtype=np.float32)
        Xva[:, num_idx] = features.loc[mask_val, num_cols].to_numpy(dtype=np.float32)

    encoder = None
    if cat_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )
        tr_cat = features.loc[mask_train, cat_cols].astype("string").fillna("__MISSING__")
        va_cat = features.loc[mask_val, cat_cols].astype("string").fillna("__MISSING__")
        tr_enc = encoder.fit_transform(tr_cat)
        va_enc = encoder.transform(va_cat)
        for j, col in enumerate(cat_cols):
            idx = feature_names.index(col)
            if len(encoder.categories_[j]) > 255:
                raise ValueError(
                    f"Categorical feature {col} has {len(encoder.categories_[j])} levels; HGB max is 255."
                )
            Xtr[:, idx] = tr_enc[:, j]
            Xva[:, idx] = va_enc[:, j]

    y_train = safe_numeric(train.loc[mask_train, "control_success"]).astype(int).to_numpy()
    y_val = safe_numeric(train.loc[mask_val, "control_success"]).astype(int).to_numpy()
    row_id = train.loc[mask_val, "row_id"].to_numpy() if "row_id" in train else train.index[mask_val].to_numpy()
    return EncodedData(
        X_train=Xtr,
        X_val=Xva,
        y_train=y_train,
        y_val=y_val,
        val_row_id=row_id,
        feature_names=feature_names,
        categorical_mask=cat_mask,
        encoder=encoder,
        categorical_columns=cat_cols,
    )


def fit_hgb(data: EncodedData, selected_indices: np.ndarray, args: argparse.Namespace) -> tuple[HistGradientBoostingClassifier, dict[str, float]]:
    cat_mask = data.categorical_mask[selected_indices]
    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        categorical_features=cat_mask,
        early_stopping=False,
        random_state=args.random_state,
    )
    t0 = time.perf_counter()
    model.fit(data.X_train[:, selected_indices], data.y_train)
    pred = model.predict_proba(data.X_val[:, selected_indices])[:, 1]
    metrics = metrics_dict(data.y_val, pred)
    metrics["elapsed_seconds"] = float(time.perf_counter() - t0)
    metrics["feature_count"] = int(len(selected_indices))
    return model, metrics


def save_plot(frame: pd.DataFrame, x: str, y: str, title: str, path: Path, topn: int = 30) -> None:
    if frame.empty:
        return
    plot = frame.sort_values(x, ascending=False).head(topn).sort_values(x)
    plt.figure(figsize=(9, max(5, 0.28 * len(plot))))
    plt.barh(plot[y].astype(str), plot[x].astype(float))
    plt.title(title)
    plt.xlabel(x)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> None:
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train, low_memory=False)
    if args.mode == "quick":
        val = train[train["season"] == args.validation_season]
        hist = train[train["season"] < args.validation_season]
        hist = hist.sample(n=min(250_000, len(hist)), random_state=args.random_state)
        val = val.sample(n=min(80_000, len(val)), random_state=args.random_state)
        train = pd.concat([hist, val], ignore_index=True)
        args.permutation_sample = min(args.permutation_sample, 20_000)
        args.permutation_repeats = min(args.permutation_repeats, 2)
        args.lofo_max_features = min(args.lofo_max_features, 8)
        args.max_iter = min(args.max_iter, 180)

    require_columns(train, ["season", "pitcher_id", "control_success", "row_id"], "train")
    mapping = load_mapping(args.mapping)
    tm = load_trackman(args.trackman, mapping)
    if args.mode == "quick" and len(tm) > 500_000:
        tm = tm.sample(n=500_000, random_state=args.random_state)

    tm_features, tm_names = attach_trackman_features(train, tm)
    features, blocks, categorical = build_features(train, tm_features)

    catalog_rows = []
    for block, cols in blocks.items():
        for col in cols:
            catalog_rows.append(
                {
                    "feature": col,
                    "block": block,
                    "categorical": col in categorical,
                    "protected": col in PROTECTED_FEATURES,
                }
            )
    catalog = pd.DataFrame(catalog_rows)
    catalog.to_csv(outdir / "feature_catalog.csv", index=False)

    data = encode_split(features, train, categorical, args.validation_season)
    all_idx = np.arange(len(data.feature_names), dtype=int)

    model_full, full_metrics = fit_hgb(data, all_idx, args)
    model_rows = [{"model": "full", **full_metrics}]

    block_rows = []
    for block, cols in blocks.items():
        drop = set(cols)
        idx = np.array([i for i, name in enumerate(data.feature_names) if name not in drop], dtype=int)
        _, m = fit_hgb(data, idx, args)
        block_rows.append(
            {
                "block": block,
                "removed_feature_count": len(drop),
                "brier_without": m["brier"],
                "delta_brier_vs_full": m["brier"] - full_metrics["brier"],
                "elapsed_seconds": m["elapsed_seconds"],
            }
        )
    block_df = pd.DataFrame(block_rows).sort_values("delta_brier_vs_full", ascending=False)
    block_df.to_csv(outdir / "block_ablation.csv", index=False)

    rng = np.random.default_rng(args.random_state)
    n_val = len(data.y_val)
    n_perm = min(args.permutation_sample, n_val)
    perm_idx = np.sort(rng.choice(n_val, size=n_perm, replace=False)) if n_perm < n_val else np.arange(n_val)
    perm = permutation_importance(
        model_full,
        data.X_val[perm_idx],
        data.y_val[perm_idx],
        scoring="neg_brier_score",
        n_repeats=args.permutation_repeats,
        random_state=args.random_state,
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": data.feature_names,
            "permutation_delta_brier_mean": perm.importances_mean,
            "permutation_delta_brier_std": perm.importances_std,
        }
    ).merge(catalog, on="feature", how="left")
    perm_df = perm_df.sort_values("permutation_delta_brier_mean", ascending=False)
    perm_df.to_csv(outdir / "permutation_importance.csv", index=False)
    save_plot(
        perm_df,
        "permutation_delta_brier_mean",
        "feature",
        "HGB permutation importance (Brier increase)",
        outdir / "permutation_importance_top30.png",
    )

    weak = perm_df[~perm_df["feature"].isin(PROTECTED_FEATURES)].sort_values(
        "permutation_delta_brier_mean", ascending=True
    )
    lofo_features = weak.head(args.lofo_max_features)["feature"].tolist()
    lofo_rows = []
    for feature in lofo_features:
        idx = np.array([i for i, name in enumerate(data.feature_names) if name != feature], dtype=int)
        _, m = fit_hgb(data, idx, args)
        delta = m["brier"] - full_metrics["brier"]
        lofo_rows.append(
            {
                "feature": feature,
                "brier_without": m["brier"],
                "delta_brier_vs_full": delta,
                "elapsed_seconds": m["elapsed_seconds"],
                "decision": "DROP"
                if delta <= -abs(args.lofo_improvement_threshold)
                else "KEEP",
            }
        )
    lofo_df = pd.DataFrame(lofo_rows)
    if not lofo_df.empty:
        lofo_df = lofo_df.merge(catalog[["feature", "block"]], on="feature", how="left")
        lofo_df = lofo_df.sort_values("delta_brier_vs_full", ascending=False)
    lofo_df.to_csv(outdir / "lofo_results.csv", index=False)
    if not lofo_df.empty:
        save_plot(
            lofo_df,
            "delta_brier_vs_full",
            "feature",
            "LOFO: Brier change when feature is removed",
            outdir / "lofo_delta_brier.png",
            topn=len(lofo_df),
        )

    drop_features = set(lofo_df.loc[lofo_df["decision"] == "DROP", "feature"]) if not lofo_df.empty else set()
    selected_names = [f for f in data.feature_names if f not in drop_features]
    selected_idx = np.array([data.feature_names.index(f) for f in selected_names], dtype=int)
    selected_model, selected_metrics = fit_hgb(data, selected_idx, args)
    model_rows.append({"model": "selected", **selected_metrics})
    pd.DataFrame(model_rows).to_csv(outdir / "model_scores.csv", index=False)

    full_pred = model_full.predict_proba(data.X_val)[:, 1]
    selected_pred = selected_model.predict_proba(data.X_val[:, selected_idx])[:, 1]
    pd.DataFrame(
        {
            "row_id": data.val_row_id,
            "y_true": data.y_val,
            "pred_full": full_pred,
            "pred_selected": selected_pred,
        }
    ).to_csv(outdir / "validation_predictions.csv.gz", index=False, compression="gzip")

    (outdir / "selected_features.txt").write_text("\n".join(selected_names) + "\n", encoding="utf-8")
    (outdir / "dropped_features.txt").write_text("\n".join(sorted(drop_features)) + "\n", encoding="utf-8")

    artifact = {
        "model": selected_model,
        "ordinal_encoder": data.encoder,
        "all_feature_names": data.feature_names,
        "selected_feature_names": selected_names,
        "categorical_columns": data.categorical_columns,
        "categorical_mask_selected": data.categorical_mask[selected_idx],
        "validation_season": args.validation_season,
    }
    joblib.dump(artifact, outdir / "hgb_selected_model.joblib", compress=3)

    config = RunConfig(
        train_path=args.train,
        trackman_path=args.trackman,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        validation_season=args.validation_season,
        random_state=args.random_state,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        permutation_sample=args.permutation_sample,
        permutation_repeats=args.permutation_repeats,
        lofo_max_features=args.lofo_max_features,
        lofo_improvement_threshold=args.lofo_improvement_threshold,
        quick_rows=330000 if args.mode == "quick" else None,
    )
    summary = {
        "config": asdict(config),
        "train_rows": int(len(data.y_train)),
        "validation_rows": int(len(data.y_val)),
        "feature_count_full": len(data.feature_names),
        "trackman_feature_count": len(tm_names),
        "dropped_features": sorted(drop_features),
        "full_metrics": full_metrics,
        "selected_metrics": selected_metrics,
        "brier_change_selected_minus_full": selected_metrics["brier"] - full_metrics["brier"],
    }
    (outdir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    run_pipeline(parse_args())
