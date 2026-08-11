from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "control_success"
ID_COLUMN = "row_id"

RAW_NUMERIC_FEATURES = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
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
    "home_win_expectancy",
    "away_win_expectancy",
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
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

RAW_CATEGORICAL_FEATURES = [
    "top_bottom",
    "game_type",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "base_state",
]

DERIVED_CATEGORICAL_FEATURES = [
    "count_state",
    "inning_phase",
    "platoon_matchup",
    "tm_mapping_grade",
]

CATEGORICAL_FEATURES = [*RAW_CATEGORICAL_FEATURES, *DERIVED_CATEGORICAL_FEATURES]

TRACKMAN_NUMERIC_COLUMNS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]

TRACKMAN_USE_COLUMNS = [
    "season",
    "pitcher_trackman_id",
    "pitch_type_group",
    *TRACKMAN_NUMERIC_COLUMNS,
]

ASOF_TREND_FEATURES = [
    "asof_pitcher_success_trend_1v3",
    "asof_pitcher_success_trend_1v5",
    "asof_pitcher_success_trend_3v5",
    "asof_pitcher_middle_trend_1v3",
    "asof_pitcher_middle_trend_1v5",
    "asof_pitcher_middle_trend_3v5",
    "asof_pitcher_success_vs_prev5",
    "asof_pitcher_middle_vs_prev5",
    "asof_pitcher_batter_success_gap",
    "asof_pitcher_batter_middle_gap",
    "asof_pitcher_strike_ball_gap",
    "asof_success_trend_source_n",
    "asof_middle_trend_source_n",
]


@dataclass(frozen=True)
class FeatureBundle:
    features: pd.DataFrame
    target: pd.Series
    row_ids: pd.Series
    seasons: pd.Series
    train_rows: np.ndarray
    test_rows: np.ndarray
    categorical_features: list[str]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def _safe_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_asof_trend_features(current: pd.DataFrame) -> pd.DataFrame:
    """Build row-wise trend contrasts from official pre-pitch ``asof`` columns.

    These features never use ``control_success``. A contrast remains missing when
    either source value is missing; the two ``source_n`` columns let LightGBM
    distinguish an unavailable trend from an observed zero trend.
    """
    success = {
        window: current[f"asof_pitcher_prev{window}_game_success_rate"]
        for window in [1, 3, 5]
    }
    middle = {
        window: current[f"asof_pitcher_prev{window}_game_middle_rate"]
        for window in [1, 3, 5]
    }

    out = pd.DataFrame(index=current.index)
    out["asof_pitcher_success_trend_1v3"] = success[1] - success[3]
    out["asof_pitcher_success_trend_1v5"] = success[1] - success[5]
    out["asof_pitcher_success_trend_3v5"] = success[3] - success[5]
    out["asof_pitcher_middle_trend_1v3"] = middle[1] - middle[3]
    out["asof_pitcher_middle_trend_1v5"] = middle[1] - middle[5]
    out["asof_pitcher_middle_trend_3v5"] = middle[3] - middle[5]

    out["asof_pitcher_success_vs_prev5"] = (
        current["asof_pitcher_success_rate"] - success[5]
    )
    out["asof_pitcher_middle_vs_prev5"] = (
        current["asof_pitcher_middle_rate"] - middle[5]
    )
    out["asof_pitcher_batter_success_gap"] = (
        current["asof_pitcher_success_rate"]
        - current["asof_batter_success_rate"]
    )
    out["asof_pitcher_batter_middle_gap"] = (
        current["asof_pitcher_middle_rate"]
        - current["asof_batter_middle_rate"]
    )
    out["asof_pitcher_strike_ball_gap"] = (
        current["asof_pitcher_strike_rate"]
        - current["asof_pitcher_ball_rate"]
    )
    out["asof_success_trend_source_n"] = pd.concat(
        [success[1], success[3], success[5]], axis=1
    ).notna().sum(axis=1)
    out["asof_middle_trend_source_n"] = pd.concat(
        [middle[1], middle[3], middle[5]], axis=1
    ).notna().sum(axis=1)
    return out[ASOF_TREND_FEATURES]


def build_current_features(
    raw: pd.DataFrame,
    include_asof_trends: bool = False,
) -> pd.DataFrame:
    required = [ID_COLUMN, *RAW_NUMERIC_FEATURES, *RAW_CATEGORICAL_FEATURES]
    _require_columns(raw, required, "pitch data")

    frame = raw[required].copy()
    frame = _safe_numeric(frame, RAW_NUMERIC_FEATURES)

    for column in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]:
        frame[column] = _normalize_id(frame[column])
    for column in ["top_bottom", "game_type", "base_state"]:
        frame[column] = frame[column].astype("string")

    balls = frame["balls_before"]
    strikes = frame["strikes_before"]
    score_diff = frame["score_diff_pitcher_team"]

    frame["count_state"] = (
        balls.astype("Int64").astype("string")
        + "-"
        + strikes.astype("Int64").astype("string")
    )
    frame["pitcher_ahead"] = (strikes > balls).astype("int8")
    frame["batter_ahead"] = (balls > strikes).astype("int8")
    frame["two_strike"] = (strikes == 2).astype("int8")
    frame["three_ball"] = (balls == 3).astype("int8")
    frame["full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")

    frame["has_runner"] = (frame["num_runners_on"] > 0).astype("int8")
    frame["risp"] = (
        (frame["runner_on_2b"] == 1) | (frame["runner_on_3b"] == 1)
    ).astype("int8")
    frame["bases_loaded"] = (frame["num_runners_on"] == 3).astype("int8")

    frame["is_tie"] = (score_diff == 0).astype("int8")
    frame["score_diff_abs"] = score_diff.abs()
    frame["close_game"] = (score_diff.abs() <= 2).astype("int8")
    frame["late_inning"] = (frame["inning"] >= 7).astype("int8")
    frame["extra_inning"] = (frame["inning"] >= 10).astype("int8")
    frame["abs_era"] = (frame["season"] >= 2024).astype("int8")

    inning = frame["inning"]
    frame["inning_phase"] = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        ["early", "middle", "late"],
        default="extra",
    )
    frame["inning_phase"] = frame["inning_phase"].astype("string")

    pitcher_hand = frame["pitcher_hand"].astype("string")
    batter_hand = frame["batter_hand"].astype("string")
    frame["platoon_matchup"] = np.where(
        pitcher_hand.isna() | batter_hand.isna(),
        "missing",
        np.where(pitcher_hand == batter_hand, "same", "opposite"),
    )
    frame["platoon_matchup"] = frame["platoon_matchup"].astype("string")

    for source in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        frame[f"{source}_log1p"] = np.log1p(frame[source].clip(lower=0))
        frame[f"{source}_missing"] = frame[source].isna().astype("int8")

    if include_asof_trends:
        frame = pd.concat([frame, build_asof_trend_features(frame)], axis=1)

    return frame


def _lookup(
    rows: pd.DataFrame,
    table: pd.DataFrame,
    keys: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(np.nan, index=rows.index, columns=value_columns)

    indexed = table.set_index(keys)
    if len(keys) == 1:
        key = rows[keys[0]]
        return pd.DataFrame(
            {column: key.map(indexed[column]) for column in value_columns},
            index=rows.index,
        )

    row_index = pd.MultiIndex.from_frame(rows[keys])
    matched = indexed[value_columns].reindex(row_index)
    matched.index = rows.index
    return matched


def _smoothed_table(
    history: pd.DataFrame,
    keys: list[str],
    prefix: str,
    smoothing: float,
    prior_keys: list[str] | None = None,
) -> pd.DataFrame:
    grouped = (
        history.groupby(keys, observed=True, dropna=False)[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )
    if grouped.empty:
        return grouped

    global_rate = float(history[TARGET].mean())
    prior_keys = prior_keys or []
    if prior_keys:
        prior = (
            history.groupby(prior_keys, observed=True, dropna=False)[TARGET]
            .mean()
            .rename("_prior")
            .reset_index()
        )
        grouped = grouped.merge(prior, on=prior_keys, how="left")
        prior_value = grouped["_prior"].fillna(global_rate)
    else:
        prior_value = global_rate

    grouped[f"{prefix}_rate"] = (
        grouped["sum"] + smoothing * prior_value
    ) / (grouped["count"] + smoothing)
    grouped[f"{prefix}_n"] = grouped["count"].astype("float32")
    grouped[f"{prefix}_log_n"] = np.log1p(grouped["count"]).astype("float32")
    return grouped[[*keys, f"{prefix}_rate", f"{prefix}_n", f"{prefix}_log_n"]]


def _recent_table(
    history: pd.DataFrame,
    n_pitches: int,
    global_rate: float,
    smoothing: float,
) -> pd.DataFrame:
    recent = history.groupby("pitcher_id", observed=True, sort=False).tail(n_pitches)
    grouped = recent.groupby("pitcher_id", observed=True)[TARGET].agg(["sum", "count"])
    prefix = f"pitcher_recent{n_pitches}"
    grouped[f"{prefix}_rate"] = (
        grouped["sum"] + smoothing * global_rate
    ) / (grouped["count"] + smoothing)
    grouped[f"{prefix}_n"] = grouped["count"].astype("float32")
    return grouped[[f"{prefix}_rate", f"{prefix}_n"]].reset_index()


def build_history_features(
    train_raw: pd.DataFrame,
    all_current: pd.DataFrame,
    smoothing: float = 50.0,
) -> pd.DataFrame:
    _require_columns(train_raw, [TARGET], "train data")
    train_current = all_current.loc[train_raw.index].copy()
    train_current[TARGET] = pd.to_numeric(train_raw[TARGET], errors="raise").astype("int8")
    train_current["_source_order"] = np.arange(len(train_current), dtype="int64")

    history_columns = [
        "pitcher_career_rate",
        "pitcher_career_n",
        "pitcher_career_log_n",
        "pitcher_prev_season_rate",
        "pitcher_prev_season_n",
        "pitcher_prev_season_log_n",
        "pitcher_count_rate",
        "pitcher_count_n",
        "pitcher_count_log_n",
        "pitcher_runner_rate",
        "pitcher_runner_n",
        "pitcher_runner_log_n",
        "pitcher_risp_rate",
        "pitcher_risp_n",
        "pitcher_risp_log_n",
        "pitcher_batter_hand_rate",
        "pitcher_batter_hand_n",
        "pitcher_batter_hand_log_n",
        "pitcher_recent20_rate",
        "pitcher_recent20_n",
        "pitcher_recent50_rate",
        "pitcher_recent50_n",
        "pitcher_recent100_rate",
        "pitcher_recent100_n",
        "pitcher_last_observed_season",
        "pitcher_season_gap",
    ]
    output = pd.DataFrame(np.nan, index=all_current.index, columns=history_columns)

    for season in sorted(all_current["season"].dropna().astype(int).unique()):
        target_rows = all_current.loc[all_current["season"] == season]
        history = train_current.loc[train_current["season"] < season]
        if target_rows.empty or history.empty:
            continue

        global_rate = float(history[TARGET].mean())
        specs = [
            (["pitcher_id"], "pitcher_career", []),
            (["pitcher_id", "count_state"], "pitcher_count", ["count_state"]),
            (["pitcher_id", "has_runner"], "pitcher_runner", ["has_runner"]),
            (["pitcher_id", "risp"], "pitcher_risp", ["risp"]),
            (
                ["pitcher_id", "batter_hand"],
                "pitcher_batter_hand",
                ["batter_hand"],
            ),
        ]
        for keys, prefix, prior_keys in specs:
            table = _smoothed_table(history, keys, prefix, smoothing, prior_keys)
            values = [f"{prefix}_rate", f"{prefix}_n", f"{prefix}_log_n"]
            output.loc[target_rows.index, values] = _lookup(
                target_rows, table, keys, values
            ).to_numpy()

        previous = history.loc[history["season"] == season - 1]
        if not previous.empty:
            previous_table = _smoothed_table(
                previous,
                ["pitcher_id"],
                "pitcher_prev_season",
                smoothing,
            )
            previous_values = [
                "pitcher_prev_season_rate",
                "pitcher_prev_season_n",
                "pitcher_prev_season_log_n",
            ]
            output.loc[target_rows.index, previous_values] = _lookup(
                target_rows,
                previous_table,
                ["pitcher_id"],
                previous_values,
            ).to_numpy()

        for n_pitches in [20, 50, 100]:
            table = _recent_table(history, n_pitches, global_rate, smoothing=20.0)
            values = [f"pitcher_recent{n_pitches}_rate", f"pitcher_recent{n_pitches}_n"]
            output.loc[target_rows.index, values] = _lookup(
                target_rows, table, ["pitcher_id"], values
            ).to_numpy()

        last_season = (
            history.groupby("pitcher_id", observed=True)["season"]
            .max()
            .rename("pitcher_last_observed_season")
            .reset_index()
        )
        matched = _lookup(
            target_rows,
            last_season,
            ["pitcher_id"],
            ["pitcher_last_observed_season"],
        )
        output.loc[target_rows.index, "pitcher_last_observed_season"] = matched.iloc[
            :, 0
        ].to_numpy()
        output.loc[target_rows.index, "pitcher_season_gap"] = (
            season - matched.iloc[:, 0]
        ).to_numpy()

    output["pitcher_history_missing"] = output["pitcher_career_n"].isna().astype("int8")
    output["prev_vs_career_rate"] = (
        output["pitcher_prev_season_rate"] - output["pitcher_career_rate"]
    )
    output["recent50_vs_career_rate"] = (
        output["pitcher_recent50_rate"] - output["pitcher_career_rate"]
    )
    return output.astype({column: "float32" for column in output.columns})


def aggregate_trackman(
    path: str | Path,
    chunksize: int = 350_000,
    nrows: int | None = None,
) -> pd.DataFrame:
    path = Path(path)
    partials: list[pd.DataFrame] = []
    remaining = nrows

    reader = pd.read_csv(
        path,
        usecols=TRACKMAN_USE_COLUMNS,
        encoding="utf-8-sig",
        chunksize=chunksize,
        nrows=nrows,
        low_memory=False,
    )
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)

        chunk["pitcher_trackman_id"] = _normalize_id(chunk["pitcher_trackman_id"])
        chunk["season"] = pd.to_numeric(chunk["season"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["pitcher_trackman_id", "season"])
        chunk["season"] = chunk["season"].astype("int16")

        for column in TRACKMAN_NUMERIC_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            chunk[f"{column}__sq"] = np.square(chunk[column])

        pitch_group = chunk["pitch_type_group"].astype("string").str.lower()
        chunk["usage_fastball"] = pitch_group.eq("fastball").astype("int8")
        chunk["usage_breaking"] = pitch_group.eq("breaking").astype("int8")
        chunk["usage_offspeed"] = pitch_group.eq("offspeed").astype("int8")
        chunk["usage_other"] = (
            ~(pitch_group.isin(["fastball", "breaking", "offspeed"]))
        ).astype("int8")
        chunk["tm_pitch_n"] = 1

        aggregation: dict[str, tuple[str, str]] = {"tm_pitch_n": ("tm_pitch_n", "sum")}
        for column in TRACKMAN_NUMERIC_COLUMNS:
            aggregation[f"{column}__n"] = (column, "count")
            aggregation[f"{column}__sum"] = (column, "sum")
            aggregation[f"{column}__sqsum"] = (f"{column}__sq", "sum")
        for usage in ["usage_fastball", "usage_breaking", "usage_offspeed", "usage_other"]:
            aggregation[f"{usage}__sum"] = (usage, "sum")

        grouped = chunk.groupby(
            ["pitcher_trackman_id", "season"], observed=True
        ).agg(**aggregation)
        partials.append(grouped)

    if not partials:
        raise ValueError(f"No Trackman rows were read from {path}")

    summary = pd.concat(partials).groupby(level=[0, 1], observed=True).sum()
    return summary.reset_index()


def load_mapping(
    path: str | Path,
    accepted_grades: Iterable[str] = ("확정", "높음"),
) -> pd.DataFrame:
    mapping = pd.read_csv(path, encoding="utf-8-sig")
    required = ["pitcher_id", "pitcher_trackman_id", "신뢰등급", "매칭순도"]
    _require_columns(mapping, required, "pitcher mapping")
    mapping["pitcher_id"] = _normalize_id(mapping["pitcher_id"])
    mapping["pitcher_trackman_id"] = _normalize_id(mapping["pitcher_trackman_id"])
    mapping["매칭순도"] = pd.to_numeric(mapping["매칭순도"], errors="coerce")
    mapping["mapping_accepted"] = (
        mapping["신뢰등급"].isin(set(accepted_grades))
        & mapping["pitcher_trackman_id"].notna()
    )
    return mapping


def _trackman_statistics(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["pitcher_trackman_id"])

    grouped = summary.groupby("pitcher_trackman_id", observed=True).sum(numeric_only=True)
    out = pd.DataFrame(index=grouped.index)
    out[f"{prefix}_pitch_n"] = grouped["tm_pitch_n"]

    for column in TRACKMAN_NUMERIC_COLUMNS:
        count = grouped[f"{column}__n"].replace(0, np.nan)
        mean = grouped[f"{column}__sum"] / count
        variance = grouped[f"{column}__sqsum"] / count - np.square(mean)
        out[f"{prefix}_{column}_mean"] = mean
        out[f"{prefix}_{column}_std"] = np.sqrt(variance.clip(lower=0))

    total = grouped["tm_pitch_n"].replace(0, np.nan)
    for usage in ["usage_fastball", "usage_breaking", "usage_offspeed", "usage_other"]:
        out[f"{prefix}_{usage}_rate"] = grouped[f"{usage}__sum"] / total

    return out.reset_index()


def build_trackman_features(
    all_current: pd.DataFrame,
    trackman_summary: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    accepted = mapping.loc[mapping["mapping_accepted"]].drop_duplicates("pitcher_id")
    id_map = accepted.set_index("pitcher_id")["pitcher_trackman_id"]
    purity_map = mapping.drop_duplicates("pitcher_id").set_index("pitcher_id")["매칭순도"]
    grade_map = mapping.drop_duplicates("pitcher_id").set_index("pitcher_id")["신뢰등급"]

    mapped_id = all_current["pitcher_id"].map(id_map)
    output = pd.DataFrame(index=all_current.index)
    output["tm_mapping_purity"] = all_current["pitcher_id"].map(purity_map).astype("float32")
    output["tm_mapping_grade"] = (
        all_current["pitcher_id"].map(grade_map).fillna("미매칭").astype("string")
    )
    output["tm_mapping_accepted"] = mapped_id.notna().astype("int8")

    career_feature_names = ["tm_career_pitch_n"]
    previous_feature_names = ["tm_prev_pitch_n"]
    earlier_feature_names = ["tm_earlier_pitch_n"]
    for column in TRACKMAN_NUMERIC_COLUMNS:
        career_feature_names.extend(
            [f"tm_career_{column}_mean", f"tm_career_{column}_std"]
        )
        previous_feature_names.extend(
            [f"tm_prev_{column}_mean", f"tm_prev_{column}_std"]
        )
        earlier_feature_names.extend(
            [f"tm_earlier_{column}_mean", f"tm_earlier_{column}_std"]
        )
    for usage in ["usage_fastball", "usage_breaking", "usage_offspeed", "usage_other"]:
        career_feature_names.append(f"tm_career_{usage}_rate")
        previous_feature_names.append(f"tm_prev_{usage}_rate")
        earlier_feature_names.append(f"tm_earlier_{usage}_rate")

    for column in [*career_feature_names, *previous_feature_names]:
        output[column] = np.nan
    output["tm_last_observed_season"] = np.nan
    output["tm_season_gap"] = np.nan

    for season in sorted(all_current["season"].dropna().astype(int).unique()):
        mask = all_current["season"] == season
        row_ids = mapped_id.loc[mask]
        if not mask.any():
            continue

        career_rows = trackman_summary.loc[trackman_summary["season"] < season]
        previous_rows = trackman_summary.loc[trackman_summary["season"] == season - 1]
        earlier_rows = trackman_summary.loc[trackman_summary["season"] < season - 1]

        career = _trackman_statistics(career_rows, "tm_career").set_index(
            "pitcher_trackman_id"
        )
        previous = _trackman_statistics(previous_rows, "tm_prev").set_index(
            "pitcher_trackman_id"
        )
        earlier = _trackman_statistics(earlier_rows, "tm_earlier").set_index(
            "pitcher_trackman_id"
        )

        if not career.empty:
            output.loc[mask, career_feature_names] = career.reindex(row_ids)[
                career_feature_names
            ].to_numpy()
        if not previous.empty:
            output.loc[mask, previous_feature_names] = previous.reindex(row_ids)[
                previous_feature_names
            ].to_numpy()

        last_season = (
            career_rows.groupby("pitcher_trackman_id", observed=True)["season"].max()
        )
        observed = row_ids.map(last_season)
        output.loc[mask, "tm_last_observed_season"] = observed.to_numpy()
        output.loc[mask, "tm_season_gap"] = (season - observed).to_numpy()

        if not previous.empty and not earlier.empty:
            for column in TRACKMAN_NUMERIC_COLUMNS:
                prev_name = f"tm_prev_{column}_mean"
                earlier_name = f"tm_earlier_{column}_mean"
                delta_name = f"tm_delta_{column}_mean"
                previous_values = previous[prev_name].reindex(row_ids).to_numpy()
                earlier_values = earlier[earlier_name].reindex(row_ids).to_numpy()
                output.loc[mask, delta_name] = previous_values - earlier_values

    output["tm_history_missing"] = output["tm_career_pitch_n"].isna().astype("int8")
    numeric = output.columns.difference(["tm_mapping_grade"])
    output[numeric] = output[numeric].astype("float32")
    return output


def assemble_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame | None = None,
    trackman_summary: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
    smoothing: float = 50.0,
    include_asof_trends: bool = False,
) -> FeatureBundle:
    _require_columns(train_raw, [ID_COLUMN, "season", TARGET], "train data")
    train = train_raw.copy()
    train["_is_train"] = True

    frames = [train]
    if test_raw is not None:
        test = test_raw.copy()
        test["_is_train"] = False
        if TARGET not in test.columns:
            test[TARGET] = np.nan
        frames.append(test)

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    current = build_current_features(
        combined,
        include_asof_trends=include_asof_trends,
    )
    history = build_history_features(combined.loc[combined["_is_train"]], current, smoothing)

    feature_parts = [current.drop(columns=[ID_COLUMN]), history]
    if trackman_summary is not None and mapping is not None:
        feature_parts.append(build_trackman_features(current, trackman_summary, mapping))

    features = pd.concat(feature_parts, axis=1)
    for column in CATEGORICAL_FEATURES:
        if column not in features.columns:
            continue
        features[column] = features[column].astype("string").fillna("__MISSING__")
        features[column] = features[column].astype("category")

    numeric_columns = features.select_dtypes(exclude=["category"]).columns
    features[numeric_columns] = features[numeric_columns].replace([np.inf, -np.inf], np.nan)
    features[numeric_columns] = features[numeric_columns].astype("float32")

    train_mask = combined["_is_train"].to_numpy(dtype=bool)
    return FeatureBundle(
        features=features,
        target=pd.to_numeric(combined[TARGET], errors="coerce"),
        row_ids=combined[ID_COLUMN].astype("string"),
        seasons=pd.to_numeric(combined["season"], errors="raise").astype("int16"),
        train_rows=np.flatnonzero(train_mask),
        test_rows=np.flatnonzero(~train_mask),
        categorical_features=[
            column for column in CATEGORICAL_FEATURES if column in features.columns
        ],
    )


def assemble_raw_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame | None = None,
) -> FeatureBundle:
    """Build the official-column baseline with only model-required cleaning.

    This deliberately excludes every hand-built current-row feature, labeled
    history aggregate, smoothing result, and Trackman aggregate used by V1.
    Numeric coercion and categorical typing remain because sklearn estimators
    cannot consume arbitrary CSV strings or infinite values directly.
    """
    required = [ID_COLUMN, "season", TARGET, *RAW_NUMERIC_FEATURES, *RAW_CATEGORICAL_FEATURES]
    _require_columns(train_raw, required, "train data")

    train = train_raw.copy()
    train["_is_train"] = True
    frames = [train]
    if test_raw is not None:
        _require_columns(
            test_raw,
            [ID_COLUMN, "season", *RAW_NUMERIC_FEATURES, *RAW_CATEGORICAL_FEATURES],
            "test data",
        )
        test = test_raw.copy()
        test["_is_train"] = False
        if TARGET not in test.columns:
            test[TARGET] = np.nan
        frames.append(test)

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    features = combined[[*RAW_NUMERIC_FEATURES, *RAW_CATEGORICAL_FEATURES]].copy()

    for column in RAW_NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features[RAW_NUMERIC_FEATURES] = features[RAW_NUMERIC_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    features[RAW_NUMERIC_FEATURES] = features[RAW_NUMERIC_FEATURES].astype("float32")

    for column in RAW_CATEGORICAL_FEATURES:
        features[column] = features[column].astype("string").fillna("__MISSING__")
        features[column] = features[column].astype("category")

    train_mask = combined["_is_train"].to_numpy(dtype=bool)
    return FeatureBundle(
        features=features,
        target=pd.to_numeric(combined[TARGET], errors="coerce"),
        row_ids=combined[ID_COLUMN].astype("string"),
        seasons=pd.to_numeric(combined["season"], errors="raise").astype("int16"),
        train_rows=np.flatnonzero(train_mask),
        test_rows=np.flatnonzero(~train_mask),
        categorical_features=list(RAW_CATEGORICAL_FEATURES),
    )
