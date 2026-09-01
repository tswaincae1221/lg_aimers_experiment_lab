from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DRIFT_METRICS, PITCH_GROUPS, TRACKMAN_METRICS, TRACKMAN_REQUIRED_COLUMNS,
    EngineeredData, _atomic_to_pickle, _file_signature, _normalize_id, _read_csv,
    _require_columns, _safe_numeric, _stable_hash, _to_probability,
)

LOGGER = logging.getLogger(__name__)

def load_mapping(path: Path, accepted_grades: tuple[str, ...] = ("확정", "높음")) -> pd.DataFrame:
    mapping = _read_csv(path)
    _require_columns(mapping, ["pitcher_id", "pitcher_trackman_id"], "mapping CSV")
    grade_col = "신뢰등급" if "신뢰등급" in mapping.columns else None
    purity_col = "매칭순도" if "매칭순도" in mapping.columns else None
    mapping["pitcher_id"] = _normalize_id(mapping["pitcher_id"])
    mapping["pitcher_trackman_id"] = _normalize_id(mapping["pitcher_trackman_id"])
    if grade_col:
        mapping["mapping_accepted"] = mapping[grade_col].astype("string").isin(accepted_grades)
        mapping["tm_mapping_grade"] = mapping[grade_col].astype("string").fillna("__MISSING__")
    else:
        mapping["mapping_accepted"] = mapping["pitcher_trackman_id"].notna()
        mapping["tm_mapping_grade"] = np.where(mapping["mapping_accepted"], "accepted", "missing")
    mapping["tm_mapping_purity"] = (
        pd.to_numeric(mapping[purity_col], errors="coerce") if purity_col else np.nan
    )
    mapping = mapping.loc[mapping["mapping_accepted"]].drop_duplicates("pitcher_id", keep="first")
    return mapping[
        ["pitcher_id", "pitcher_trackman_id", "tm_mapping_grade", "tm_mapping_purity"]
    ].reset_index(drop=True)


def _group_sufficient_stats(chunk: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "pitcher_trackman_id", "pitch_type_group"]
    work = chunk[keys].copy()
    for metric in TRACKMAN_METRICS:
        values = _safe_numeric(chunk[metric])
        work[f"{metric}__n"] = values.notna().astype("int32")
        work[f"{metric}__sum"] = values.fillna(0.0)
        work[f"{metric}__sqsum"] = values.pow(2).fillna(0.0)
    work["tm_group_pitch_n"] = 1
    return work.groupby(keys, observed=True, dropna=False, sort=False).sum(numeric_only=True).reset_index()


def _context_counts(chunk: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "season",
        "pitcher_trackman_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "pitch_type_group",
    ]
    return (
        chunk.groupby(keys, observed=True, dropna=False, sort=False)
        .size()
        .rename("context_n")
        .reset_index()
    )


def _drift_sufficient_stats(chunk: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "pitcher_trackman_id", "trackman_game_id"]
    work = chunk[keys].copy()
    x_all = _safe_numeric(chunk["pitch_no"])
    for metric in DRIFT_METRICS:
        y = _safe_numeric(chunk[metric])
        valid = x_all.notna() & y.notna()
        x = x_all.where(valid, 0.0)
        yv = y.where(valid, 0.0)
        work[f"{metric}__n"] = valid.astype("int32")
        work[f"{metric}__sum_x"] = x
        work[f"{metric}__sum_x2"] = x.pow(2)
        work[f"{metric}__sum_y"] = yv
        work[f"{metric}__sum_xy"] = x * yv
    return work.groupby(keys, observed=True, dropna=False, sort=False).sum(numeric_only=True).reset_index()


def _combine_partial(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    joined = pd.concat(parts, ignore_index=True)
    return joined.groupby(keys, observed=True, dropna=False, sort=False).sum(numeric_only=True).reset_index()


def aggregate_trackman(
    path: Path,
    *,
    max_rows: int | None = None,
    chunksize: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chunk-safe Trackman aggregation.

    Returns sufficient statistics for pitch-group repeatability, contextual pitch-mix
    counts, and per-game drift slopes. Partial games/groups split across CSV chunks are
    consolidated exactly after the streaming pass.
    """
    group_parts: list[pd.DataFrame] = []
    context_parts: list[pd.DataFrame] = []
    drift_parts: list[pd.DataFrame] = []
    consumed = 0
    reader = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda column: column in set(TRACKMAN_REQUIRED_COLUMNS),
        chunksize=chunksize,
    )
    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - consumed
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        consumed += len(chunk)
        _require_columns(chunk, TRACKMAN_REQUIRED_COLUMNS, "trackman_history.csv")
        chunk["season"] = pd.to_numeric(chunk["season"], errors="coerce")
        chunk["pitcher_trackman_id"] = _normalize_id(chunk["pitcher_trackman_id"])
        chunk["pitch_type_group"] = chunk["pitch_type_group"].astype("string").str.lower()
        chunk = chunk.loc[chunk["pitch_type_group"].isin(PITCH_GROUPS)].copy()
        if chunk.empty:
            continue
        chunk["batter_hand"] = chunk["batter_hand"].astype("string").fillna("__MISSING__")
        group_parts.append(_group_sufficient_stats(chunk))
        context_parts.append(_context_counts(chunk))
        drift_parts.append(_drift_sufficient_stats(chunk))
        LOGGER.info("Trackman aggregated rows: %s", f"{consumed:,}")
        if max_rows is not None and consumed >= max_rows:
            break

    group_keys = ["season", "pitcher_trackman_id", "pitch_type_group"]
    context_keys = [
        "season",
        "pitcher_trackman_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "pitch_type_group",
    ]
    drift_keys = ["season", "pitcher_trackman_id", "trackman_game_id"]
    return (
        _combine_partial(group_parts, group_keys),
        _combine_partial(context_parts, context_keys),
        _combine_partial(drift_parts, drift_keys),
    )


def load_or_aggregate_trackman(
    path: Path,
    output_dir: Path,
    *,
    max_rows: int | None,
    chunksize: int,
    rerun: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signature = _stable_hash(
        {
            "trackman": _file_signature(path),
            "max_rows": max_rows,
            "aggregation_version": 2,
        }
    )[:16]
    cache_dir = output_dir / "cache"
    paths = {
        "group": cache_dir / f"trackman_group_{signature}.pkl",
        "context": cache_dir / f"trackman_context_{signature}.pkl",
        "drift": cache_dir / f"trackman_drift_{signature}.pkl",
    }
    if not rerun and all(target.is_file() for target in paths.values()):
        LOGGER.info("Trackman cache hit: %s", signature)
        return (
            pd.read_pickle(paths["group"]),
            pd.read_pickle(paths["context"]),
            pd.read_pickle(paths["drift"]),
        )
    result = aggregate_trackman(path, max_rows=max_rows, chunksize=chunksize)
    for key, frame in zip(("group", "context", "drift"), result, strict=True):
        _atomic_to_pickle(frame, paths[key])
    return result


def _pooled_stat_table(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    keys = ["pitcher_trackman_id", "pitch_type_group"]
    numeric = [column for column in history.columns if column not in ["season", *keys]]
    pooled = history.groupby(keys, observed=True, sort=False)[numeric].sum().reset_index()
    for metric in TRACKMAN_METRICS:
        n = pooled[f"{metric}__n"].astype("float64")
        total = pooled[f"{metric}__sum"].astype("float64")
        sq = pooled[f"{metric}__sqsum"].astype("float64")
        pooled[f"{metric}_mean"] = total / n.replace(0, np.nan)
        variance = (sq - total.pow(2) / n.replace(0, np.nan)) / (n - 1).replace(0, np.nan)
        pooled[f"{metric}_sd"] = np.sqrt(variance.clip(lower=0))
    return pooled


def _shrunken_dispersion_profiles(
    pooled: pd.DataFrame,
    *,
    shrinkage_pitches: float,
) -> pd.DataFrame:
    if pooled.empty:
        return pooled
    out = pooled[["pitcher_trackman_id", "pitch_type_group", "tm_group_pitch_n"]].copy()
    out["tm_group_reliability"] = pooled["tm_group_pitch_n"] / (
        pooled["tm_group_pitch_n"] + shrinkage_pitches
    )
    component_sets = {
        "release": ["rel_height_sd", "rel_side_sd", "extension_sd"],
        "velocity": ["rel_speed_sd", "zone_speed_sd"],
        "movement": ["induced_vert_break_sd", "horz_break_sd"],
        "spin": ["spin_rate_sd"],
    }
    for family, columns in component_sets.items():
        ratios = []
        for column in columns:
            source = pooled[column].astype("float64")
            medians = pooled.groupby("pitch_type_group", observed=True)[column].transform("median")
            ratio = source / medians.replace(0, np.nan)
            adjusted = 1.0 + out["tm_group_reliability"] * (ratio - 1.0)
            ratios.append(adjusted)
        out[f"tm_{family}_dispersion"] = pd.concat(ratios, axis=1).mean(axis=1, skipna=True)
    return out


def _game_slopes(drift_stats: pd.DataFrame, min_game_pitches: int = 10) -> pd.DataFrame:
    if drift_stats.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "pitcher_trackman_id",
                "trackman_game_id",
                *[f"{metric}_slope" for metric in DRIFT_METRICS],
            ]
        )
    result = drift_stats[["season", "pitcher_trackman_id", "trackman_game_id"]].copy()
    for metric in DRIFT_METRICS:
        n = drift_stats[f"{metric}__n"].astype("float64")
        sx = drift_stats[f"{metric}__sum_x"].astype("float64")
        sx2 = drift_stats[f"{metric}__sum_x2"].astype("float64")
        sy = drift_stats[f"{metric}__sum_y"].astype("float64")
        sxy = drift_stats[f"{metric}__sum_xy"].astype("float64")
        denominator = n * sx2 - sx.pow(2)
        slope = (n * sxy - sx * sy) / denominator.replace(0, np.nan)
        result[f"{metric}_slope"] = slope.where(n >= min_game_pitches)
    return result


def _weighted_expected(
    row_rates: pd.DataFrame,
    values: pd.DataFrame,
    group_columns: dict[str, str],
) -> pd.Series:
    numerator = pd.Series(0.0, index=row_rates.index, dtype="float64")
    denominator = pd.Series(0.0, index=row_rates.index, dtype="float64")
    for group, value_col in group_columns.items():
        rate = _to_probability(row_rates[group])
        value = _safe_numeric(values[value_col])
        valid = rate.notna() & value.notna()
        numerator = numerator + (rate * value).where(valid, 0.0)
        denominator = denominator + rate.where(valid, 0.0)
    return (numerator / denominator.replace(0, np.nan)).astype("float32")


def build_trackman_features(
    main: EngineeredData,
    mapping: pd.DataFrame,
    group_stats: pd.DataFrame,
    context_stats: pd.DataFrame,
    drift_stats: pd.DataFrame,
    *,
    shrinkage_pitches: float = 100.0,
    context_smoothing: float = 50.0,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    """Build season-safe compact Trackman priors for every main-data row."""
    features = pd.DataFrame(index=main.features.index)
    blocks: dict[str, list[str]] = {
        "trackman_repeatability": [],
        "trackman_context": [],
        "trackman_drift": [],
    }
    categorical: list[str] = []

    pitcher_id = main.features.attrs.get("pitcher_id")
    balls = main.features.attrs.get("balls_before")
    strikes = main.features.attrs.get("strikes_before")
    batter_hand = main.features.attrs.get("batter_hand")
    if any(item is None for item in (pitcher_id, balls, strikes, batter_hand)):
        raise ValueError("Main feature join keys are missing; build_main_features must run first")

    map_index = mapping.set_index("pitcher_id")
    trackman_id = pitcher_id.map(map_index["pitcher_trackman_id"])
    features["tm_available"] = trackman_id.notna().astype("int8")
    features["tm_mapping_purity"] = pitcher_id.map(map_index["tm_mapping_purity"]).astype("float32")
    features["tm_mapping_grade"] = pitcher_id.map(map_index["tm_mapping_grade"]).astype("string").fillna(
        "__MISSING__"
    )
    categorical.append("tm_mapping_grade")
    blocks["trackman_repeatability"].extend(
        ["tm_available", "tm_mapping_purity", "tm_mapping_grade"]
    )

    game_slopes = _game_slopes(drift_stats)
    prediction_seasons = sorted(main.seasons.dropna().astype(int).unique())
    for season in prediction_seasons:
        row_mask = main.seasons.astype(int).eq(season)
        row_idx = main.features.index[row_mask]
        row_tm = trackman_id.loc[row_idx]
        if row_tm.notna().sum() == 0:
            continue

        historical_group = group_stats.loc[pd.to_numeric(group_stats["season"], errors="coerce") < season]
        pooled = _pooled_stat_table(historical_group)
        profiles = _shrunken_dispersion_profiles(pooled, shrinkage_pitches=shrinkage_pitches)
        if profiles.empty:
            continue

        # Pivot pitcher x group profiles into row-addressable columns.
        pivot = profiles.pivot(index="pitcher_trackman_id", columns="pitch_type_group")
        row_profile = pd.DataFrame(index=row_idx)
        for group in PITCH_GROUPS:
            for family in ("release", "velocity", "movement", "spin"):
                source = (f"tm_{family}_dispersion", group)
                name = f"tm_{group}_{family}_dispersion"
                if source in pivot.columns:
                    row_profile[name] = row_tm.map(pivot[source])
                else:
                    row_profile[name] = np.nan
            rel_source = ("tm_group_reliability", group)
            rel_name = f"tm_{group}_reliability"
            row_profile[rel_name] = (
                row_tm.map(pivot[rel_source]) if rel_source in pivot.columns else np.nan
            )

        row_rates = pd.DataFrame(
            {
                "fastball": main.features.loc[row_idx, "asof_pitcher_fastball_rate"],
                "breaking": main.features.loc[row_idx, "asof_pitcher_breaking_rate"],
                "offspeed": main.features.loc[row_idx, "asof_pitcher_offspeed_rate"],
            },
            index=row_idx,
        )
        for family in ("release", "velocity", "movement", "spin"):
            group_columns = {group: f"tm_{group}_{family}_dispersion" for group in PITCH_GROUPS}
            expected_name = f"tm_expected_{family}_dispersion"
            features.loc[row_idx, expected_name] = _weighted_expected(
                row_rates, row_profile, group_columns
            )
            if expected_name not in blocks["trackman_repeatability"]:
                blocks["trackman_repeatability"].append(expected_name)

        rel_columns = {group: f"tm_{group}_reliability" for group in PITCH_GROUPS}
        features.loc[row_idx, "tm_expected_reliability"] = _weighted_expected(
            row_rates, row_profile, rel_columns
        )
        if "tm_expected_reliability" not in blocks["trackman_repeatability"]:
            blocks["trackman_repeatability"].append("tm_expected_reliability")

        # Context-conditioned pitch-group probabilities with shrinkage to pitcher overall usage.
        historical_context = context_stats.loc[
            pd.to_numeric(context_stats["season"], errors="coerce") < season
        ].copy()
        if not historical_context.empty:
            hist_counts = (
                historical_context.groupby(
                    [
                        "pitcher_trackman_id",
                        "balls_before",
                        "strikes_before",
                        "batter_hand",
                        "pitch_type_group",
                    ],
                    observed=True,
                    dropna=False,
                    sort=False,
                )["context_n"]
                .sum()
                .reset_index()
            )
            overall_counts = (
                historical_group.groupby(
                    ["pitcher_trackman_id", "pitch_type_group"], observed=True, sort=False
                )["tm_group_pitch_n"]
                .sum()
                .reset_index()
            )
            overall_pivot = overall_counts.pivot(
                index="pitcher_trackman_id", columns="pitch_type_group", values="tm_group_pitch_n"
            ).fillna(0.0)
            overall_total = overall_pivot.sum(axis=1).replace(0, np.nan)
            overall_rates = overall_pivot.div(overall_total, axis=0)

            ctx_pivot = hist_counts.pivot_table(
                index=["pitcher_trackman_id", "balls_before", "strikes_before", "batter_hand"],
                columns="pitch_type_group",
                values="context_n",
                aggfunc="sum",
                fill_value=0.0,
            )
            context_keys = pd.MultiIndex.from_arrays(
                [
                    row_tm.astype("string"),
                    pd.to_numeric(balls.loc[row_idx], errors="coerce"),
                    pd.to_numeric(strikes.loc[row_idx], errors="coerce"),
                    batter_hand.loc[row_idx].astype("string"),
                ],
                names=["pitcher_trackman_id", "balls_before", "strikes_before", "batter_hand"],
            )
            matched = ctx_pivot.reindex(context_keys).fillna(0.0)
            matched.index = row_idx
            ctx_total = matched.sum(axis=1)
            smoothed_rates = pd.DataFrame(index=row_idx)
            for group in PITCH_GROUPS:
                ctx_count = matched[group] if group in matched.columns else pd.Series(0.0, index=row_idx)
                prior_rate = (
                    row_tm.map(overall_rates[group])
                    if group in overall_rates.columns
                    else pd.Series(0.0, index=row_idx)
                )
                smoothed_rates[group] = (ctx_count + context_smoothing * prior_rate.fillna(0.0)) / (
                    ctx_total + context_smoothing
                )
            features.loc[row_idx, "tm_context_pitchmix_n_log1p"] = np.log1p(ctx_total)
            features.loc[row_idx, "tm_context_reliability"] = ctx_total / (ctx_total + context_smoothing)
            for family in ("release", "movement"):
                group_columns = {group: f"tm_{group}_{family}_dispersion" for group in PITCH_GROUPS}
                name = f"tm_context_expected_{family}_dispersion"
                features.loc[row_idx, name] = _weighted_expected(
                    smoothed_rates, row_profile, group_columns
                )
            for name in (
                "tm_context_pitchmix_n_log1p",
                "tm_context_reliability",
                "tm_context_expected_release_dispersion",
                "tm_context_expected_movement_dispersion",
            ):
                if name not in blocks["trackman_context"]:
                    blocks["trackman_context"].append(name)

        # Historical fatigue/drift susceptibility, averaged over prior games only.
        historical_slopes = game_slopes.loc[
            pd.to_numeric(game_slopes["season"], errors="coerce") < season
        ].copy()
        if not historical_slopes.empty:
            pitcher_slopes = historical_slopes.groupby("pitcher_trackman_id", observed=True).agg(
                tm_velocity_decay_slope=("rel_speed_slope", "mean"),
                tm_rel_height_abs_slope=("rel_height_slope", lambda x: x.abs().mean()),
                tm_rel_side_abs_slope=("rel_side_slope", lambda x: x.abs().mean()),
                tm_extension_abs_slope=("extension_slope", lambda x: x.abs().mean()),
                tm_drift_game_n=("trackman_game_id", "nunique"),
            )
            pitcher_slopes["tm_release_drift_abs_slope"] = pitcher_slopes[
                ["tm_rel_height_abs_slope", "tm_rel_side_abs_slope", "tm_extension_abs_slope"]
            ].mean(axis=1, skipna=True)
            features.loc[row_idx, "tm_velocity_decay_slope"] = row_tm.map(
                pitcher_slopes["tm_velocity_decay_slope"]
            )
            features.loc[row_idx, "tm_release_drift_abs_slope"] = row_tm.map(
                pitcher_slopes["tm_release_drift_abs_slope"]
            )
            features.loc[row_idx, "tm_drift_game_n_log1p"] = np.log1p(
                row_tm.map(pitcher_slopes["tm_drift_game_n"])
            )
            for name in (
                "tm_velocity_decay_slope",
                "tm_release_drift_abs_slope",
                "tm_drift_game_n_log1p",
            ):
                if name not in blocks["trackman_drift"]:
                    blocks["trackman_drift"].append(name)

    for column in features.columns:
        if column not in categorical:
            features[column] = _safe_numeric(features[column]).astype("float32")
        else:
            features[column] = features[column].astype("string").fillna("__MISSING__")
    return features, blocks, categorical


def merge_trackman(
    main: EngineeredData,
    trackman: pd.DataFrame,
    blocks: dict[str, list[str]],
    categorical: list[str],
) -> EngineeredData:
    features = pd.concat([main.features, trackman], axis=1)
    merged_blocks = {name: list(cols) for name, cols in main.blocks.items()}
    for name, cols in blocks.items():
        merged_blocks[name] = list(cols)
    return EngineeredData(
        features=features,
        target=main.target,
        row_ids=main.row_ids,
        seasons=main.seasons,
        blocks=merged_blocks,
        categorical_features=[*main.categorical_features, *categorical],
    )

