from __future__ import annotations

import numpy as np
import pandas as pd

from .common import EngineeredData, TARGET, _safe_numeric, _to_probability


def leakage_safe_season_trend_prior(
    train: pd.DataFrame,
    seasons: pd.Series,
    *,
    fallback: float = 0.5,
    min_history_seasons: int = 4,
) -> pd.DataFrame:
    """Build a leakage-safe season-level target trend prior.

    For a row in season S, only target observations from seasons < S are used.
    With fewer than ``min_history_seasons`` completed seasons, the prior falls
    back to the row-count-weighted historical mean. Once enough seasons exist,
    an ordinary least-squares linear trend is fitted to completed season rates
    and extrapolated to S. The extrapolated probability is clipped to [0.01, 0.99].

    Returned columns are constant within a target season:
    - season_trend_prior: extrapolated (or fallback) success-rate prior
    - season_trend_slope: fitted per-season rate slope; 0 before trend fitting
    - season_trend_history_n: number of completed seasons used
    """
    if TARGET not in train.columns or "season" not in train.columns:
        raise ValueError(f"train must contain 'season' and '{TARGET}'")

    source = pd.DataFrame(
        {
            "season": pd.to_numeric(train["season"], errors="raise").astype(int),
            "target": pd.to_numeric(train[TARGET], errors="raise").astype("float64"),
        }
    )
    season_stats = (
        source.groupby("season", observed=True)["target"]
        .agg(["mean", "count"])
        .sort_index()
    )

    requested = pd.to_numeric(seasons, errors="coerce")
    rows: dict[int, tuple[float, float, int]] = {}
    for season in sorted(requested.dropna().astype(int).unique()):
        history = season_stats.loc[season_stats.index < season]
        history_n = int(len(history))
        if history_n == 0:
            prior = float(fallback)
            slope = 0.0
        elif history_n < min_history_seasons:
            total = float(history["count"].sum())
            if total > 0:
                prior = float((history["mean"] * history["count"]).sum() / total)
            else:
                prior = float(fallback)
            slope = 0.0
        else:
            x = history.index.to_numpy(dtype="float64")
            y = history["mean"].to_numpy(dtype="float64")
            slope, intercept = np.polyfit(x, y, deg=1)
            prior = float(intercept + slope * float(season))
            slope = float(slope)
        rows[season] = (float(np.clip(prior, 0.01, 0.99)), slope, history_n)

    prior_map = {season: values[0] for season, values in rows.items()}
    slope_map = {season: values[1] for season, values in rows.items()}
    history_map = {season: values[2] for season, values in rows.items()}
    return pd.DataFrame(
        {
            "season_trend_prior": requested.map(prior_map).astype("float64"),
            "season_trend_slope": requested.map(slope_map).astype("float64"),
            "season_trend_history_n": requested.map(history_map).astype("float64"),
        },
        index=seasons.index,
    )


def _shrink_with_prior(
    rate: pd.Series,
    n: pd.Series,
    prior: pd.Series,
    strength: float,
) -> pd.Series:
    rate_v = _to_probability(rate)
    n_v = _safe_numeric(n).clip(lower=0)
    prior_v = _to_probability(prior)
    return ((rate_v * n_v) + (prior_v * strength)) / (n_v + strength)


def add_season_trend_features(
    engineered: EngineeredData,
    raw_train: pd.DataFrame,
    *,
    shrinkage: float = 50.0,
    fallback: float = 0.5,
    min_history_seasons: int = 4,
) -> EngineeredData:
    """Attach the season trend prior and use it for pitcher/batter shrinkage.

    ``build_main_features`` already creates leakage-safe row-wise features. This
    function replaces only the prior inside the two shrunk success-rate features
    and adds the season-trend block; all other features and join attrs are kept.
    """
    trend = leakage_safe_season_trend_prior(
        raw_train,
        engineered.seasons,
        fallback=fallback,
        min_history_seasons=min_history_seasons,
    )

    features = engineered.features.copy()
    attrs = dict(engineered.features.attrs)
    for column in trend.columns:
        features[column] = trend[column].astype("float32")

    features["pitcher_success_shrunk"] = _shrink_with_prior(
        raw_train["asof_pitcher_success_rate"],
        raw_train["asof_pitcher_n"],
        trend["season_trend_prior"],
        shrinkage,
    ).astype("float32")
    features["batter_success_shrunk"] = _shrink_with_prior(
        raw_train["asof_batter_success_rate"],
        raw_train["asof_batter_n"],
        trend["season_trend_prior"],
        shrinkage,
    ).astype("float32")
    features.attrs = attrs

    blocks = {name: list(columns) for name, columns in engineered.blocks.items()}
    blocks["season_trend"] = list(trend.columns)

    return EngineeredData(
        features=features,
        target=engineered.target,
        row_ids=engineered.row_ids,
        seasons=engineered.seasons,
        blocks=blocks,
        categorical_features=list(engineered.categorical_features),
    )
