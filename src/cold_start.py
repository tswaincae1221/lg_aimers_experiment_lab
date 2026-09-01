"""Leakage-safe cold-start priors for pitcher command models.

The lookup produced here is static at inference time.  A row in season Y may
only use Trackman profiles and outcomes from Y-1.  Test rows are never compared
with, or aggregated with, other test rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIOR_COLUMNS = (
    "cold_start_prior",
    "cold_start_source",
    "cold_start_neighbor_distance",
    "cold_start_neighbor_count",
)


@dataclass(frozen=True)
class ColdStartConfig:
    neighbors: int = 20
    min_profile_pitches: int = 50
    distance_floor: float = 1e-6
    rookie_min_pitchers: int = 3


def _pitcher_season_outcomes(train: pd.DataFrame) -> pd.DataFrame:
    """One equally weighted outcome per pitcher-season."""
    required = {"pitcher_id", "season", "control_success", "asof_pitcher_n"}
    missing = required.difference(train.columns)
    if missing:
        raise KeyError(f"train columns missing: {sorted(missing)}")

    work = train.loc[:, list(required)].copy()
    work["control_success"] = pd.to_numeric(work["control_success"], errors="coerce")
    out = work.groupby(["pitcher_id", "season"], as_index=False, sort=False).agg(
        season_success_rate=("control_success", "mean"),
        season_pitches=("control_success", "count"),
        min_asof_n=("asof_pitcher_n", "min"),
    )
    # A pitcher is a true observed rookie only when an official pre-pitch
    # career count of zero is actually present.  Merely first appearing in a
    # truncated dataset is not enough.
    out["is_rookie"] = out["min_asof_n"].fillna(np.inf).eq(0)
    return out


def _season_fallbacks(outcomes: pd.DataFrame, config: ColdStartConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for season, part in outcomes.groupby("season", sort=True):
        valid = part.loc[part["season_pitches"].gt(0), "season_success_rate"]
        rookies = part.loc[
            part["is_rookie"] & part["season_pitches"].gt(0),
            "season_success_rate",
        ]
        league = float(valid.median())
        if len(rookies) >= config.rookie_min_pitchers:
            rookie = float(rookies.median())
        else:
            rookie = league
        rows.append(
            {
                "season": int(season) + 1,
                "rookie_prior": rookie,
                "league_prior": league,
                "rookie_pitchers": int(len(rookies)),
            }
        )
    return pd.DataFrame(rows)


def build_cold_start_lookup(
    train: pd.DataFrame,
    trackman_profiles: pd.DataFrame,
    *,
    profile_columns: list[str] | tuple[str, ...] | None = None,
    config: ColdStartConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build `(pitcher_id, query season)` KNN priors and seasonal fallbacks.

    `trackman_profiles` must contain one row per pitcher-season.  Numeric
    Trackman columns are robustly standardized within each source season.  The
    target pitcher's source-season profile is compared only with other pitchers
    from that same season, whose same-season official success rate is already
    known by query season Y.
    """
    config = config or ColdStartConfig()
    required = {"pitcher_id", "season"}
    missing = required.difference(trackman_profiles.columns)
    if missing:
        raise KeyError(f"trackman profile columns missing: {sorted(missing)}")

    profiles = trackman_profiles.copy()
    if profile_columns is None:
        blocked = {"pitcher_id", "season", "tm_source_season", "tm_mapping_purity"}
        profile_columns = [
            c
            for c in profiles.columns
            if c not in blocked and pd.api.types.is_numeric_dtype(profiles[c]) and c != "tm_n"
        ]
    profile_columns = list(profile_columns)
    if not profile_columns:
        raise ValueError("at least one numeric Trackman profile column is required")

    if "tm_n" in profiles:
        profiles = profiles.loc[
            pd.to_numeric(profiles["tm_n"], errors="coerce").ge(config.min_profile_pitches)
        ].copy()
    profiles = profiles.drop_duplicates(["pitcher_id", "season"], keep="last")

    outcomes = _pitcher_season_outcomes(train)
    fallbacks = _season_fallbacks(outcomes, config)
    candidates = profiles.merge(
        outcomes[["pitcher_id", "season", "season_success_rate", "season_pitches"]],
        on=["pitcher_id", "season"],
        how="inner",
        validate="one_to_one",
    )

    rows: list[dict[str, float | int | str]] = []
    for source_season, part in candidates.groupby("season", sort=True):
        part = part.reset_index(drop=True)
        values = part[profile_columns].apply(pd.to_numeric, errors="coerce")
        median = values.median(axis=0)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        scale = (q3 - q1).replace(0, np.nan).fillna(1.0)
        z = values.fillna(median).sub(median).div(scale).to_numpy(dtype=float)
        rates = part["season_success_rate"].to_numpy(dtype=float)
        ids = part["pitcher_id"].to_numpy()

        for i, pitcher_id in enumerate(ids):
            distance = np.sqrt(np.mean((z - z[i]) ** 2, axis=1))
            eligible = np.isfinite(distance) & np.isfinite(rates)
            eligible[i] = False  # never copy the pitcher's own outcome
            idx = np.flatnonzero(eligible)
            if idx.size == 0:
                continue
            idx = idx[np.argsort(distance[idx], kind="stable")[: config.neighbors]]
            weights = 1.0 / np.maximum(distance[idx], config.distance_floor)
            prior = float(np.average(rates[idx], weights=weights))
            rows.append(
                {
                    "pitcher_id": pitcher_id,
                    "season": int(source_season) + 1,
                    "cold_start_prior": prior,
                    "cold_start_source": "trackman_knn",
                    "cold_start_neighbor_distance": float(
                        np.average(distance[idx], weights=weights)
                    ),
                    "cold_start_neighbor_count": int(len(idx)),
                }
            )

    lookup = pd.DataFrame(rows)
    if lookup.empty:
        lookup = pd.DataFrame(columns=["pitcher_id", "season", *PRIOR_COLUMNS])
    else:
        lookup = lookup.sort_values(["season", "pitcher_id"], kind="stable")
        lookup = lookup.reset_index(drop=True)
    return lookup, fallbacks


def attach_cold_start_prior(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    fallbacks: pd.DataFrame,
    *,
    replace_success_rate: bool = True,
) -> pd.DataFrame:
    """Attach priors row-independently and optionally replace only missing rates."""
    required = {"pitcher_id", "season", "asof_pitcher_n", "asof_pitcher_success_rate"}
    missing = required.difference(rows.columns)
    if missing:
        raise KeyError(f"row columns missing: {sorted(missing)}")

    out = rows.copy()
    keys = out[["pitcher_id", "season"]].copy()
    keys["_row_order"] = np.arange(len(out))
    attached = keys.merge(
        lookup,
        on=["pitcher_id", "season"],
        how="left",
        validate="many_to_one",
    ).merge(fallbacks, on="season", how="left", validate="many_to_one")
    attached = attached.sort_values("_row_order", kind="stable")

    rate = pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
    count = pd.to_numeric(out["asof_pitcher_n"], errors="coerce")
    cold = count.fillna(0).le(0) | rate.isna()
    knn = pd.to_numeric(attached["cold_start_prior"], errors="coerce")
    rookie = pd.to_numeric(attached["rookie_prior"], errors="coerce")
    league = pd.to_numeric(attached["league_prior"], errors="coerce")
    prior = knn.fillna(rookie).fillna(league).fillna(0.49)
    source = np.select(
        [knn.notna(), rookie.notna(), league.notna()],
        ["trackman_knn", "rookie_median", "league_median"],
        default="fixed_0.49",
    )

    out["cold_start_prior"] = prior.to_numpy(dtype="float32")
    out["cold_start_source"] = np.where(cold, source, "self_history")
    out["cold_start_neighbor_distance"] = pd.to_numeric(
        attached["cold_start_neighbor_distance"], errors="coerce"
    ).to_numpy(dtype="float32")
    out["cold_start_neighbor_count"] = (
        pd.to_numeric(attached["cold_start_neighbor_count"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="int16")
    )
    out["is_cold_start"] = cold.to_numpy(dtype="int8")
    if replace_success_rate:
        out.loc[cold, "asof_pitcher_success_rate"] = prior.loc[cold].to_numpy()
    return out
