"""Build a leakage-safe train + Trackman feature overlay from deterministic mapping."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "outputs"
METRICS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]


def main():
    xw = pd.read_csv(OUT / "pitcher_crosswalk_deterministic.csv")
    mapper = xw.set_index("pitcher_trackman_id")["pitcher_id"]
    purity = xw.set_index("pitcher_trackman_id")["purity"]
    cols = ["pitcher_trackman_id", "season", "pitch_type_group"] + METRICS
    tm = pd.read_csv(DATA / "trackman_history.csv", usecols=cols)
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(mapper)
    tm["tm_mapping_purity"] = tm["pitcher_trackman_id"].map(purity)
    tm = tm.dropna(subset=["pitcher_id"]).copy()
    tm["pitcher_id"] = tm["pitcher_id"].astype("int64")

    g = tm.groupby(["pitcher_id", "season"], sort=False)
    wide = g.size().rename("tm_n").to_frame()
    wide["tm_mapping_purity"] = g["tm_mapping_purity"].first()
    for c in METRICS:
        wide[f"tm_{c}_mean"] = g[c].mean()
        wide[f"tm_{c}_std"] = g[c].std()
    mix = (
        tm[tm["pitch_type_group"].isin(["fastball", "breaking", "offspeed"])]
        .groupby(["pitcher_id", "season"])["pitch_type_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns=lambda c: f"tm_mix_{c}")
    )
    wide = wide.join(mix, how="left").reset_index()
    wide.to_csv(OUT / "trackman_pitcher_season_features.csv", index=False)

    train_cols = ["row_id", "pitcher_id", "season", "control_success"]
    tr = pd.read_csv(DATA / "train.csv", usecols=train_cols)
    q = tr[["pitcher_id", "season"]].copy()
    q["query_season"] = q.pop("season").astype("float64")
    q["_order"] = np.arange(len(q))
    w = wide.copy()
    w["tm_source_season"] = w.pop("season").astype("float64")
    joined = (
        pd.merge_asof(
            q.sort_values("query_season", kind="stable"),
            w.sort_values("tm_source_season", kind="stable"),
            left_on="query_season",
            right_on="tm_source_season",
            by="pitcher_id",
            direction="backward",
            allow_exact_matches=False,
        )
        .sort_values("_order", kind="stable")
        .reset_index(drop=True)
    )
    feature_cols = [c for c in joined if c.startswith("tm_")]
    overlay = tr.copy()
    overlay[feature_cols] = joined[feature_cols]
    overlay.to_csv(OUT / "train_trackman_overlay.csv.gz", index=False, compression="gzip")
    overlay.head(200).to_csv(OUT / "train_trackman_overlay_preview.csv", index=False)
    coverage = overlay.groupby("season")["tm_source_season"].apply(lambda s: s.notna().mean())
    print(f"mapped Trackman rows={len(tm):,}/{1_793_078:,} ({len(tm) / 1_793_078:.1%})")
    print(f"pitcher-season table={wide.shape}; overlay={overlay.shape}")
    print("strict-prior coverage by train season:\n" + coverage.to_string())
    print(overlay.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
