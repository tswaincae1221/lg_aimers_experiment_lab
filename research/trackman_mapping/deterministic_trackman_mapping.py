"""Deterministic train <-> Trackman pitcher crosswalk via game fingerprints.

The target is never read.  Games are reconstructed from contiguous train row order,
matched to Trackman games by matchup/date-part and the ordered pre-pitch state
sequence, and pitcher IDs are then voted from aligned pitch positions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TEAM = {
    "DOO_BEA": 12,
    "HAN_EAG": 17,
    "KIA_TIG": 16,
    "KIW_HER": 14,
    "KT_WIZ": 20,
    "LG_TWI": 13,
    "LOT_GIA": 15,
    "NC_DIN": 19,
    "SAM_LIO": 18,
    "SSG_LAN": 21,
    "SK_WYV": 21,
}

STATE = [
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
]


def seq_hash(frame: pd.DataFrame) -> str:
    values = frame[STATE].astype(str).agg("|".join, axis=1).str.cat(sep=";")
    return hashlib.blake2b(values.encode(), digest_size=16).hexdigest()


def load_train() -> pd.DataFrame:
    cols = [
        "row_id",
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
        "top_bottom",
        "balls_before",
        "strikes_before",
        "outs_before",
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
    ]
    tr = pd.read_csv(DATA / "train.csv", usecols=cols)
    tr["pitcher_hand"] = tr["pitcher_hand"].map({2: "R", 1: "L"})
    tr["batter_hand"] = tr["batter_hand"].map({2: "R", 1: "L"})
    return tr


def assign_train_games(tr: pd.DataFrame) -> pd.DataFrame:
    p = tr.shift()
    team_lo = tr[["pitcher_team_id", "batter_team_id"]].min(axis=1)
    team_hi = tr[["pitcher_team_id", "batter_team_id"]].max(axis=1)
    flipped_same_inning = (
        tr["inning"].eq(p["inning"]) & p["top_bottom"].eq("B") & tr["top_bottom"].eq("T")
    )
    boundary = (
        tr["season"].ne(p["season"])
        | tr["game_month"].ne(p["game_month"])
        | tr["game_dayofweek"].ne(p["game_dayofweek"])
        | team_lo.ne(team_lo.shift())
        | team_hi.ne(team_hi.shift())
        | tr["inning"].lt(p["inning"])
        | flipped_same_inning
    )
    boundary.iloc[0] = True
    tr = tr.copy()
    tr["train_game_seq"] = boundary.cumsum().astype("int32")
    tr["pitch_no_reconstructed"] = tr.groupby("train_game_seq").cumcount().add(1).astype("int16")
    return tr


def load_trackman() -> pd.DataFrame:
    cols = [
        "trackman_id",
        "trackman_game_id",
        "game_date",
        "pitch_no",
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
        "top_bottom",
        "balls_before",
        "strikes_before",
        "outs_before",
        "pitcher_team",
        "batter_team",
        "pitcher_trackman_id",
        "batter_trackman_id",
        "pitcher_hand",
        "batter_hand",
    ]
    tm = pd.read_csv(DATA / "trackman_history.csv", usecols=cols)
    tm["pitcher_team_id"] = tm["pitcher_team"].map(TEAM)
    tm["batter_team_id"] = tm["batter_team"].map(TEAM)
    tm["top_bottom"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B"})
    tm["pitcher_hand"] = tm["pitcher_hand"].map({"Right": "R", "Left": "L"})
    tm["batter_hand"] = tm["batter_hand"].map({"Right": "R", "Left": "L"})
    return tm.dropna(subset=["pitcher_team_id", "batter_team_id"]).copy()


def game_catalog(df: pd.DataFrame, game_col: str, order_col: str) -> pd.DataFrame:
    rows = []
    for gid, g in df.sort_values([game_col, order_col], kind="stable").groupby(
        game_col, sort=False
    ):
        if not {"T", "B"}.issubset(set(g["top_bottom"].dropna())):
            continue
        first = g.iloc[0]
        home = int(g.loc[g["top_bottom"].eq("T"), "pitcher_team_id"].mode().iloc[0])
        away = int(g.loc[g["top_bottom"].eq("B"), "pitcher_team_id"].mode().iloc[0])
        rows.append(
            {
                game_col: gid,
                "season": int(first["season"]),
                "game_month": int(first["game_month"]),
                "game_dayofweek": int(first["game_dayofweek"]),
                "home_team_id": home,
                "away_team_id": away,
                "n_pitches": len(g),
                "state_hash": seq_hash(g),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    tr = assign_train_games(load_train())
    tm = load_trackman()
    tc = game_catalog(tr, "train_game_seq", "pitch_no_reconstructed")
    mc = game_catalog(tm, "trackman_game_id", "pitch_no")
    keys = [
        "season",
        "game_month",
        "game_dayofweek",
        "home_team_id",
        "away_team_id",
        "n_pitches",
        "state_hash",
    ]
    tc_unique = tc[tc.groupby(keys)["train_game_seq"].transform("size").eq(1)]
    mc_unique = mc[mc.groupby(keys)["trackman_game_id"].transform("size").eq(1)]
    pairs = tc_unique.merge(mc_unique, on=keys, how="inner", validate="one_to_one")
    print(f"train games={len(tc):,}, trackman games={len(mc):,}, exact matched={len(pairs):,}")

    aligned = tr.merge(pairs[["train_game_seq", "trackman_game_id"]], on="train_game_seq")
    aligned = aligned.merge(
        tm[["trackman_game_id", "pitch_no", "pitcher_trackman_id", "batter_trackman_id"]],
        left_on=["trackman_game_id", "pitch_no_reconstructed"],
        right_on=["trackman_game_id", "pitch_no"],
        validate="one_to_one",
    )
    votes = (
        aligned.groupby(["pitcher_trackman_id", "pitcher_id"]).size().rename("votes").reset_index()
    )
    votes["total_votes"] = votes.groupby("pitcher_trackman_id")["votes"].transform("sum")
    votes["purity"] = votes["votes"] / votes["total_votes"]
    best = (
        votes.sort_values(["pitcher_trackman_id", "votes", "pitcher_id"], kind="stable")
        .groupby("pitcher_trackman_id")
        .tail(1)
    )
    best = best.sort_values("pitcher_trackman_id").reset_index(drop=True)
    best.to_csv(OUT / "pitcher_crosswalk_deterministic.csv", index=False)
    pairs.to_csv(OUT / "game_crosswalk_deterministic.csv", index=False)
    aligned.head(500).to_csv(OUT / "train_trackman_exact_preview.csv", index=False)
    print(
        f"pitchers mapped={len(best):,}, mean purity={best.purity.mean():.6f}, "
        f"min purity={best.purity.min():.6f}"
    )
    print(best.purity.describe().to_string())
    print("outputs:", OUT)


if __name__ == "__main__":
    main()
