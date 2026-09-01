"""Derive the 2024-backtest crosswalk using matched games through 2023 only."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deterministic_trackman_mapping as dm


def main():
    pairs = pd.read_csv(dm.OUT / "game_crosswalk_deterministic.csv")
    pairs = pairs[pairs["season"] < 2024]
    tr = dm.assign_train_games(dm.load_train())
    tm = dm.load_trackman()
    aligned = tr.merge(pairs[["train_game_seq", "trackman_game_id"]], on="train_game_seq")
    aligned = aligned.merge(
        tm[["trackman_game_id", "pitch_no", "pitcher_trackman_id"]],
        left_on=["trackman_game_id", "pitch_no_reconstructed"],
        right_on=["trackman_game_id", "pitch_no"],
        validate="one_to_one",
    )
    v = aligned.groupby(["pitcher_trackman_id", "pitcher_id"]).size().rename("votes").reset_index()
    v["total_votes"] = v.groupby("pitcher_trackman_id")["votes"].transform("sum")
    v["purity"] = v["votes"] / v["total_votes"]
    best = (
        v.sort_values(["pitcher_trackman_id", "votes", "pitcher_id"], kind="stable")
        .groupby("pitcher_trackman_id")
        .tail(1)
    )
    best.to_csv(dm.OUT / "pitcher_crosswalk_deterministic_pre2024.csv", index=False)
    print(
        f"pre-2024 matched games={len(pairs):,}; "
        f"pitchers={len(best):,}; mean purity={best.purity.mean():.6f}"
    )


if __name__ == "__main__":
    main()
