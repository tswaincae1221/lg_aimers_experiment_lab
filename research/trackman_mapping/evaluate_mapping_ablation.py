"""Controlled 2024 holdout A/B test: legacy vs deterministic Trackman mapping."""

from pathlib import Path
import functools

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "outputs"
TEAM = {
    "DOO_BEA": 12, "HAN_EAG": 17, "KIA_TIG": 16, "KIW_HER": 14,
    "KT_WIZ": 20, "LG_TWI": 13, "LOT_GIA": 15, "NC_DIN": 19,
    "SAM_LIO": 18, "SSG_LAN": 21, "SK_WYV": 21,
}
KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before"]


def make_key(df, cols):
    return functools.reduce(lambda a, b: a + "|" + b, [df[c].astype(str) for c in cols])


def legacy_crosswalk(tm, tr):
    t = tm[tm["season"] < 2024].copy()
    r = tr[tr["season"] < 2024].copy()
    t["pitcher_team_id"] = t["pitcher_team"].map(TEAM)
    t["batter_team_id"] = t["batter_team"].map(TEAM)
    t = t.dropna(subset=["pitcher_team_id", "batter_team_id"])
    t["pitcher_team_id"] = t["pitcher_team_id"].astype("int16")
    t["batter_team_id"] = t["batter_team_id"].astype("int16")
    t["top_bottom"] = t["top_bottom"].map({"Top": "T", "Bottom": "B"})
    t["pitcher_hand"] = t["pitcher_hand"].map({"Right": "R", "Left": "L"})
    t["batter_hand"] = t["batter_hand"].map({"Right": "R", "Left": "L"})
    r["pitcher_hand"] = r["pitcher_hand"].map({2: "R", 1: "L"})
    r["batter_hand"] = r["batter_hand"].map({2: "R", 1: "L"})
    cols = KEY + ["pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]
    tk, rk = make_key(t, cols), make_key(r, cols)
    tm_one, tr_one = tk.groupby(tk).transform("size").eq(1), rk.groupby(rk).transform("size").eq(1)
    a = pd.DataFrame({"K": tk[tm_one], "pitcher_trackman_id": t.loc[tm_one, "pitcher_trackman_id"]})
    b = pd.DataFrame({"K": rk[tr_one], "pitcher_id": r.loc[tr_one, "pitcher_id"]})
    seed = b.merge(a, on="K")
    v = seed.groupby(["pitcher_trackman_id", "pitcher_id"]).size().rename("votes").reset_index()
    v["total_votes"] = v.groupby("pitcher_trackman_id")["votes"].transform("sum")
    v["purity"] = v["votes"] / v["total_votes"]
    return v.sort_values(["pitcher_trackman_id", "votes", "pitcher_id"]).groupby("pitcher_trackman_id").tail(1)


def trackman_wide(tm, xw):
    mapper = xw.set_index("pitcher_trackman_id")["pitcher_id"]
    pur = xw.set_index("pitcher_trackman_id")["purity"]
    t = tm.copy()
    t["pitcher_id"] = t["pitcher_trackman_id"].map(mapper)
    t["purity"] = t["pitcher_trackman_id"].map(pur)
    t = t.dropna(subset=["pitcher_id"])
    t["pitcher_id"] = t["pitcher_id"].astype("int64")
    t = t[t["pitch_type_group"].isin(["fastball", "breaking", "offspeed"])]
    g = t.groupby(["pitcher_id", "season", "pitch_type_group"], observed=True)
    z = g.size().rename("n").to_frame()
    z["spin_std"] = g["spin_rate"].std()
    z["purity"] = g["purity"].first()
    z = z.reset_index()
    z.loc[z["n"] < 50, "spin_std"] = np.nan
    peer = z.groupby(["season", "pitch_type_group"], observed=True)
    raw = (1 - peer["spin_std"].rank(pct=True)) * 100
    z["score"] = 50 + (raw - 50) * z["n"] / (z["n"] + 100)
    z["w"] = z["n"] / z.groupby(["pitcher_id", "season"])["n"].transform("sum")
    z["wx"] = z["score"] * z["w"]
    wide = z.groupby(["pitcher_id", "season"]).agg(
        spin_stability_score=("wx", "sum"), tm_n=("n", "sum"),
        tm_purity=("purity", "first"),
    ).reset_index()
    return wide


def attach(df, wide, suffix):
    q = df[["pitcher_id", "season"]].copy()
    q["query_season"] = q.pop("season").astype(float)
    q["_order"] = np.arange(len(q))
    w = wide.copy(); w["season"] = w["season"].astype(float)
    m = pd.merge_asof(q.sort_values("query_season"), w.sort_values("season"),
                      left_on="query_season", right_on="season", by="pitcher_id",
                      direction="backward", allow_exact_matches=False).sort_values("_order")
    score = m["spin_stability_score"].fillna(-1).to_numpy(np.float32)
    grade = np.where(m["tm_n"].isna(), "D", np.where(m["tm_purity"] >= .9, "A", "B"))
    return pd.DataFrame({f"spin_{suffix}": score, f"grade_{suffix}": grade})


def metrics(y, p):
    b = brier_score_loss(y, p); r = y.mean(); bss = 100000 * (1 - b / (r * (1-r)))
    return {"brier": b, "bss": bss, "auc": roc_auc_score(y, p), "pred_mean": p.mean()}


def main():
    features = ["season", "game_month", "inning", "balls_before", "strikes_before",
                "outs_before", "li", "pitcher_hand", "batter_hand", "pitcher_team_id",
                "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
                "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
                "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate"]
    tr = pd.read_csv(DATA / "train.csv", usecols=list(dict.fromkeys(features + KEY + ["pitcher_id", "control_success"])))
    tm_cols = list(dict.fromkeys(KEY + ["pitcher_team", "batter_team", "pitcher_hand", "batter_hand",
                                      "pitcher_trackman_id", "pitch_type_group", "spin_rate"]))
    tm = pd.read_csv(DATA / "trackman_history.csv", usecols=tm_cols)
    old = legacy_crosswalk(tm, tr)
    new = pd.read_csv(OUT / "pitcher_crosswalk_deterministic_pre2024.csv")
    old.to_csv(OUT / "pitcher_crosswalk_legacy_pre2024.csv", index=False)
    print(f"legacy pitchers={len(old)}, purity={old.purity.mean():.6f}; deterministic={len(new)}, purity={new.purity.mean():.6f}")
    fo, fn = attach(tr, trackman_wide(tm, old), "old"), attach(tr, trackman_wide(tm, new), "new")
    base = tr[features].copy()
    cats = ["pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]
    for c in cats: base[c] = base[c].fillna(-1).astype(str)
    base[[c for c in features if c not in cats]] = base[[c for c in features if c not in cats]].fillna(0.49).astype("float32")
    y = tr["control_success"].to_numpy()
    dev, val = tr.season.lt(2024).to_numpy(), tr.season.eq(2024).to_numpy()
    results = []
    for name, extra in [("legacy", fo), ("deterministic", fn)]:
        X = base.copy(); X["spin_stability_score"] = extra.iloc[:, 0].to_numpy(); X["data_confidence_grade"] = extra.iloc[:, 1].to_numpy()
        model = CatBoostClassifier(iterations=250, depth=7, learning_rate=.05, l2_leaf_reg=8,
            random_seed=42, loss_function="Logloss", verbose=50, allow_writing_files=False,
            thread_count=-1)
        model.fit(X.loc[dev], y[dev], cat_features=cats + ["data_confidence_grade"])
        p = model.predict_proba(X.loc[val])[:, 1]
        row = {"mapping": name, **metrics(y[val], p)}; results.append(row); print(row)
    out = pd.DataFrame(results)
    out.to_csv(OUT / "mapping_ablation_2024.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
