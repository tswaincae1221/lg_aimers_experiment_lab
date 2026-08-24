# Pitcher Bayesian Shrinkage (C) Experiment

## Goal

Test whether a pitcher's raw pre-pitch control success rate should be trusted directly, or shrunk toward a leakage-safe league/season prior when the pitcher has limited history.

The experiment is stacked on the fixed-350 Top20 HGB setup from the HGB feature-selection V2 work. The original Top20 experiment is not modified.

## Hypothesis

For pitcher `i` at a given pre-pitch row,

```text
p_shrunk = (n * p_raw + m * p_prior) / (n + m)
```

where:

- `p_raw` = official `asof_pitcher_success_rate`
- `n` = official `asof_pitcher_n`
- `m` = prior strength (pseudo-pitch count)
- `p_prior` = leakage-safe season trend prior

For a row in season `S`, the prior uses only labeled seasons `< S`. Same-season and future target labels are excluded.

## Validation contract

- Training rows: `season < 2024` (2019-2023)
- Validation rows: `season == 2024`
- Full HGB iterations: 350 fixed
- Primary metric: Brier Score (lower is better)
- Secondary metrics: Brier Skill Score and AUC

## Compared variants

1. `raw_top20`: fixed Top20 with raw `asof_pitcher_success_rate` replacing `pitcher_success_shrunk`.
2. `c_shrunk_top20`: existing Top20 with leakage-safe `pitcher_success_shrunk`.
3. `c_shrunk_plus_reliability`: C-shrunk Top20 plus `pitcher_asof_reliability = n/(n+m)`.

## Prior-strength sweep

Full mode tests:

```text
m = 5, 10, 20, 30, 50, 75, 100, 150, 200
```

Quick mode tests `m = 10, 50, 100` with 5,000 rows per season, up to 300,000 Trackman rows, and HGB 120 iterations.

## Low-history analysis

Validation rows are split by `asof_pitcher_n`:

- `n < 20`
- `20 <= n < 50`
- `50 <= n < 100`
- `100 <= n < 300`
- `n >= 300`

The key comparison is raw vs shrunk Brier in the first two buckets.

## Colab

Open:

```text
notebooks/pitcher_shrinkage_c_colab.ipynb
```

Recommended order:

1. Mount Google Drive.
2. Sync/check out `agent/pitcher-shrinkage-c`.
3. Run package/version cell.
4. Confirm file paths and `MODE = 'quick'`.
5. Run the precheck cell. It verifies file existence and all required train/Trackman columns.
6. Run Quick.
7. If Quick succeeds, change only `MODE = 'full'` and rerun configuration, precheck, execution, and results cells.

### Cache rule

Trackman cache keys include `max_rows`. Therefore a cache created by a full run (`max_rows=None`) cannot be a cache hit for Quick (`max_rows=300000`). The Colab notebook intentionally does **not** pass the existing full-cache directory in Quick mode. Quick writes/uses its own 300k-row cache under the Quick output directory. Full mode may reuse `hgb_feature_selection_v2_full/cache` when its signature matches.

### Error diagnostics

The execution cell streams stdout/stderr together and retains the last 120 lines. If the subprocess exits non-zero, those lines are printed under `EXPERIMENT FAILED — LAST LOG LINES`, so the actual internal exception is visible instead of only a final `CalledProcessError` wrapper.

## Output files

- `pitcher_shrinkage_scores.csv`
- `pitcher_shrinkage_segment_scores.csv`
- `pitcher_shrinkage_summary.json`
- `validation_predictions_pitcher_shrinkage_best.csv.gz`

## Decision rule

Prefer promotion into the main model when:

1. full-run overall Brier is no worse than the comparison model,
2. improvement is consistent for low-history pitchers,
3. the selected `m` is not an isolated unstable optimum, and
4. adding `pitcher_asof_reliability` gives reproducible incremental value.

The fixed-350 full-run reference from the parent HGB V2 work is Brier `0.2480569104` and AUC `0.5489911`; interpret the shrinkage experiment primarily through its raw-vs-C comparison generated within the same run.
