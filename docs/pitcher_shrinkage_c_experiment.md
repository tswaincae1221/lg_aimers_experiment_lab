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

The practical expectation is that shrinkage helps most when `n` is small, because an extreme raw rate from only a few pitches is noisy.

## Validation contract

- Training rows: `season < 2024` (2019-2023)
- Validation rows: `season == 2024`
- Full HGB iterations: 350 fixed
- Primary metric: Brier Score (lower is better)
- Secondary metrics: Brier Skill Score and AUC
- Same Top20 context and Trackman feature pipeline as the existing Top20 experiment

## Compared variants

### 1. `raw_top20`

Uses the fixed Top20 feature set, but replaces `pitcher_success_shrunk` with raw `asof_pitcher_success_rate`.

This is the direct no-shrinkage reference.

### 2. `c_shrunk_top20`

Uses the existing Top20 feature set, including `pitcher_success_shrunk`.

### 3. `c_shrunk_plus_reliability`

Uses the C-shrunk Top20 set and adds:

```text
pitcher_asof_reliability = n / (n + m)
```

This explicitly tells HGB how much observed pitcher history supports the shrunk rate.

## Prior-strength sweep

Full mode tests:

```text
m = 5, 10, 20, 30, 50, 75, 100, 150, 200
```

The existing default is `m=50`; this experiment checks whether that value is actually best on the 2024 holdout.

Quick mode uses `m = 10, 50, 100` and smaller row limits only for a smoke test.

## Low-history analysis

Overall Brier alone is not enough for this question, so validation rows are also split by `asof_pitcher_n`:

- `n < 20`
- `20 <= n < 50`
- `50 <= n < 100`
- `100 <= n < 300`
- `n >= 300`

The key comparison is raw vs shrunk Brier in the first two buckets. If shrinkage improves those buckets while preserving overall Brier, that supports using population information for rookies / low-history pitchers.

## Colab

Open:

```text
notebooks/pitcher_shrinkage_c_colab.ipynb
```

Recommended order:

1. Mount Google Drive.
2. Clone/check out `agent/pitcher-shrinkage-c`.
3. Resolve `train.csv` and `trackman_history.csv` paths.
4. Run `MODE = "quick"` first.
5. If quick mode succeeds, change to `MODE = "full"` and rerun from the configuration cell.
6. Compare overall and pitcher-history-bucket Brier tables.

The notebook attempts to reuse an existing HGB V2 Trackman cache if one is found in Drive.

## Output files

- `pitcher_shrinkage_scores.csv`
- `pitcher_shrinkage_segment_scores.csv`
- `pitcher_shrinkage_summary.json`
- `validation_predictions_pitcher_shrinkage_best.csv.gz`

## Decision rule

Do not replace the current Top20 feature definition solely because one small-history bucket improves.

Prefer promotion into the main model when:

1. full-run overall Brier is no worse than the current comparison model,
2. improvement is consistent for low-history pitchers,
3. the selected `m` is not an isolated unstable optimum, and
4. adding `pitcher_asof_reliability` gives reproducible incremental value.

The current fixed-350 full-run reference from the parent HGB V2 work is Brier `0.2480569104` and AUC `0.5489911`; the shrinkage experiment should be interpreted relative to the raw/C Top20 comparison generated in the same run rather than mixing results from different preprocessing or capacity settings.
