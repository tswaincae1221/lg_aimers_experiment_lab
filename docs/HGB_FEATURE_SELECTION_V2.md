# HGB Literature-Grounded Feature Selection V2

This pipeline is a Google Colab feature-selection pipeline built around the data-loading, season-forward validation, HGB preprocessing, and Trackman leakage rules that already worked in the project.

## Validation contract

- Train: seasons `< 2024` (2019-2023)
- Final holdout: season `== 2024`
- **Full HGB feature-selection runs use exactly 350 iterations.**
- There is no 2023-based iteration tuning in the full experiment.
- The same HGB capacity is used for the full model, block ablation, LOFO, and the selected model.
- Primary metric: Brier Score (lower is better).

This fixed-capacity policy was adopted after the previous run selected only 4 iterations on 2023 and severely underfit the 2024 holdout.

## Competition-safety rules implemented

- Only official pre-pitch columns are used from `train.csv`.
- No current-pitch actual location, result, or actual pitch type is used.
- Official `asof_*` columns are used row-wise as provided.
- No rolling/expanding statistics are created from evaluation rows.
- Trackman features for a row in season `S` use only Trackman seasons `< S`.
- `season_trend_prior` for season `S` uses only labeled target data from seasons `< S`.
- Trackman is joined through `resources/pitcher_trackman_mapping.csv`; only grades `확정` and `높음` are accepted by default.

## Feature blocks

### 1. official_baseline
Official pre-pitch numeric/categorical context and official `asof_*` rates. High-cardinality `pitcher_id` and `batter_id` are retained only as join keys and are not sent directly to HGB.

### 2. control_prior
- `asof_pitcher_n_log1p`
- `asof_batter_n_log1p`
- `pitcher_success_shrunk`
- `batter_success_shrunk`
- pitcher/batter as-of reliability

The two shrunk success-rate features use `season_trend_prior` as their prior once the new season-trend block is attached.

### 3. season_trend
Added to handle the strong season-level success-rate drift that tree splits cannot naturally extrapolate beyond the latest training year.

For a target season `S`, only completed seasons `< S` are used.

- `season_trend_prior`: predicted season-level success-rate prior
- `season_trend_slope`: fitted annual success-rate slope
- `season_trend_history_n`: number of completed seasons available to the trend fit

Behavior:

- If fewer than 4 completed seasons are available, use the row-count-weighted mean of prior-season target rates.
- Once at least 4 completed seasons are available, fit a linear trend to completed season-level success rates and extrapolate to `S`.
- The extrapolated prior is clipped to `[0.01, 0.99]`.
- Same-season and future-season target labels are never used.

Example: the 2024 row-level prior can use 2019-2023 labels, but cannot use any 2024 labels.

### 4. recent_form
- success form: prev1/3/5 minus long-term success
- middle form: prev1/3/5 minus long-term middle rate
- prev1-vs-prev3, prev1-vs-prev5, prev3-vs-prev5 trends
- recent success/middle volatility

All are row-wise transformations of official pre-pitch `asof_*` columns.

### 5. count_intent
- `count_state`
- ball-strike difference
- first pitch / two strike / three ball / full count
- pitcher ahead / behind
- selected count x control interactions

### 6. matchup
- `hand_pair`, `hand_match`
- pitcher-batter success and middle gaps
- `success_interact`
- `reverse_x_middle`
- batter-control-threat proxy interactions (`threat_x_reverse`, `threat_x_middle`, `threat_x_ball`)
- hand x pitch-mix interactions

### 7. game_context
- `base_out_state`
- RISP / first base open
- absolute score difference / close game / late inning
- `log1p(LI)` / high leverage
- full-count x LI / late-close-high-LI
- runner-out pressure
- pitcher-team win expectancy and pressure

### 8. pitchmix
- entropy
- dominant pitch-group rate
- pitch-group rate gaps

### 9. trackman_repeatability
Trackman is aggregated by pitcher x historical season x pitch group (`fastball`, `breaking`, `offspeed`). Dispersion is based on within-group standard deviations and is normalized against the historical peer median, then shrunk toward league-average dispersion for small samples.

Families:
- release: `rel_height`, `rel_side`, `extension`
- velocity: `rel_speed`, `zone_speed`
- movement: `induced_vert_break`, `horz_break`
- spin: `spin_rate`

Group-specific dispersions are intermediate calculations only. The HGB input stays compact and receives pitch-mix-weighted expected release/velocity/movement/spin dispersion plus reliability/mapping indicators.

### 10. trackman_context
Historical Trackman counts estimate:

`P(pitch_group | pitcher, balls, strikes, batter_hand)`

with shrinkage to the pitcher's historical overall pitch-group mix. These probabilities weight release/movement dispersion to create context-conditioned expected mechanics features without using the current pitch's actual pitch type.

### 11. trackman_drift
Chunk-safe sufficient statistics estimate historical pitcher tendencies for:
- velocity decay slope
- absolute release drift slope
- number of contributing games

## HGB preprocessing

The pipeline follows the existing working `experiment_models.py` pattern:

- categorical feature -> frequency encoding fit on the training split only
- numeric feature -> training-split median imputation
- validation categories unseen in training -> 0 frequency
- arrays are `float32`

Full-run HGB parameters:

- learning rate: 0.06
- max leaf nodes: 31
- min samples leaf: 250
- L2 regularization: 3.0
- **max iterations: 350 fixed**
- early stopping: off

`quick` mode may cap iterations at 120 for smoke testing only. Full mode overrides `--max-iter` and always uses 350.

## Feature-selection sequence

1. Fit full candidate model at 350 fixed iterations.
2. Block ablation: remove one feature block at a time and re-fit at the same 350 iterations.
3. Permutation importance on 2024 holdout, scored by `delta Brier`.
4. LOFO/drop-column re-fitting for the weakest permutation candidates at the same 350 iterations.
5. Drop a LOFO candidate only when removing it worsens Brier by no more than the configured tolerance (`1e-5` default).
6. Fit all selected features together at 350 iterations.
7. If combined feature removal worsens 2024 Brier by more than `5e-5` by default, automatically fall back to the full feature set.

Sign conventions:

- block `delta_brier = Brier(without block) - Brier(full)`
- permutation `delta_brier = Brier(permuted) - Brier(baseline)`
- LOFO `delta_brier = Brier(without feature) - Brier(full)`

Positive values mean the removed/permuted feature helped the full model.

## Colab execution

Use `notebooks/hgb_feature_selection_v2_colab.ipynb`.

The old notebook cell may still pass `--tuning-season 2023`; this argument is retained only for backward compatibility and is ignored. Full mode still runs 350 fixed iterations.

Run `quick` first. It caps each season to 5,000 rows, Trackman to 300,000 rows, permutation sample to 15,000, LOFO to 8 candidates, and HGB to at most 120 iterations.

After quick succeeds, change `MODE = "full"` and rerun the pipeline cell. Full mode uses the entire 2019-2023 training set, the 2024 holdout, full Trackman history, and 350 fixed HGB iterations.

## Outputs

- `model_scores.csv`
- `block_ablation.csv`
- `permutation_importance.csv`
- `lofo_results.csv`
- `feature_selection.csv`
- `feature_catalog.csv`
- `selected_features.txt`
- `dropped_features.txt`
- `validation_predictions.csv.gz`
- `hgb_selected_model.joblib`
- `run_summary.json`

`run_summary.json` now records `iteration_policy: fixed`, `fixed_iteration: 350`, and the season-trend configuration.

The original competition CSVs are never copied into GitHub.

## Colab resume behavior

Trackman sufficient-statistic aggregation is cached under `OUTPUT_DIR/cache`, while block ablation, permutation importance, and LOFO progress are checkpointed under `OUTPUT_DIR/checkpoints`.

The run signature now includes pipeline version 3, fixed iteration count, season-trend settings, data signatures, and feature names. Therefore old 4-iteration selection checkpoints are not reused by the new full run. Pass `--rerun` only when you intentionally want to ignore matching caches/checkpoints and recompute everything.
