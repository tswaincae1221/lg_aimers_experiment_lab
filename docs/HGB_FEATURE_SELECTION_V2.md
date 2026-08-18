# HGB Literature-Grounded Feature Selection V2

This pipeline is a clean rewrite for Google Colab. It keeps the data-loading, season-forward validation, HGB preprocessing, and Trackman leakage rules that already worked in the existing project, while replacing the candidate feature design with the feature-selection plan discussed in the project review.

## Validation contract

- Train: seasons `< 2024` (2019-2023)
- Final holdout: season `== 2024`
- HGB iteration tuning: seasons `< 2023` -> validate on 2023
- After tuning, the iteration count is fixed for full model, block ablation, and LOFO so feature effects are not mixed with hyperparameter changes.
- Primary metric: Brier Score (lower is better)

## Competition-safety rules implemented

- Only official pre-pitch columns are used from `train.csv`.
- No current-pitch actual location, result, or actual pitch type is used.
- Official `asof_*` columns are used row-wise as provided.
- No rolling/expanding statistics are created from evaluation rows.
- Trackman features for a row in season `S` use only Trackman seasons `< S`.
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

Shrinkage prior for a row uses only labeled seasons strictly before that row's season. If no prior season exists, the fallback prior is 0.5.

### 3. recent_form
- success form: prev1/3/5 minus long-term success
- middle form: prev1/3/5 minus long-term middle rate
- prev1-vs-prev3, prev1-vs-prev5, prev3-vs-prev5 trends
- recent success/middle volatility

All are row-wise transformations of official pre-pitch `asof_*` columns.

### 4. count_intent
- `count_state`
- ball-strike difference
- first pitch / two strike / three ball / full count
- pitcher ahead / behind
- selected count x control interactions

### 5. matchup
- `hand_pair`, `hand_match`
- pitcher-batter success and middle gaps
- `success_interact`
- `reverse_x_middle`
- batter-control-threat proxy interactions (`threat_x_reverse`, `threat_x_middle`, `threat_x_ball`)
- hand x pitch-mix interactions

### 6. game_context
- `base_out_state`
- RISP / first base open
- absolute score difference / close game / late inning
- `log1p(LI)` / high leverage
- full-count x LI / late-close-high-LI
- runner-out pressure
- pitcher-team win expectancy and pressure

### 7. pitchmix
- entropy
- dominant pitch-group rate
- pitch-group rate gaps

### 8. trackman_repeatability
Trackman is aggregated by pitcher x historical season x pitch group (`fastball`, `breaking`, `offspeed`). Dispersion is based on within-group standard deviations and is normalized against the historical peer median, then shrunk toward league-average dispersion for small samples.

Families:
- release: `rel_height`, `rel_side`, `extension`
- velocity: `rel_speed`, `zone_speed`
- movement: `induced_vert_break`, `horz_break`
- spin: `spin_rate`

Group-specific dispersions are intermediate calculations only. The HGB input stays compact and receives the pitch-mix-weighted expected release/velocity/movement/spin dispersion features plus reliability/mapping indicators.

### 9. trackman_context
Historical Trackman counts estimate:

`P(pitch_group | pitcher, balls, strikes, batter_hand)`

with shrinkage to the pitcher's historical overall pitch-group mix. These probabilities weight release/movement dispersion to create context-conditioned expected mechanics features without using the current pitch's actual pitch type.

### 10. trackman_drift
Chunk-safe sufficient statistics are used to estimate per-game slopes versus `pitch_no`, then historical pitcher averages are created for:
- velocity decay slope
- absolute release drift slope
- number of contributing games

## HGB preprocessing

The pipeline follows the existing working `experiment_models.py` pattern:

- categorical feature -> frequency encoding fit on the training split only
- numeric feature -> training-split median imputation
- validation categories unseen in training -> 0 frequency
- arrays are `float32`

HGB defaults match the existing experiment configuration:

- learning rate: 0.06
- max leaf nodes: 31
- min samples leaf: 250
- L2 regularization: 3.0
- max iterations: 350 before 2023 iteration tuning

## Feature-selection sequence

1. Fit full candidate model.
2. Block ablation: remove one feature block at a time and re-fit using the same HGB iteration count.
3. Permutation importance on 2024 holdout, scored by `delta Brier`.
4. LOFO/drop-column re-fitting for the weakest permutation candidates.
5. Drop a LOFO candidate only when removing it worsens Brier by no more than the configured tolerance (`1e-5` default).
6. Fit all selected features together.
7. If combined feature removal worsens 2024 Brier by more than `5e-5` by default, automatically fall back to the full feature set.

Sign conventions:

- block `delta_brier = Brier(without block) - Brier(full)`
- permutation `delta_brier = Brier(permuted) - Brier(baseline)`
- LOFO `delta_brier = Brier(without feature) - Brier(full)`

Positive values mean the removed/permuted feature helped the full model.

## Colab execution

Use `notebooks/hgb_feature_selection_v2_colab.ipynb`.

Run `quick` first. It caps each season to 5,000 rows, Trackman to 300,000 rows, permutation sample to 15,000, LOFO to 8 candidates, and HGB tuning to at most 120 iterations.

After quick succeeds, change `MODE = "full"` and rerun the pipeline cell.

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

The original competition CSVs are never copied into GitHub.

## Colab resume behavior

The V2 runner follows the earlier working LOFO runner's checkpoint pattern. Trackman sufficient-statistic aggregation is cached under `OUTPUT_DIR/cache`, while block ablation, permutation importance, and LOFO progress are checkpointed under `OUTPUT_DIR/checkpoints`. Re-running the same command resumes completed work when the input files and run parameters match. Pass `--rerun` only when you intentionally want to ignore the cache/checkpoints and recompute everything.
