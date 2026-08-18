# Colab: literature-grounded HGB feature selection

This pipeline is designed for the LG Aimers control-success task and follows the competition constraints summarized in the repository docs.

## Data paths

Default Colab/Drive paths:

```text
/content/drive/MyDrive/aimers_data/train.csv
/content/drive/MyDrive/aimers_data/trackman_history.csv
```

Pitcher mapping is read from the repository:

```text
resources/pitcher_trackman_mapping.csv
```

The raw competition data must stay outside GitHub.

## One-cell Colab run

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/tswaincae1221/lg_aimers_experiment_lab.git
%cd /content/lg_aimers_experiment_lab
!git checkout agent/hgb-literature-feature-selection
!pip install -q -r requirements.txt joblib

!python -m src.hgb_feature_selection_pipeline \
  --train /content/drive/MyDrive/aimers_data/train.csv \
  --trackman /content/drive/MyDrive/aimers_data/trackman_history.csv \
  --mapping resources/pitcher_trackman_mapping.csv \
  --output-dir /content/drive/MyDrive/aimers_data/results/hgb_literature_feature_selection \
  --validation-season 2024 \
  --mode full
```

Use `--mode quick` first to verify paths and schema.

## What the pipeline does

1. Loads train, Trackman and the confirmed/high-confidence pitcher mapping.
2. Builds competition-safe row-wise features from official pre-pitch/as-of columns.
3. Builds compact Trackman historical mechanics priors with a strict `season < prediction season` cutoff.
4. Fits a fixed HistGradientBoostingClassifier on 2019-2023 and validates on 2024.
5. Runs feature-block ablations.
6. Computes holdout permutation importance using negative Brier score.
7. Runs actual drop-column LOFO retraining on the weakest non-protected candidates.
8. Retrains a compact selected HGB model and saves diagnostics/model artifacts to Drive.

## Main engineered features

- empirical-Bayes pitcher success prior and reliability
- recent-vs-career success/middle form and trend
- explicit count state, count flags and count-control interactions
- handedness matchup, batter-pitcher gaps
- base-out, RISP, leverage and score context
- pitch-mix entropy/dominance
- compact historical Trackman release/movement/velocity/spin dispersion
- context-conditioned expected Trackman mechanics using count + batter hand
- historical within-game mechanics drift slopes and inning interactions

The current pitch's actual pitch type, location, result, or test-row rolling statistics are never used.

## Output files

```text
feature_catalog.csv
block_ablation.csv
permutation_importance.csv
permutation_importance_top30.png
lofo_results.csv
lofo_delta_brier.png
model_scores.csv
validation_predictions.csv.gz
selected_features.txt
dropped_features.txt
hgb_selected_model.joblib
run_summary.json
```

Primary selection criterion is 2024 holdout Brier score. A feature is only automatically dropped in the LOFO stage if removing it improves Brier by at least `1e-5` by default.

## Notes

- Run `quick` before `full` in Colab.
- Full permutation/LOFO work is CPU-bound; GPU runtime is not required for scikit-learn HGB.
- Do not upload `train.csv` or `trackman_history.csv` to GitHub.
- The Trackman prior is intentionally historical: for a 2024 row it uses Trackman seasons through 2023 only; for a future 2025 evaluation row it can use through 2024.
