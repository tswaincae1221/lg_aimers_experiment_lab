# Trackman-v2 CatBoost experiment

This folder reproduces the 2026-08-23 LG Aimers Trackman-v2 feature experiment.

## Experiment rule

- Feature-block selection: train 2019–2022, validate 2023.
- Final holdout: train 2019–2023, validate 2024.
- Trackman features for prediction season `S` use only Trackman rows with `season < S`.
- The Trackman context join normalizes `batter_hand` as `1 -> Left`, `2 -> Right`.
- The 2024 holdout is not used to choose the A→I feature sequence.

## Colab GPU quick start

1. Start a Colab GPU runtime.
2. Put `train.csv` and `trackman_history.csv` in Google Drive, e.g. `/content/drive/MyDrive/aimers_data/`.
3. Run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone -b agent/tmv2-paper-features https://github.com/tswaincae1221/lg_aimers_experiment_lab.git
%cd /content/lg_aimers_experiment_lab
!pip -q install catboost==1.2.8 openpyxl
```

Define paths:

```python
import os
os.environ['TMV2_ROOT'] = '/content/tmv2_work'
os.environ['TMV2_OUT'] = '/content/tmv2_work/tmv2_experiment'
os.environ['TMV2_TRAIN'] = '/content/drive/MyDrive/aimers_data/train.csv'
os.environ['TMV2_TRACKMAN'] = '/content/drive/MyDrive/aimers_data/trackman_history.csv'
os.environ['TMV2_MAPPING'] = '/content/lg_aimers_experiment_lab/resources/pitcher_trackman_mapping.csv'
os.environ['TMV2_TASK_TYPE'] = 'GPU'
os.environ['TMV2_DEVICES'] = '0'
```

Build the hand-fixed 104-feature cache:

```bash
!python experiments/tmv2/prepare_tmv2_cache.py \
  --repo-root /content/lg_aimers_experiment_lab \
  --train "$TMV2_TRAIN" \
  --trackman "$TMV2_TRACKMAN" \
  --mapping "$TMV2_MAPPING" \
  --work-dir "$TMV2_ROOT"
```

Build the leakage-safe TM-v2 physical features:

```bash
!python experiments/tmv2/build_tmv2_features.py
```

Run the 2023 screening sequence. `150` is the fixed screening iteration used in the recorded experiment:

```bash
!python experiments/tmv2/run_tmv2_screen.py --label A_baseline104 --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label B_release_axis --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label C_plus_ellipse --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label D_plus_true_context --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label E_plus_fatigue --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label F_plus_movement --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label G_plus_mechanics_change --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label H_G_clean_old_tm --valid 2023 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label I_G_clean_plus_archetype --valid 2023 --iterations 150
```

The 2023 winner was `I_G_clean_plus_archetype`. Verify only A and I on the 2024 holdout:

```bash
!python experiments/tmv2/run_tmv2_screen.py --label A_baseline104 --valid 2024 --iterations 150
!python experiments/tmv2/run_tmv2_screen.py --label I_G_clean_plus_archetype --valid 2024 --iterations 150
```

## Native CatBoost GPU challenger

The screening runner intentionally uses a low-cost one-hot configuration to compare feature blocks. For the production challenger, use native CatBoost categorical handling / CTR1:

```bash
!python experiments/tmv2/run_tmv2_native.py --label A_baseline104 --valid 2024 --iterations 216
!python experiments/tmv2/run_tmv2_native.py --label I_G_clean_plus_archetype --valid 2024 --iterations 216
```

Do not treat the native result as final until the iteration count is selected on 2023 rather than 2024.

## Recorded result

See `results/TMV2_EXPERIMENT_REPORT_2026-08-23.md`.

Key holdout result in the screening environment:

- A: Brier `0.247934274`, score-like `749.64`
- I: Brier `0.247929446`, score-like `751.57`
- Delta Brier: `-0.000004829`
- 2024 paired block-bootstrap 95% CI: `[-0.00002776, +0.00001745]`

The gain is directionally positive but not conclusive. The most repeatable subgroup signal was `game_type == F`.
