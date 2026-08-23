# LG Aimers Trackman-v2 experiment — 2026-08-23

## Design

- Selection: train seasons 2019–2022, validation 2023.
- Final holdout: train seasons 2019–2023, validation 2024.
- 2024 was not used to select feature blocks.
- Screening model: CatBoost, 150 trees, depth 7, learning rate 0.03, L2 5, fixed seed 42.
- Trackman features for season S use only Trackman seasons < S.
- `batter_hand` is normalized as `1 -> Left`, `2 -> Right` before Trackman-context joins.

## Selected candidate

Candidate **I** replaces eight redundant / QC-oriented Trackman features with 22 TM-v2 features. Net feature count: **118**.

Removed from the 104-feature baseline:

- `tm_available`
- `tm_mapping_purity`
- `tm_mapping_grade`
- `tm_expected_release_dispersion`
- `tm_expected_movement_dispersion`
- `tm_context_expected_release_dispersion`
- `tm_context_expected_movement_dispersion`
- `tm_release_drift_abs_slope`

Added TM-v2 blocks:

- release-axis dispersion: side, height, extension
- release covariance ellipses: coronal, sagittal
- true count-conditioned mechanics shifts: release side/height, extension, velocity, spin
- signed fatigue/drift: side, height, extension, late/early release-dispersion ratio
- movement stability: IVB and horizontal break dispersion
- previous-season vs older-career mechanics shifts
- low-cardinality CatBoost archetypes: release slot, release stability, fatigue tier

## 2023 selection

| Variant | Features | Brier | Delta vs A |
|---|---:|---:|---:|
| A baseline104 | 104 | 0.252999514 | 0 |
| B + release axis | 107 | 0.253056782 | +0.000057268 |
| C + ellipse | 109 | 0.253031814 | +0.000032300 |
| D + true context | 114 | 0.253089507 | +0.000089993 |
| E + fatigue | 118 | 0.253094439 | +0.000094925 |
| F + movement | 120 | 0.253120496 | +0.000120982 |
| G + mechanics change | 123 | 0.253028148 | +0.000028634 |
| H remove redundant old TM | 115 | 0.253197752 | +0.000198238 |
| **I H + mechanics archetypes** | **118** | **0.252952537** | **-0.000046977** |

Paired block bootstrap, block size 500, 10,000 replicates:

- observed Delta Brier (I - A): **-0.00004698**
- 95% CI: **[-0.00007622, -0.00001788]**
- P(I better): **99.95%**

Reverse ablation from I worsened 2023 Brier for every removed block, indicating interaction effects rather than a single dominant continuous feature.

## 2024 holdout

| Variant | Brier | Brier Skill Score-like | AUC | Prediction bias |
|---|---:|---:|---:|---:|
| A baseline104 | 0.247934274 | 749.64 | 0.549324 | +0.007188 |
| **I TM-v2** | **0.247929446** | **751.57** | 0.549322 | +0.007136 |

Holdout changes:

- Delta Brier: **-0.000004829**
- Score-like: **+1.93 points**
- AUC: essentially unchanged (-0.0000022)
- 95% paired block-bootstrap CI: **[-0.00002776, +0.00001745]**
- P(I better): **66.05%**

The 2024 improvement is directionally consistent but not statistically conclusive.

## Regime diagnostic

The strongest stable signal is in `game_type == F`.

| Split | F Delta Brier | R Delta Brier |
|---|---:|---:|
| 2023 | **-0.000377** | -0.000008 |
| 2024 | **-0.0000381** | -0.00000036 |

This suggests TM-v2 is most useful as a **mechanics-regime signal for the more variable F population**, rather than as a universal replacement for the global 104-feature model.

## Decision

1. Keep the hand-normalization fix permanently.
2. Do **not** replace the global Native Cat104 baseline with the screening I model yet.
3. Promote TM-v2 I to the next **Native CatBoost GPU challenger**.
4. Prioritize F-regime use in future OOF routing / DES experiments, but select any routing rule on season-forward OOF only.
5. On GPU, compare Native Cat104 vs Native I under identical CTR1 settings, then test CTR2 only after the CTR1 comparison.
