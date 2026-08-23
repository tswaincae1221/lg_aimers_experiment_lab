# Rolling historical pitcher clustering experiment

## Design

For prediction season S, each pitcher profile is built only from information available before S. The profile combines long-horizon control rates, workload/reliability, pitch mix, Trackman release geometry/stability, movement dispersion, velocity/release drift, and recent-vs-older control change. KMeans candidates K={6,8,10} are compared using pre-2024 silhouette, adjusted Rand index (ARI), and aligned year-to-year cluster retention.

## Cluster selection (2024 excluded)

| K | silhouette_pre2024 | ARI_pre2024 | retention_pre2024 | rank_sum |
|---:|---:|---:|---:|---:|
| **6** | **0.109648** | **0.421826** | **0.686368** | **3** |
| 8 | 0.096319 | 0.327343 | 0.574952 | 7 |
| 10 | 0.094479 | 0.360083 | 0.548598 | 8 |

Selected K=6.

## K=6 stability

| from | to | common pitchers | ARI | aligned retention |
|---:|---:|---:|---:|---:|
| 2020 | 2021 | 190 | 0.2813 | 0.5737 |
| 2021 | 2022 | 205 | 0.3913 | 0.6341 |
| 2022 | 2023 | 241 | 0.4523 | 0.7386 |
| 2023 | 2024 | 229 | 0.3986 | 0.6943 |

The rolling clusters are moderately stable, but not stable enough to assume a fixed career archetype for every pitcher.

## Cluster ID as a direct predictive feature

| season | baseline Brier | +cluster6 Brier | Delta Brier |
|---:|---:|---:|---:|
| 2022 | 0.24462039 | **0.24460057** | -0.00001982 |
| 2023 | **0.25223145** | 0.25229352 | +0.00006207 |
| 2024 | **0.24928057** | 0.24932349 | +0.00004292 |

Paired block bootstrap:

- 2022 P(cluster better)=85.82%, CI crosses zero.
- 2023 P(cluster better)=0.28%, significantly worse.
- 2024 P(cluster better)=0.46%, significantly worse.

Conclusion: do not add the K=6 cluster ID directly as a global predictive feature.

## 2024 Native Cat104 vs HGB104 by cluster (diagnostic)

| cluster | n | F share | Cat104 Brier | HGB104 Brier | HGB-Cat |
|---:|---:|---:|---:|---:|---:|
| -1 | 50,348 | 0.125 | **0.247646** | 0.247883 | +0.000237 |
| 0 | 40,797 | 0.031 | **0.248511** | 0.248526 | +0.000015 |
| 1 | 48,279 | 0.150 | **0.247125** | 0.247216 | +0.000091 |
| 2 | 363 | 1.000 | 0.246980 | **0.245813** | **-0.001167** |
| 3 | 6,550 | 0.784 | **0.245977** | 0.246224 | +0.000247 |
| 4 | 96,258 | 0.083 | **0.248129** | 0.248211 | +0.000083 |
| 5 | 10,912 | 0.155 | **0.249345** | 0.249591 | +0.000246 |

Only cluster 2 favors HGB, but it contains just 363 2024 rows and seven pitchers. Even an oracle-like Cat/HGB route at this cluster granularity has only about 1.7e-6 overall Brier headroom, so Cat/HGB routing by K=6 cluster is not worth deploying.

## Decision

1. K=6 is the best of 6/8/10 for descriptive pitcher archetypes.
2. Do not use cluster ID as a direct global feature.
3. Do not hard-route Cat104/HGB only from cluster ID.
4. Keep clusters as an analysis/competence variable: test whether TM-v2 or future specialists consistently win in the same clusters across multiple forward seasons.
5. Prefer continuous mechanics/reliability features and CatBoost interactions over hard cluster labels unless a cluster-specific gain repeats across forward folds.
