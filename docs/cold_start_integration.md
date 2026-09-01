# Cold-start prior integration

The implementation is in `src/preprocess/cold_start.py`. It deliberately does
not train a model or edit the frozen S1 artifact by itself.

## Training notebook

After constructing the leakage-safe, one-row-per-pitcher-season Trackman table
(for example `analysis/outputs/trackman_pitcher_season_features.csv`), run:

```python
from src.preprocess.cold_start import (
    ColdStartConfig,
    attach_cold_start_prior,
    build_cold_start_lookup,
)

TM_PROFILE_COLUMNS = [
    c for c in trackman_pitcher_season_features.columns
    if c.startswith("tm_")
    and c not in {"tm_n", "tm_mapping_purity", "tm_source_season"}
]

COLD_START_LOOKUP, COLD_START_FALLBACKS = build_cold_start_lookup(
    train,
    trackman_pitcher_season_features,
    profile_columns=TM_PROFILE_COLUMNS,
    config=ColdStartConfig(neighbors=20, min_profile_pitches=50),
)
train = attach_cold_start_prior(
    train, COLD_START_LOOKUP, COLD_START_FALLBACKS,
)
```

Do this before the existing `build_features(train, ...)` call so derived season
and interaction features see the selected prior. Add these model features:

```python
COLD_START_FEATURES = [
    "cold_start_prior",
    "cold_start_source",
    "cold_start_neighbor_distance",
    "cold_start_neighbor_count",
    "is_cold_start",
]
CATEGORICAL_COLS += ["cold_start_source", "is_cold_start"]
FINAL_FEATURES += COLD_START_FEATURES
```

The feature builder must copy the five columns above from its input before it
selects `FINAL_FEATURES`. Store both tables in the submission bundle:

```python
bundle["cold_start_lookup"] = COLD_START_LOOKUP
bundle["cold_start_fallbacks"] = COLD_START_FALLBACKS
```

At inference, immediately after loading `test.csv` and before `build_features`,
apply the exact same static lookup:

```python
test = attach_cold_start_prior(
    test,
    bundle["cold_start_lookup"],
    bundle["cold_start_fallbacks"],
)
```

For a portable competition ZIP, copy `attach_cold_start_prior` into `script.py`
because the ZIP currently ships only `script.py`, requirements, and the bundle.

## Required validation

Compare the frozen S1 against this version on the 2024 holdout, especially:

- `asof_pitcher_n == 0`
- `asof_pitcher_n < 50`
- source: `trackman_knn`, `rookie_median`, `league_median`

The 2024 lookup must be constructed solely from 2023 Trackman profiles and 2023
outcomes. Do not use a full-data lookup for historical holdout scoring.
