import numpy as np
import pandas as pd

from src.experiment_models import fit_validation_model


def test_constant_model_uses_training_rate() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    valid = pd.DataFrame({"x": [5.0, 6.0]})
    result = fit_validation_model(
        {"type": "constant", "params": {}},
        train,
        pd.Series([0, 1, 1, 0]),
        valid,
        pd.Series([0, 1]),
        [],
        seed=42,
        n_jobs=1,
    )
    assert np.allclose(result.probability, 0.5)


def test_hist_gbdt_can_tune_on_separate_year_and_refit() -> None:
    rng = np.random.default_rng(42)
    tune_train = pd.DataFrame({"x": rng.normal(size=80)})
    tune_y = pd.Series((tune_train["x"] > 0).astype("int8"))
    tune_valid = pd.DataFrame({"x": rng.normal(size=30)})
    tune_valid_y = pd.Series((tune_valid["x"] > 0).astype("int8"))
    full_train = pd.concat([tune_train, tune_valid], ignore_index=True)
    full_y = pd.concat([tune_y, tune_valid_y], ignore_index=True)
    holdout = pd.DataFrame({"x": [-1.0, 1.0]})

    result = fit_validation_model(
        {
            "type": "hist_gbdt",
            "refit_after_tuning": True,
            "params": {
                "max_iter": 8,
                "min_samples_leaf": 2,
                "early_stopping": False,
            },
        },
        full_train,
        full_y,
        holdout,
        pd.Series([0, 1]),
        [],
        seed=42,
        n_jobs=1,
        tuning_data=(tune_train, tune_y, tune_valid, tune_valid_y),
    )

    assert result.probability.shape == (2,)
    assert 1 <= result.best_iteration <= 8
    assert result.importance.empty
