from __future__ import annotations

"""GPU-safe CatBoost experiment wrapper.

CatBoost does not support BrierScore as a GPU eval_metric. This wrapper keeps
GPU training with Logloss, then selects the tree count by explicitly computing
Brier score on the 2023 validation predictions at staged checkpoints.

It reuses the feature engineering, variants, scoring, and output logic from
catboost_native_experiment.py so the experiment stays directly comparable to
HGB-104.
"""

import logging
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss

from . import catboost_native_experiment as base

LOGGER = logging.getLogger(__name__)
_ORIGINAL_BASE_PARAMS = base._base_params


def _gpu_safe_base_params(args, *, iterations: int, max_ctr_complexity: int) -> dict:
    params = _ORIGINAL_BASE_PARAMS(
        args,
        iterations=iterations,
        max_ctr_complexity=max_ctr_complexity,
    )
    # BrierScore is not supported as a GPU eval_metric in CatBoost.
    # Optimize Logloss on GPU and select iterations with explicit Brier below.
    params["eval_metric"] = "Logloss"
    return params


def _predict_brier(model, valid_pool, y_valid: np.ndarray, ntree_end: int) -> float:
    probability = model.predict_proba(valid_pool, ntree_end=int(ntree_end))[:, 1]
    return float(brier_score_loss(y_valid, probability))


def _select_iterations_brier(
    frame: pd.DataFrame,
    categorical: list[str],
    target: pd.Series,
    seasons: pd.Series,
    args,
) -> tuple[int, pd.DataFrame]:
    train_mask = seasons.astype(int) < args.selection_season
    valid_mask = seasons.astype(int) == args.selection_season
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("selection split is empty")

    x_train = frame.loc[train_mask].reset_index(drop=True)
    x_valid = frame.loc[valid_mask].reset_index(drop=True)
    y_train = target.loc[train_mask].to_numpy("int8")
    y_valid = target.loc[valid_mask].to_numpy("int8")

    train_pool = base._pool(x_train, y_train, categorical)
    valid_pool = base._pool(x_valid, y_valid, categorical)

    model = CatBoostClassifier(
        **_gpu_safe_base_params(
            args,
            iterations=args.max_tune_iterations,
            max_ctr_complexity=args.anchor_max_ctr_complexity,
        )
    )

    LOGGER.info(
        "GPU-safe tuning: train seasons < %d, validate %d, features=%d, cats=%d, max_iter=%d",
        args.selection_season,
        args.selection_season,
        frame.shape[1],
        len(categorical),
        args.max_tune_iterations,
    )

    start = time.perf_counter()
    model.fit(train_pool, eval_set=valid_pool, use_best_model=False)
    train_elapsed = time.perf_counter() - start

    # Coarse Brier scan. We intentionally do not use CatBoost's GPU eval_metric
    # for Brier because that metric is unsupported on GPU.
    coarse_step = 25
    coarse_iterations = list(
        range(args.min_select_iterations, args.max_tune_iterations + 1, coarse_step)
    )
    if coarse_iterations[-1] != args.max_tune_iterations:
        coarse_iterations.append(args.max_tune_iterations)

    rows: list[dict] = []
    LOGGER.info("Computing explicit 2023 Brier on %d coarse checkpoints", len(coarse_iterations))
    for iteration in coarse_iterations:
        rows.append(
            {
                "iteration": int(iteration),
                "selection_brier": _predict_brier(model, valid_pool, y_valid, iteration),
                "stage": "coarse",
            }
        )

    coarse = pd.DataFrame(rows)
    coarse_best = int(coarse.loc[coarse["selection_brier"].idxmin(), "iteration"])

    # Refine near the best coarse checkpoint at 1-tree resolution.
    lo = max(args.min_select_iterations, coarse_best - coarse_step + 1)
    hi = min(args.max_tune_iterations, coarse_best + coarse_step - 1)
    existing = set(coarse["iteration"].astype(int))
    refine_iterations = [i for i in range(lo, hi + 1) if i not in existing]

    LOGGER.info(
        "Refining Brier around coarse best=%d over [%d, %d] (%d extra checkpoints)",
        coarse_best,
        lo,
        hi,
        len(refine_iterations),
    )
    for iteration in refine_iterations:
        rows.append(
            {
                "iteration": int(iteration),
                "selection_brier": _predict_brier(model, valid_pool, y_valid, iteration),
                "stage": "refine",
            }
        )

    curve = pd.DataFrame(rows).sort_values("iteration", ignore_index=True)
    best_idx = curve["selection_brier"].idxmin()
    selected = int(curve.loc[best_idx, "iteration"])
    best_brier = float(curve.loc[best_idx, "selection_brier"])
    curve["eligible"] = True
    curve["selected"] = curve["iteration"].eq(selected)
    curve["train_elapsed_seconds"] = float(train_elapsed)

    LOGGER.info(
        "Selected %d iterations by explicit 2023 Brier %.9f; max-iter GPU fit %.1fs",
        selected,
        best_brier,
        train_elapsed,
    )
    return selected, curve


def run(args) -> None:
    # Monkey-patch only the two internals that need GPU/Brier compatibility.
    base._base_params = _gpu_safe_base_params
    base._select_iterations = _select_iterations_brier
    base.run(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(base.parse_args())


if __name__ == "__main__":
    main()
