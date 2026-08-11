from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def calibration_table(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype="float64")
    probability = np.clip(np.asarray(probability, dtype="float64"), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    frame = pd.DataFrame({"target": truth, "prediction": probability, "bin": bucket})
    result = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("target", "size"),
            actual_rate=("target", "mean"),
            prediction_mean=("prediction", "mean"),
            prediction_min=("prediction", "min"),
            prediction_max=("prediction", "max"),
        )
        .reset_index()
    )
    result["absolute_gap"] = (result["actual_rate"] - result["prediction_mean"]).abs()
    return result


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype="int8")
    probability = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    target_rate = float(truth.mean())
    brier = float(brier_score_loss(truth, probability))
    reference = target_rate * (1.0 - target_rate)
    calibration = calibration_table(truth, probability)
    ece = float((calibration["n"] * calibration["absolute_gap"]).sum() / len(truth))
    auc = float(roc_auc_score(truth, probability)) if np.unique(truth).size == 2 else np.nan
    return {
        "valid_n": int(len(truth)),
        "target_rate": target_rate,
        "prediction_mean": float(probability.mean()),
        "prediction_std": float(probability.std()),
        "brier": brier,
        "brier_skill_score": float(100_000.0 * (1.0 - brier / reference))
        if reference
        else 0.0,
        "logloss": float(log_loss(truth, probability, labels=[0, 1])),
        "auc": auc,
        "ece_10bin": ece,
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def write_run_artifacts(
    run_dir: Path,
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    importance: pd.DataFrame,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(y_true, dtype="int8")
    probability = np.asarray(probability, dtype="float64")
    calibration = calibration_table(truth, probability)
    calibration.to_csv(run_dir / "calibration_bins.csv", index=False)

    plt.figure(figsize=(6.2, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect")
    plt.plot(
        calibration["prediction_mean"],
        calibration["actual_rate"],
        marker="o",
        color="#2166ac",
        label="Model",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed success rate")
    plt.title("Calibration curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.2)
    _save_figure(run_dir / "calibration_curve.png")

    plt.figure(figsize=(7.2, 4.8))
    plt.hist(
        probability[truth == 0], bins=40, alpha=0.55, density=True, label="target=0"
    )
    plt.hist(
        probability[truth == 1], bins=40, alpha=0.55, density=True, label="target=1"
    )
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Prediction distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    _save_figure(run_dir / "prediction_distribution.png")

    if not importance.empty:
        importance.to_csv(run_dir / "feature_importance.csv", index=False)
        top = importance.head(30).sort_values("importance", ascending=True)
        plt.figure(figsize=(8.5, max(5.5, len(top) * 0.24)))
        plt.barh(top["feature"], top["importance"], color="#4393c3")
        plt.xlabel(str(top["importance_type"].iloc[0]))
        plt.title("Top feature importance")
        _save_figure(run_dir / "feature_importance_top30.png")


def add_baseline_deltas(
    leaderboard: pd.DataFrame,
    baseline_experiment: str,
) -> pd.DataFrame:
    result = leaderboard.copy()
    baseline = result.loc[result["experiment"] == baseline_experiment]
    if baseline.empty:
        result["brier_delta_vs_baseline"] = np.nan
        result["brier_improvement_pct"] = np.nan
        return result
    baseline_brier = float(baseline.iloc[0]["brier"])
    result["brier_delta_vs_baseline"] = result["brier"] - baseline_brier
    result["brier_improvement_pct"] = (
        (baseline_brier - result["brier"]) / baseline_brier * 100.0
    )
    return result


def add_comparison_deltas(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise deltas for experiments that declare a comparison row."""
    result = leaderboard.copy()
    result["comparison_brier"] = np.nan
    result["brier_delta_vs_comparison"] = np.nan
    result["brier_improvement_vs_comparison_pct"] = np.nan
    if "comparison_experiment" not in result.columns:
        return result

    brier_by_experiment = result.set_index("experiment")["brier"]
    comparison_brier = result["comparison_experiment"].map(brier_by_experiment)
    result["comparison_brier"] = comparison_brier
    result["brier_delta_vs_comparison"] = result["brier"] - comparison_brier
    result["brier_improvement_vs_comparison_pct"] = (
        (comparison_brier - result["brier"]) / comparison_brier * 100.0
    )
    return result


def add_preprocessing_deltas(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Compare engineered and raw-minimal experiments under matched conditions."""
    result = leaderboard.copy()
    result["preprocessing_comparison_brier"] = np.nan
    result["brier_delta_vs_preprocessing"] = np.nan
    result["brier_improvement_vs_preprocessing_pct"] = np.nan
    if "preprocessing_comparison_experiment" not in result.columns:
        return result

    brier_by_experiment = result.set_index("experiment")["brier"]
    comparison_brier = result["preprocessing_comparison_experiment"].map(
        brier_by_experiment
    )
    result["preprocessing_comparison_brier"] = comparison_brier
    result["brier_delta_vs_preprocessing"] = result["brier"] - comparison_brier
    result["brier_improvement_vs_preprocessing_pct"] = (
        (comparison_brier - result["brier"]) / comparison_brier * 100.0
    )
    return result


def write_global_artifacts(
    output_dir: Path,
    leaderboard: pd.DataFrame,
    baseline_experiment: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    board = leaderboard.sort_values("brier", ascending=True).reset_index(drop=True)
    board.to_csv(output_dir / "leaderboard.csv", index=False)

    plt.figure(figsize=(9.0, max(4.5, len(board) * 0.48)))
    plot_data = board.sort_values("brier", ascending=False)
    colors = [
        "#2166ac" if name == baseline_experiment else "#67a9cf"
        for name in plot_data["experiment"]
    ]
    plt.barh(plot_data["experiment"], plot_data["brier"], color=colors)
    plt.xlabel("Brier Score (lower is better)")
    plt.title("2024 validation leaderboard")
    for index, value in enumerate(plot_data["brier"]):
        plt.text(value, index, f" {value:.6f}", va="center", fontsize=8)
    plt.grid(axis="x", alpha=0.2)
    _save_figure(output_dir / "leaderboard_brier.png")

    deltas = board.dropna(subset=["brier_delta_vs_baseline"]).copy()
    if not deltas.empty:
        deltas = deltas.sort_values("brier_delta_vs_baseline", ascending=False)
        colors = [
            "#1a9850" if value < 0 else "#d73027"
            for value in deltas["brier_delta_vs_baseline"]
        ]
        plt.figure(figsize=(9.0, max(4.5, len(deltas) * 0.48)))
        plt.barh(deltas["experiment"], deltas["brier_delta_vs_baseline"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Brier delta vs baseline (negative is better)")
        plt.title("Improvement over baseline")
        plt.grid(axis="x", alpha=0.2)
        _save_figure(output_dir / "improvement_vs_baseline.png")

    comparisons = board.dropna(subset=["brier_delta_vs_comparison"]).copy()
    if not comparisons.empty:
        comparisons = comparisons.sort_values(
            "brier_delta_vs_comparison", ascending=False
        )
        colors = [
            "#1a9850" if value < 0 else "#d73027"
            for value in comparisons["brier_delta_vs_comparison"]
        ]
        labels = [
            f"{row.experiment}\nvs {row.comparison_experiment}"
            for row in comparisons.itertuples()
        ]
        plt.figure(figsize=(10.5, max(4.5, len(comparisons) * 0.7)))
        plt.barh(labels, comparisons["brier_delta_vs_comparison"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Brier delta vs paired comparison (negative is better)")
        plt.title("Incremental feature effect within the same model")
        plt.grid(axis="x", alpha=0.2)
        _save_figure(output_dir / "improvement_vs_comparison.png")

    preprocessing = board.dropna(subset=["brier_delta_vs_preprocessing"]).copy()
    if not preprocessing.empty:
        preprocessing = preprocessing.sort_values(
            "brier_delta_vs_preprocessing", ascending=False
        )
        colors = [
            "#1a9850" if value < 0 else "#d73027"
            for value in preprocessing["brier_delta_vs_preprocessing"]
        ]
        labels = [
            f"{row.experiment}\nvs {row.preprocessing_comparison_experiment}"
            for row in preprocessing.itertuples()
        ]
        plt.figure(figsize=(10.5, max(4.5, len(preprocessing) * 0.7)))
        plt.barh(labels, preprocessing["brier_delta_vs_preprocessing"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Brier delta: engineered minus raw-minimal (negative is better)")
        plt.title("Feature engineering effect with matched HistGradientBoosting")
        plt.grid(axis="x", alpha=0.2)
        _save_figure(output_dir / "preprocessing_before_after.png")

    model_families = {"LightGBM", "HistGradientBoosting", "XGBoost", "CatBoost"}
    four_model_rows = board.loc[
        board.get("model_family", pd.Series(index=board.index, dtype="object")).isin(
            model_families
        )
    ].copy()
    if not four_model_rows.empty:
        four_model_rows.sort_values(
            ["preprocessing_stage", "brier"],
            ascending=[True, True],
        ).to_csv(output_dir / "four_model_stage_comparison.csv", index=False)

        final_models = four_model_rows.loc[
            four_model_rows["preprocessing_stage"] == "v1_all_threat"
        ].sort_values("brier", ascending=True)
        if not final_models.empty:
            final_models.to_csv(
                output_dir / "four_model_final_comparison.csv",
                index=False,
            )
            plot_data = final_models.sort_values("brier", ascending=False)
            plt.figure(figsize=(8.5, max(4.5, len(plot_data) * 0.65)))
            plt.barh(plot_data["model_family"], plot_data["brier"], color="#4393c3")
            for index, value in enumerate(plot_data["brier"]):
                plt.text(value, index, f" {value:.6f}", va="center", fontsize=9)
            plt.xlabel("Brier Score (lower is better)")
            plt.title("Four-model comparison: V1 all + batter-threat features")
            plt.grid(axis="x", alpha=0.2)
            _save_figure(output_dir / "four_model_final_comparison.png")

    if {"elapsed_seconds", "brier"}.issubset(board.columns):
        plt.figure(figsize=(7.0, 5.0))
        plt.scatter(board["elapsed_seconds"], board["brier"], color="#2166ac", s=48)
        for _, row in board.iterrows():
            plt.annotate(
                row["experiment"],
                (row["elapsed_seconds"], row["brier"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        plt.xlabel("Training time (seconds)")
        plt.ylabel("Brier Score")
        plt.title("Score vs training time")
        plt.grid(alpha=0.2)
        _save_figure(output_dir / "score_vs_time.png")


def write_markdown_report(
    output_dir: Path,
    leaderboard: pd.DataFrame,
    baseline_experiment: str,
    validation_season: int,
) -> None:
    board = leaderboard.sort_values("brier", ascending=True).reset_index(drop=True)
    best = board.iloc[0]
    baseline = board.loc[board["experiment"] == baseline_experiment]
    lines = [
        "# LG Aimers 자동 실험 보고서",
        "",
        (
            f"- 검증 규칙: `{validation_season}` 미만 시즌 학습 → "
            f"`{validation_season}` 시즌 검증"
        ),
        f"- 완료 실험 수: `{len(board)}`",
        f"- 최고 실험: `{best['experiment']}`",
        f"- 최고 Brier Score: `{best['brier']:.8f}`",
    ]
    if not baseline.empty:
        baseline_brier = float(baseline.iloc[0]["brier"])
        delta = float(best["brier"] - baseline_brier)
        lines.extend(
            [
                f"- 기준 실험: `{baseline_experiment}` / `{baseline_brier:.8f}`",
                f"- 최고 실험의 기준 대비 변화: `{delta:+.8f}` (음수면 개선)",
            ]
        )
    lines.extend(["", "## 리더보드", ""])
    display_columns = [
        "experiment",
        "feature_set",
        "model",
        "model_family",
        "input_view",
        "preprocessing_stage",
        "feature_count",
        "brier",
        "brier_delta_vs_baseline",
        "comparison_experiment",
        "brier_delta_vs_comparison",
        "preprocessing_comparison_experiment",
        "brier_delta_vs_preprocessing",
        "auc",
        "logloss",
        "ece_10bin",
        "elapsed_seconds",
    ]
    display = board[[column for column in display_columns if column in board.columns]].copy()
    headers = display.columns.tolist()
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in display.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if pd.isna(value):
                formatted.append("")
            elif isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.6f}")
            else:
                formatted.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            "## 해석 기준",
            "",
            "- Brier Score와 Log Loss는 낮을수록 좋습니다.",
            (
                "- AUC는 높을수록 좋지만, 대회 주 지표가 Brier라면 "
                "채택 판단은 Brier를 우선합니다."
            ),
            "- `brier_delta_vs_baseline < 0`이면 기준 모델보다 개선된 것입니다.",
            (
                "- `brier_delta_vs_comparison < 0`이면 동일 모델의 지정 비교 실험보다 "
                "추가 피처가 개선된 것입니다."
            ),
            (
                "- `brier_delta_vs_preprocessing < 0`이면 동일 조건의 원본 최소처리 "
                "모델보다 피처 엔지니어링 후 성능이 개선된 것입니다."
            ),
            (
                "- 표본 제한을 사용한 quick 결과는 코드 점검용이며 "
                "최종 모델 채택에 사용하지 않습니다."
            ),
            (
                "- `four_models` 실험은 2023으로 반복 수를 고른 뒤 2019~2023으로 "
                "재학습하고, 2024는 마지막 평가에만 사용합니다."
            ),
            "",
        ]
    )
    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
