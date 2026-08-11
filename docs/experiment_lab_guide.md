# 자동 실험 운영 가이드

## 팀 공통 규칙

- 학습 행: `season < 2024`, 즉 2019~2023 시즌
- 검증 행: `season == 2024`
- 1순위 지표: Brier Score
- 기준 실험: `v1__lgbm_base`
- 개선 정의: `후보 Brier - 기준 Brier < 0`
- quick 모드 점수는 코드 점검에만 사용

다른 팀원이 만든 실험도 위 분할, 동일 원본 데이터, full 모드를 사용해야 직접 비교할
수 있습니다.

## 권장 실험 순서

### 1단계: 피처 효과 분리

`full + starter`를 실행합니다.

| 실험 | 확인할 내용 |
|---|---|
| `v1__lgbm_base` | V1 재현 여부 |
| `v1_asof__lgbm_base` | asof 추세 13개의 순효과 |
| `v1_asof_threat_ball_only__lgbm_base` | `threat_x_ball` 1개의 추가 효과 |
| `v1_asof_threat_no_ball__lgbm_base` | 핵심 후보를 제외한 나머지 5개의 효과 |
| `v1_asof_threat__lgbm_base` | 타자 위협 교차피처 6개 전체의 추가 효과 |
| `v1_asof_situation__lgbm_base` | 상황 상호작용의 추가 효과 |
| `v1_all__lgbm_base` | pitchmix·신뢰도까지 합친 효과 |

각 행은 모델 파라미터가 같으므로 Brier 변화는 피처 묶음 차이로 해석할 수 있습니다.
타자 위협 실험은 `comparison_experiment=v1_asof__lgbm_base`가 지정돼 있으므로
`brier_delta_vs_comparison`에서 기존 최고 asof 구성 대비 순수 변화가 자동 계산됩니다.

### 2단계: 모델 비교

피처 묶음이 정해진 뒤 `extended`를 실행합니다. 기본 설정은 전체 row-wise 피처를
사용하지만, 1단계 최고 피처가 다르면 `config/experiments.json`의 `feature_set`을
그 이름으로 맞춘 뒤 실행합니다.

### 3단계: 제거 실험

- `v1_all_no_ids__lgbm_base`: ID 의존성 확인
- `v1_all_no_trackman__lgbm_base`: Trackman 실제 기여 확인

제거했는데 점수가 좋아지면 해당 그룹이 잡음이나 과적합 원인일 수 있습니다. 단,
한 번의 2024 검증에 맞춘 판단일 수 있으므로 최종 제출 전에는 2022·2023 보조 검증도
별도로 확인하는 편이 안전합니다. 팀 공식 비교표는 여전히 2024 단일 검증을 사용합니다.

## 새 하이퍼파라미터 추가 예시

`models`에 다음 설정을 추가합니다.

```json
"lgbm_custom": {
  "type": "lightgbm",
  "description": "직접 만든 설정",
  "params": {
    "n_estimators": 1800,
    "early_stopping_rounds": 120,
    "learning_rate": 0.02,
    "num_leaves": 47,
    "min_child_samples": 400,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.3,
    "reg_lambda": 5.0
  }
}
```

그리고 `experiments`에 조합을 추가합니다.

```json
{
  "name": "v1_asof__lgbm_custom",
  "feature_set": "v1_asof",
  "model": "lgbm_custom",
  "presets": ["extended"]
}
```

실험 이름은 고유하게 유지합니다. 이름이 같고 설정만 바뀌어도 설정 해시가 달라져 새
실행으로 기록되지만, 리더보드 가독성을 위해 이름도 함께 바꾸는 편이 좋습니다.

## 결과 해석

| 열 | 의미 | 좋은 방향 |
|---|---|---|
| `brier` | 확률 예측 오차 | 낮음 |
| `brier_delta_vs_baseline` | V1 기준 대비 Brier 차이 | 음수 |
| `brier_improvement_pct` | V1 기준 대비 상대 개선률 | 높음 |
| `comparison_experiment` | 동일 모델로 맞춘 직접 비교 대상 | - |
| `brier_delta_vs_comparison` | 후보−직접 비교 대상 Brier | 음수 |
| `brier_improvement_vs_comparison_pct` | 직접 비교 대상 대비 상대 개선률 | 높음 |
| `logloss` | 틀린 확신에 큰 벌점을 주는 확률 오차 | 낮음 |
| `auc` | 양성·음성 순위 구분력 | 높음 |
| `ece_10bin` | calibration 오차 | 낮음 |
| `prediction_std` | 예측 확률의 퍼짐 정도 | 맥락에 따라 판단 |
| `elapsed_seconds` | 학습·예측 시간 | 낮음 |

Brier가 개선됐지만 AUC가 하락할 수 있습니다. 대회 주 지표가 Brier라면 Brier를
우선하되 calibration 곡선과 예측 분포를 함께 확인합니다.

## 오류와 재시작

개별 실험 오류는 `runs/<실행>/error.txt`에 저장됩니다. 나머지 조합은 계속
실행됩니다. 같은 데이터와 같은 설정으로 다시 실행하면 성공 조합은 자동으로
건너뜁니다.

강제로 다시 실행하려면 `--rerun`을 사용하거나 Colab에서 `RERUN=True`로 바꿉니다.
기존 이력은 삭제되지 않고 새 행으로 누적됩니다.
