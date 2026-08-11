# 4개 모델·전처리 전후 비교 설계

## 비교 질문

이 프리셋은 한 번의 실행으로 세 가지 질문을 분리합니다.

1. 공식 원본 피처만 쓸 때와 전체 피처 엔지니어링 후 중 어느 쪽이 좋은가?
2. 타자 위협도 교차피처 6개가 각 모델에서 추가로 개선되는가?
3. 최종 피처 조건에서 네 모델 중 어느 모델의 2024 Brier Score가 가장 낮은가?

## 12개 실험 행렬

| 모델 | 원본 최소처리 | V1 전체 | V1 전체 + 위협도 6개 |
|---|---|---|---|
| LightGBM | `raw_minimal__lgbm_four` | `v1_all__lgbm_four` | `v1_all_threat__lgbm_four` |
| HistGradientBoosting | `raw_minimal__hist_gbdt_four` | `v1_all__hist_gbdt_four` | `v1_all_threat__hist_gbdt_four` |
| XGBoost | `raw_minimal__xgboost_four` | `v1_all__xgboost_four` | `v1_all_threat__xgboost_four` |
| CatBoost | `raw_minimal__catboost_four` | `v1_all__catboost_four` | `v1_all_threat__catboost_four` |

`raw_minimal`은 공식 원본 숫자형·범주형 열만 사용합니다. 파생 상황 피처, 과거
정답 집계, smoothing, Trackman 집계는 포함하지 않습니다. 다만 모델 실행에 필요한
숫자 변환, 무한대 제거, 범주형 인코딩, 결측 처리는 유지합니다.

## 시간 분할

```text
2019~2022: 반복 수 탐색용 학습
2023: 최적 반복 수 선택
2019~2023: 선택된 반복 수로 최종 재학습
2024: 최종 검증 한 번
```

LightGBM·XGBoost·CatBoost는 2023 early stopping으로 반복 수를 고릅니다.
HistGradientBoosting은 최대 반복까지 학습한 2023 staged prediction 중 Brier가 가장
낮은 반복을 고릅니다. 이후 네 모델 모두 2019~2023 전체로 다시 학습합니다.

이 구조에서는 2024 정답이 반복 수 선택에 사용되지 않습니다. 따라서 기존
`extended` 실험과 달리 2024를 완전한 최종 검증 세트로 해석할 수 있습니다.

## 결과 판정

- 전처리·피처 엔지니어링 효과:
  `brier_delta_vs_preprocessing = Brier(v1_all) - Brier(raw_minimal)`
- 위협도 6개 효과:
  `brier_delta_vs_comparison = Brier(v1_all_threat) - Brier(v1_all)`
- 두 값 모두 음수면 해당 단계가 개선된 것입니다.
- 최종 네 모델 순위는 `four_model_final_comparison.csv`에서 Brier 오름차순으로
  확인합니다.

모델마다 범주형을 다루는 방식은 고유 구현을 따릅니다. LightGBM과 CatBoost는
범주형을 직접 처리하고, HistGradientBoosting과 XGBoost는 학습 분할에서 구한 빈도
인코딩을 사용합니다. 이는 각 모델이 실제로 요구하는 입력 형식 차이입니다.

최종 점수 차이가 매우 작으면 단일 Brier 순위만으로 확정하지 말고 저장된 2024
예측값으로 경기 단위 bootstrap을 추가 수행해야 합니다.
