# V1.1 `asof` 추세 피처 실험

## 목적

V1의 모델 파라미터와 팀 공통 검증 방식인 **2019~2023 학습 → 2024 검증**을
고정한 채, 공식 `asof_` 열에서 만든 13개 추세·대조 피처만 추가해 실제 개선
여부를 확인한다. V1은 155개, V1.1은 168개 피처를 사용한다(Trackman을 포함한
전체 구성 기준).

## 추가 피처

| 묶음 | 피처 | 의미 |
|---|---|---|
| 성공률 추세 | `success_trend_1v3`, `1v5`, `3v5` | 짧은 구간과 긴 구간의 성공률 차이 |
| 가운데 비율 추세 | `middle_trend_1v3`, `1v5`, `3v5` | 최근 가운데 몰림의 상승·하락 |
| 평소 대비 최근 | `success_vs_prev5`, `middle_vs_prev5` | 전체 `asof` 상태와 최근 5경기의 차이 |
| 투수·타자 대조 | `pitcher_batter_success_gap`, `middle_gap` | 투수 상태와 상대 타자 상태의 차이 |
| 판정 구성 | `strike_ball_gap` | 스트라이크 비율과 볼 비율의 차이 |
| 가용성 | `success_trend_source_n`, `middle_trend_source_n` | 1·3·5경기 값 중 존재하는 수(0~3) |

실제 열 이름에는 모두 `asof_` 접두사가 붙는다. 차이를 계산할 두 값 중 하나라도
결측이면 결과도 결측으로 유지한다. 0이라는 실제 추세와 결측을 혼동하지 않도록
`source_n` 두 개를 함께 넣는다.

현재 행에 공식 제공된 사전 집계 열만 조합하며 `control_success`를 읽지 않는다.
기존 과거 제구력·Trackman은 계속 `season < s` 규칙을 사용한다.

## Colab에서 실행

데이터 복사와 패키지 설치가 끝난 상태에서 저장소 폴더로 이동한다.

```python
%cd /content/aimers-main
```

먼저 V1 기준 모델을 같은 조건으로 다시 실행한다.

```python
!python -u -m src.first_model \
  --train data/train.csv \
  --trackman data/trackman_history.csv \
  --mapping resources/pitcher_trackman_mapping.csv \
  --output-dir results/v1_baseline \
  --cv-seasons 2024 \
  --n-jobs 2
```

다음으로 `--asof-trends`만 추가해 V1.1을 실행한다.

```python
!python -u -m src.first_model \
  --train data/train.csv \
  --trackman data/trackman_history.csv \
  --mapping resources/pitcher_trackman_mapping.csv \
  --output-dir results/v1_1_asof_trend \
  --cv-seasons 2024 \
  --asof-trends \
  --n-jobs 2
```

두 실행 모두 `--max-rows-per-season`과 `--max-trackman-rows`를 넣지 않는다. 표본
빠른 점검 점수는 정식 비교에 쓰지 않는다.

## 결과 비교

```python
!python -m src.compare_experiments \
  --baseline results/v1_baseline/metrics.csv \
  --candidate results/v1_1_asof_trend/metrics.csv \
  --output results/asof_trend_comparison.csv
```

`brier_delta`는 `V1.1 - V1`이므로 **음수이면 개선**이다. 채택 조건은 다음과 같다.

1. 2024 Brier Score가 V1보다 낮아질 것.
2. `feature_importance.csv`에서 추가 피처가 실제로 사용되는지 확인할 것.
3. 평균 예측 확률과 실제 성공률의 차이가 더 커지지 않을 것.
4. AUC와 Log Loss가 크게 악화하지 않을 것.

결과가 비슷하거나 나빠지면 13개 전체를 유지하지 말고, 중요도가 있는 묶음만 다시
분리해 재실험한다.
