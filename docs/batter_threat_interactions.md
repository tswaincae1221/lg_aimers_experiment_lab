# 타자 위협도 교차피처 실험

## 목적

기존 최고 후보인 `V1 + asof 추세 13개`에 투수·타자의 사전 상태를 결합한 교차피처
6개를 추가합니다. 모든 비율은 현재 투구 전에 제공된 공식 `asof_` 열만 사용하며,
현재 행의 `control_success`는 읽지 않습니다.

## 정의

타자 위협도 대리변수는 다음과 같습니다.

```text
batter_control_threat_proxy = 1 - asof_batter_success_rate
```

이는 홈런·장타 능력의 직접 지표가 아니라, 해당 타자를 상대할 때 투수의 제구 성공이
어려웠던 정도를 나타내는 대리변수입니다.

| 피처 | 계산식 |
|---|---|
| `hand_match` | 투수 손과 타자 손이 같으면 1, 다르면 0, 결측이면 결측 |
| `success_interact` | `asof_pitcher_success_rate × asof_batter_success_rate` |
| `reverse_x_middle` | `asof_pitcher_reverse_rate × asof_pitcher_middle_rate` |
| `threat_x_reverse` | `batter_control_threat_proxy × asof_pitcher_reverse_rate` |
| `threat_x_middle` | `batter_control_threat_proxy × asof_pitcher_middle_rate` |
| `threat_x_ball` | `batter_control_threat_proxy × asof_pitcher_ball_rate` |

공식 비율이 0~1 범위를 벗어나면 그 값은 유효한 확률로 보지 않고 결측으로 처리합니다.
한 소스라도 결측이면 해당 곱도 결측으로 유지합니다.

## 자동 실험

`full + starter`는 같은 LightGBM 설정으로 아래를 비교합니다.

| 실험 | 추가 피처 수 | 비교 목적 |
|---|---:|---|
| `v1_asof__lgbm_base` | 0 | 직접 비교 기준 |
| `v1_asof_threat_ball_only__lgbm_base` | 1 | 가장 유망한 피처 단독 효과 |
| `v1_asof_threat_no_ball__lgbm_base` | 5 | `threat_x_ball` 없이도 개선되는지 |
| `v1_asof_threat__lgbm_base` | 6 | 전체 묶음 효과 |

`extended`에서는 Logistic Regression과 HistGradientBoosting도 각각
`v1_asof` 대 `v1_asof_threat`로 비교합니다.

판정은 `brier_delta_vs_comparison`을 우선합니다.

- 음수: 같은 모델에서 교차피처 추가 후 개선
- 양수: 같은 모델에서 교차피처 추가 후 악화
- 전체 6개가 ball-only보다 좋음: 나머지 피처도 추가 기여 가능
- 전체 6개와 ball-only가 비슷함: 단순한 ball-only 구성이 더 효율적일 수 있음
- no-ball만 악화: `threat_x_ball`이 개선의 중심이라는 근거

2024 Brier 차이가 매우 작으면 단일 점수만으로 확정하지 말고 저장된 2024 예측값을
동일 행끼리 비교하거나 경기 단위 bootstrap으로 안정성을 추가 확인합니다.
