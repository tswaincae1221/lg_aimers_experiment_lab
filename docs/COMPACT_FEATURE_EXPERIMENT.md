# 핵심 피처 축소 실험

## 결론

- 성능 우선 추천: HGB C, Trackman 전체 제거 137개, Brier `0.24817527`
- 단순성 우선 추천: HGB D, 핵심 88개, Brier `0.24829631`
- CatBoost 자체 최고: E, 핵심 88개+대표 Trackman 19개, Brier `0.24978767`

HGB C는 전체 194개보다 피처를 29.4% 줄이고 Brier를 `0.00003681` 개선했으며
학습 시간도 약 30.7% 줄었습니다. HGB D는 피처를 54.6%, 시간을 51.2% 줄이면서
전체 대비 악화가 사전 허용선 `0.0001` 안인 `0.00008423`입니다.

## 검증 설계

- 학습: 2019~2023, 1,221,585행
- 검증: 2024, 253,507행
- 모델 1: HistGradientBoosting, 350회 고정
- 모델 2: CatBoost, 2023에서 반복 수 선택 후 2019~2023 재학습
- seed: 42
- 주 지표: Brier Score, 낮을수록 좋음
- 실행 환경: Python 3.12.13, pandas 2.2.3, scikit-learn 1.8.0, CatBoost 1.2.8

2024는 마지막 평가에만 사용했습니다. `quick`은 코드 검증용이며 아래 표는 전부
`full` 결과입니다.

## A–E 단계와 결과

| 단계 | 구성 | 피처 | HGB Brier | HGB 시간(초) | CatBoost Brier | CatBoost 시간(초) |
|---|---|---:|---:|---:|---:|---:|
| A | 전체 V1+row-wise | 194 | 0.24821208 | 91.99 | 0.24991898 | 145.36 |
| B | A에서 선수 ID 2개 제거 | 192 | 0.24818476 | 104.16 | 0.24992153 | 132.86 |
| C | A에서 Trackman 전체 제거 | 137 | **0.24817527** | 63.73 | 0.24990561 | 125.28 |
| D | 핵심 프로필 | 88 | 0.24829631 | **44.93** | 0.24984364 | 58.73 |
| E | D+대표 Trackman 19개 | 107 | 0.24829042 | 59.65 | **0.24978767** | **59.16** |

CatBoost는 D에서 2회, E에서 3회만 선택되어 HGB보다 과소학습됐습니다. 따라서 현재
CatBoost 결과는 최종 모델 추천보다 축소 방향의 교차 확인용입니다.

연속 500행 블록 paired bootstrap 10,000회에서 HGB C-A Brier 차이의 95% 구간은
`[-0.000117, 0.000050]`으로 0을 포함했습니다. C의 개선이 확정적이라고 주장하기보다
성능 손실 증거 없이 피처와 시간을 줄였다고 해석하는 것이 안전합니다.

## 핵심 88개 선정 원칙

1. 경기 문맥 24개: 카운트·아웃·주자·점수·LI·좌우 매치업
2. 공식 `asof` 22개: 투수·타자 제구율, 최근 경기, 구종 구성과 표본 로그
3. 누수 방지 이력 19개: `season < 예측 시즌`의 career·직전 시즌·최근 구간
4. row-wise 상호작용 23개: 추세, 투수-타자 차이, 압박, 표본 신뢰도

선수·팀 ID, Trackman 전체 원자료, 같은 정보를 반복하는 파생치는 D에서 제외했습니다.
E는 여기에 매핑 품질, 구속·회전·무브먼트·구종 구성 대표 Trackman 19개만 더합니다.
정확한 열 목록은 `src/compact_feature_profiles.py`에 고정되어 있으며, 상위 전처리
변경으로 열이 사라지면 조용히 넘어가지 않고 실행을 실패시킵니다.

## 실행

```bash
python -m src.experiment_runner \
  --config config/experiments.json \
  --train data/train.csv \
  --test data/test.csv \
  --trackman data/trackman_history.csv \
  --mapping resources/pitcher_trackman_mapping.csv \
  --output-dir results/compact/full \
  --mode full \
  --preset compact \
  --validation-season 2024 \
  --n-jobs 4
```

총 10개 조합이 실행됩니다. 요약 수치는
`reproduced_results/compact_full_results.csv`, paired 비교는
`reproduced_results/compact_paired_block_bootstrap.csv`에 보존했습니다. 원본 데이터와
행 단위 검증 예측은 공개 저장소에 포함하지 않습니다.
