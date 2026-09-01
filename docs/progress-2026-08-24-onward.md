# 2026-08-24 이후 연구 진척사항

이 문서는 로컬 프로젝트에서 진행한 실험을 포트폴리오용으로 정리한 기록입니다.
대회 원본 데이터, 행 단위 예측, 모델 파일과 제출 ZIP은 포함하지 않습니다.

## 기준선과 검증 원칙

- 45-feature CatBoost를 안정적인 anchor로 유지했습니다.
- 최근 확인한 2024 holdout raw Brier Score는 `0.247442816`입니다.
- 모든 통계 피처는 현재 투구보다 이전 정보만 사용하는 as-of 방식으로 계산합니다.
- 2024는 최종 평가에만 사용하고, 후보 선택과 튜닝에는 사용하지 않는 것을 원칙으로 합니다.
- Brier Score를 주 지표로 사용하며 BSS, AUC와 calibration을 보조로 확인합니다.

## Trackman deterministic mapping

`train`과 Trackman의 선수 ID가 직접 연결되지 않는 문제를 해결하기 위해 타깃을 사용하지
않는 결정적 매핑 절차를 구현했습니다.

1. 연속된 train 행에서 경기를 복원합니다.
2. 양쪽 데이터의 경기·매치업 정보와 투구 전 상태열을 fingerprint로 만듭니다.
3. 양쪽에서 유일하며 상태열 전체가 일치하는 경기만 연결합니다.
4. 정렬된 투구 위치를 이용해 Trackman 투수 ID와 공식 투수 ID를 투표 방식으로 매핑합니다.
5. 홀드아웃에서는 해당 시즌보다 이전 경기만으로 매핑과 피처를 구성합니다.

확인된 결과는 다음과 같습니다.

- 정확 일치 경기: 2,418
- 매핑 투수: 591명
- 투수 매핑 평균 purity: 0.999972
- Trackman 행 연결률: 92.7% (`1,662,659 / 1,793,078`)
- 투수×시즌 Trackman 테이블: 2,045행 × 23열
- 2024 holdout 통제 실험 Brier: legacy `0.24814055` → deterministic `0.24813205`

Brier 개선 폭은 `-0.00000850`으로 작습니다. 따라서 큰 모델 성능 향상보다는 매핑의
정확성·설명 가능성과 재현성을 확보한 작업으로 해석합니다. 구현은
[`research/trackman_mapping`](../research/trackman_mapping/)에 있습니다.

## Cold-start 투수 처리

신인 또는 관측 이력이 거의 없는 투수에게 고정 리그 평균만 넣는 대신 다음 우선순위를
갖는 정적 prior를 구현했습니다.

1. 직전 시즌 Trackman 프로필의 KNN prior
2. 관측된 신인 투수 성공률 중앙값
3. 해당 시즌 리그 중앙값
4. 최종 고정 fallback

자기 자신의 결과를 이웃 타깃으로 사용하지 않으며, 예측 시즌 `Y`의 행에는 `Y-1`
이전 Trackman 프로필과 결과만 허용합니다. 구현은 [`src/cold_start.py`](../src/cold_start.py),
연결 방법은 [`cold_start_integration.md`](cold_start_integration.md)에 있습니다.

## Residual correction과 overlay

anchor 확률 자체를 대체하기보다, 선별한 집단에서만 작은 residual correction을 적용하는
구조를 실험했습니다. 학습 시에는 scikit-learn Ridge를 사용할 수 있지만 제출 시에는
NumPy와 pandas만으로 추론할 수 있도록 전처리와 계수를 plain array/dict로 저장하는
portable bundle을 구현했습니다. 코드는 [`src/ridge_residual.py`](../src/ridge_residual.py)에
있습니다.

Anchor + F-specialist overlay에서는 다음을 필수 검증 항목으로 두었습니다.

- non-F 행의 anchor 예측이 정확히 보존되는가
- F 행에만 specialist 또는 보정이 적용되는가
- subtype 복원 순서가 제출 행 순서와 일치하는가
- 2024 holdout 학습에 2024 타깃이나 사후 Trackman 정보가 들어가지 않는가

## EDA와 실험에서 얻은 결론

- 2019~2024 타깃 비율은 고정적이지 않으며 2023년 전후 F 리그에서 regime 변화가 큽니다.
- Trackman 측정값도 2021~2022 전후 분포 이동 가능성이 있어 연도 조건 검증이 필요합니다.
- 오래된 투수를 제거하거나 강하게 down-weight한 실험은 기준선보다 악화되어 보류했습니다.
- 0-2 count 특화 피처는 breaking ball에서 개선 신호가 있었지만 offspeed에서 악화되어
  일괄 적용하지 않았습니다.
- 4월·8월·9월·10월 성능이 상대적으로 약했지만 월 자체를 하드코딩한 피처는 안정적으로
  개선되지 않았습니다.
- 2023 개발 순위와 2024 최종 순위가 자주 뒤집혀 단일 연도 튜닝보다 여러 시점의 안정성
  검증을 우선하게 되었습니다.

## 현재 우선순위

1. 45-feature anchor 재현성 유지
2. F/R regime 분리 검증
3. 시점별 distribution shift와 calibration 진단
4. cold-start Trackman prior의 2024 holdout 검증
5. 특정 집단만 개선하는 selective overlay의 안정성 확인

## 보류하거나 기각한 방향

- stale pitcher 일괄 제거
- 설명 없이 피처 수만 늘리는 접근
- 특정 월 자체를 직접 하드코딩
- 한 시즌에서만 좋아진 calibration 또는 overlay를 최종안으로 채택
- 미래 시즌 Trackman이나 전체 데이터 매핑을 과거 holdout에 사용하는 방식
