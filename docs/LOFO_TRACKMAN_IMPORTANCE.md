# HGB·CatBoost LOFO 및 Trackman 제구 피처 실험

## 1. 이번 실험이 답하는 질문

이번 실험은 194개 후보를 한꺼번에 사용하지 않는다. 아래 66개만 공통 기준선으로
사용한 뒤, 피처를 하나씩 제거해 2024년 검증 Brier score가 얼마나 나빠지는지
측정한다.

| 구분 | 개수 | 포함 내용 |
|---|---:|---|
| 기본 피처 | 47 | 공식 데이터의 경기·카운트·주자·투수/타자·as-of 누적 정보 |
| 팀원 중요도 피처 | 6 | 좌우 매치와 투수-타자 위협 상호작용 |
| 추세 피처 | 13 | 이전 1·3·5경기 차이, 현재 누적 대비 최근 차이, 투수-타자 격차 |
| 합계 | 66 | HGB와 CatBoost의 공통 기준선 |

Trackman은 위 66개에 처음부터 섞지 않는다. 먼저 66개 기준선을 확정한 후,
세이버메트릭 역할별 그룹 또는 개별 피처를 **추가**해 실제 검증 성능 개선을
측정한다. 이렇게 해야 공식 피처의 제거 중요도와 Trackman의 추가 가치를 서로
반대 방향으로 혼동하지 않는다.

두 모델에서 모두 개선된 개별 Trackman 피처 중 상위 8개는 마지막에 한 묶음으로
다시 넣어 중복 정보가 함께 들어갔을 때도 개선이 남는지 확인한다. 이 묶음은 같은
2024 검증 결과로 선정하고 다시 평가한 **탐색적 확인**이므로, 독립 검증 성능으로
과대해석하면 안 된다.

## 2. 데이터 사용과 누수 방지

- 학습: `season < 2024`
- 검증: `season == 2024`
- 공식 `asof_*` 값과 과거 1·3·5경기 값만 사용한다.
- Trackman 요약은 각 행의 시즌보다 이전 시즌의 투구만 사용한다.
- 2024년 검증값으로 iteration을 고르는 조기 종료는 사용하지 않는다. HGB와
  CatBoost 모두 모든 비교에서 모델별 고정 iteration을 사용한다.
- `test.csv`와 정답 제출 파일은 중요도 계산에 필요하지 않다.

`quick`은 파이프라인 점검용 시즌별 표본이며 결론에 사용하면 안 된다. 최종
중요도는 반드시 `full` 결과로 판단한다.

## 3. 중요도 정의

### 기본·팀원·추세 피처: LOFO

각 피처 `j`를 제거한 모델의 Brier score를 `Brier(-j)`, 66개 기준선의 점수를
`Brier(all)`이라 할 때 다음 값을 사용한다.

`LOFO 중요도(j) = Brier(-j) - Brier(all)`

- 양수: 제거했더니 나빠졌으므로 유용한 피처
- 0 부근: 이 검증 구간에서는 영향이 작거나 다른 피처가 정보를 대체
- 음수: 제거했더니 좋아졌으므로 잡음·중복·과적합 가능성

HGB와 CatBoost 양쪽에서 양수인 피처를 우선하고, 한 모델에서만 큰 값은 모델
의존 피처로 별도 표시한다. CatBoost의 내장 중요도도 저장하지만, 최종 순위는
척도가 모델마다 다른 내장 중요도가 아니라 검증 Brier 변화량으로 정한다.

### Trackman: add-back

`Trackman 개선도(k) = Brier(base 66) - Brier(base 66 + k)`

- 양수: Trackman 피처 또는 그룹을 추가했을 때 개선
- 음수: 추가했을 때 악화

따라서 LOFO와 add-back 모두 최종 결과 파일에서는 **양수가 좋다**.

## 4. Trackman 피처의 세이버메트릭 분류

| 그룹 | 핵심 값 | 제구 관점의 역할 | 해석 주의점 |
|---|---|---|---|
| 릴리스·익스텐션 재현성 | `rel_height`, `rel_side`, `extension`의 과거 평균·표준편차·변화 | 릴리스 포인트와 동작 반복성의 직접 대리변수 | 평균 수준보다 표준편차·변화량이 제구 안정성과 더 직접적 |
| 구속 유지·변동성 | `rel_speed`, `zone_speed`의 평균·표준편차·변화 | 피로, 메커니즘 변화, 구속 유지의 대리변수 | 구속이 빠르다는 것과 제구가 좋다는 것은 동일하지 않음 |
| 회전 안정성 | `spin_rate`의 평균·표준편차·변화 | 회전 품질과 재현성 보조변수 | 회전수 단독 인과 해석 금지 |
| 무브먼트 재현성 | `induced_vert_break`, `horz_break`의 평균·표준편차·변화 | 공 궤적의 반복성과 변화 대리변수 | 구종 구성이 바뀌면 평균도 함께 변할 수 있음 |
| 구종 구성 | 구종 그룹별 과거 사용률 | 투구 난이도와 접근법을 통제하는 맥락변수 | 제구를 직접 측정하지 않음 |
| 표본·매핑 신뢰도 | 매핑 순도·등급, 투구 수, 관측 시차, 결측 | 요약값의 신뢰도 보정 | 낮은 표본의 극단값을 그대로 믿지 않도록 하는 변수 |

이 분류는 KBO 데이터에서 검증할 실험 가설이다. MLB의 평균값이나 임계치를 KBO에
그대로 이식하지 않는다. 회전수, IVB, 익스텐션의 공식 정의는 각각
[MLB Spin Rate](https://www.mlb.com/glossary/statcast/spin-rate),
[MLB Induced Vertical Break](https://www.mlb.com/glossary/statcast/induced-vertical-break),
[MLB Extension](https://www.mlb.com/glossary/statcast/extension), 그리고
[TrackMan 야구 측정 항목](https://www.trackman.com/baseball/Portable-B1/what-we-track)을
참고한다.

## 5. Colab 실행

노트북 `notebooks/run_lofo_trackman_colab.ipynb`를 Colab에서 연다. 노트북은
Google Drive를 마운트하고 다음 원본을 직접 읽는다.

```text
/content/drive/MyDrive/aimers_data/train.csv
/content/drive/MyDrive/aimers_data/trackman_history.csv
```

데이터를 GitHub나 Colab 로컬 디스크에 복사할 필요가 없다. 결과와 체크포인트는
다음 위치에 저장된다.

```text
/content/drive/MyDrive/aimers_data/results/lofo_trackman/{quick|full}
```

권장 순서:

1. `MODE = "quick"`, `TRACKMAN_SCOPE = "groups"`로 전체 흐름을 점검한다.
2. `MODE = "quick"`, `TRACKMAN_SCOPE = "all"`로 개별 Trackman 실험까지 확인한다.
3. `MODE = "full"`, `TRACKMAN_SCOPE = "all"`로 최종 결과를 만든다.
4. 런타임이 끊기면 같은 설정으로 실행한다. 완료된 실험은 체크포인트를 읽고 건너뛴다.

## 6. 결과 파일

| 파일 | 내용 |
|---|---|
| `base_feature_catalog.csv` | 47+6+13 피처명과 소속 |
| `trackman_feature_catalog.csv` | 모든 Trackman 파생 피처의 세이버메트릭 분류 |
| `experiment_history_{mode}.csv` | 모델 한 번마다 즉시 기록되는 재시작 체크포인트 |
| `baseline_scores.csv` | 66개 기준선의 HGB·CatBoost 성능 |
| `lofo_importance_by_model.csv` | 모델별 66개 제거 중요도 |
| `lofo_importance_combined.csv` | 두 모델 통합 순위와 양쪽 모두 양수인지 여부 |
| `trackman_addback_by_model.csv` | 그룹·개별 Trackman 추가 실험 |
| `trackman_addback_combined.csv` | 두 모델 통합 Trackman 순위 |
| `selected_trackman_features.csv` | 두 모델 합의 상위 Trackman 피처와 선정 근거 |

최종 선택은 `full` 결과에서 다음 조건을 우선한다.

- HGB와 CatBoost 양쪽에서 LOFO 중요도가 양수
- 두 모델의 평균 Brier 변화량이 크고 방향이 일치
- Trackman은 그룹 add-back이 먼저 양수이고, 그 안의 개별 피처도 반복해서 양수
- 표본·매핑 신뢰도 피처를 함께 보고 결측 또는 낮은 매핑 품질에만 의존하지 않음

LOFO는 상관된 피처가 서로를 대체하면 각각의 중요도를 낮게 평가할 수 있다. 따라서
비슷한 피처가 많은 추세·Trackman 묶음은 개별 순위뿐 아니라 그룹 add-back 결과도
같이 해석한다.
