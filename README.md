# LG Aimers: 투수 제구 성공 확률 예측

투수·타자·경기 상황과 Trackman 이력을 이용해 투구의 제구 성공 확률을 예측하는
프로젝트입니다. 단순한 단일 모델에서 출발해, 현재는 **45-feature CatBoost를 안정적인
anchor로 유지**하면서,
시간 누수 없는 `as-of` 피처, F/R 리그 차이, cold-start 투수 prior, calibration과
선택적 overlay를 연구하고 있습니다.

모델 선택에는 2019~2023을 사용하고, **2024는 최종 holdout 평가에만 사용**합니다.
주 평가지표는 Brier Score이며 Log Loss, AUC와 calibration을 함께 확인합니다.

## 현재 프로젝트 상태

| 항목 | 현재 상태 |
|---|---|
| Anchor | 45-feature CatBoost (`new_baseline_final`, `M0_final45`) |
| 2024 holdout Brier | 0.247442816 (raw anchor prediction) |
| 검증 원칙 | 각 행에서 당시 알 수 있었던 정보만 사용하는 leakage-safe as-of 방식 |
| 주요 연구 | F/R regime, temporal shift, cold-start prior, calibration, selective overlay |
| Trackman | 경기 fingerprint 기반 deterministic 선수 매핑과 이전 시즌 집계 |

2026년 8월 24일 이후의 실험과 채택·기각 판단은
[진척사항 문서](docs/progress-2026-08-24-onward.md)에 정리했습니다.

## 프로젝트는 어떻게 발전했나

이 저장소는 처음부터 완성된 파이프라인으로 시작하지 않았습니다. 성능 숫자 하나를
높이는 것보다 **재현 가능하고, 시간 누수가 없으며, 개선과 악화를 설명할 수 있는
실험 체계**를 만드는 방향으로 단계적으로 발전했습니다.

| 단계 | 출발점과 문제 | 변경한 구조 | 결과와 판단 |
|---|---|---|---|
| 1. 초기 모델 | 공식 학습 데이터와 기본 피처로 제구 성공 여부를 예측 | 기본 전처리와 LightGBM 중심의 V1 모델 구성 | 이후 실험을 비교할 최초 기준선 확보 |
| 2. 실험실화 | 노트북별 전처리·평가 방식이 달라 결과 비교가 어려움 | 설정 기반 `피처 묶음 × 모델` 실행기, 캐시, 리더보드, 자동 보고서 도입 | 같은 검증 조건에서 실험을 반복·재개할 수 있게 됨 |
| 3. 공정한 모델 비교 | 모델과 피처 변경 효과가 섞여 원인을 구분하기 어려움 | 원본 최소처리 → 전체 피처 → 위협도 교차피처의 3단계를 네 모델에 동일 적용 | 전처리 효과와 모델 효과를 분리해 비교 |
| 4. 검증 원칙 강화 | 전체 기간 집계나 미래 정보가 과거 예측에 섞일 위험 | 모든 누적 통계를 `as-of`로 계산하고, 2019~2023만 모델 선택에 사용 | 2024를 손대지 않은 최종 holdout으로 고정 |
| 5. 강한 anchor 확립 | 많은 피처가 항상 안정적인 개선으로 이어지지 않음 | 45-feature CatBoost를 `anchor`로 고정하고 신규 아이디어는 제거 실험으로 검증 | 2024 raw holdout Brier `0.247442816` 확보 |
| 6. Trackman 연결 고도화 | 공식 데이터와 Trackman 선수 ID가 직접 연결되지 않음 | 경기 fingerprint와 투구 순서를 이용한 deterministic 매핑, 이전 시즌만의 집계 구현 | 591명 매핑, 평균 purity `0.999972`, 행 연결률 92.7% |
| 7. 세분화된 보정 | 신인 투수와 F/R 리그 변화에서 하나의 전역 모델이 취약 | cold-start KNN prior, F-specialist, portable Ridge residual, selective overlay 연구 | anchor를 보존하면서 필요한 집단만 보정하는 구조로 발전 |

초기 구현과 당시 README는 [`legacy` 브랜치](../../tree/legacy)에 그대로 보존했습니다.
현재 `main`은 그 실험실을 기반으로 최근 연구 코드와 검증 원칙을 통합한 버전입니다.

## 현재 구조

```text
공식 train/test + Trackman 이력
              │
              ▼
  누수 없는 전처리와 as-of 집계
              │
      ┌───────┴────────┐
      ▼                ▼
경기 fingerprint    피처 블록 생성
선수 ID 매핑        상황·구종·신뢰도·추세
      └───────┬────────┘
              ▼
    45-feature CatBoost anchor
              │
      ┌───────┴──────────────┐
      ▼                      ▼
cold-start prior       선택 집단 residual/overlay
      └───────┬──────────────┘
              ▼
  calibration 및 2024 holdout 평가
              │
              ▼
 리더보드·그래프·실험 보고서 자동 생성
```

핵심 역할은 다음처럼 나뉩니다.

| 영역 | 역할 | 주요 위치 |
|---|---|---|
| 실험 제어 | 설정을 읽어 피처·모델 조합 실행, 캐시와 재시작 관리 | `src/experiment_runner.py`, `config/experiments.json` |
| 피처 생성 | 시점 안전한 추세·상황·구종·신뢰도 피처 구성 | `src/experiment_features.py` |
| Trackman | 경기 복원, 결정적 선수 매핑, 이전 시즌 물리 피처 집계 | `research/trackman_mapping/` |
| 신규 투수 | 관측 이력이 적은 투수의 계층적 prior 생성 | `src/cold_start.py` |
| 선택 보정 | anchor 오차를 특정 집단에서만 보정하고 이식 가능한 계수 저장 | `src/ridge_residual.py` |
| 평가·기록 | Brier·Log Loss·AUC·calibration과 전후 비교 자료 생성 | `results/experiment_lab/` 구조 |

## 변화 과정에서 유지한 원칙

- 현재 행의 정답이나 미래 시즌 정보는 피처 생성과 선수 매핑에 사용하지 않습니다.
- 2024 holdout은 최종 확인용이며, 후보 선택과 튜닝은 2019~2023 안에서 수행합니다.
- 새 아이디어는 anchor 전체를 즉시 교체하지 않고 제거 실험과 집단별 비교로 검증합니다.
- 한 시즌에서만 좋아진 결과보다 여러 시점에서 방향이 유지되는지를 우선합니다.
- 성능이 악화된 실험도 숨기지 않고 보류·기각 근거로 기록합니다.
- 대회 원본 데이터와 행 단위 예측, 학습 모델은 공개 저장소에 올리지 않습니다.

한 번의 실행으로 다음을 자동 처리합니다.

- V1 전처리와 Trackman 집계를 한 번만 수행
- 설정 파일에 적힌 피처 묶음 × 모델 조합을 순차 실행
- Brier Score, Log Loss, AUC, BSS, calibration 오차, 시간 기록
- V1 LightGBM 대비 개선량과 개선률 계산
- 실행별 예측값, calibration, 피처 중요도, 오류 보존
- 전체 리더보드, 개선폭, 점수-시간 그래프와 Markdown 보고서 생성
- 완료 조합 자동 건너뛰기와 중단 후 이어서 실행
- 공식 원본 최소처리 대 전체 피처 엔지니어링의 전·후 비교
- HistGradientBoosting·XGBoost·CatBoost·LightGBM 공정 비교 프리셋

## 가장 쉬운 실행

1. 이 저장소 내용을 GitHub 비공개 저장소에 올립니다.
2. `notebooks/run_experiment_lab_colab.ipynb`를 Colab에 업로드합니다.
3. 노트북 첫 설정 셀의 GitHub 계정·저장소 기본값을 확인합니다.
4. `MODE="quick"`, `PRESET="four_models"`로 위에서부터 실행합니다.
5. quick 실행이 통과하면 `MODE="full"`로 바꿔 12개 조합을 정식 비교합니다.

노트북의 기본 데이터 폴더는 다음과 같습니다.

```text
/content/drive/MyDrive/aimers_data/
├─ train.csv
├─ test.csv
└─ trackman_history.csv
```

## 제공 피처 블록

| 블록 | 개수 | 내용 |
|---|---:|---|
| `asof_trend` | 13 | 1·3·5경기 추세, 투수-타자 차이, strike-ball 차이 |
| `situation` | 11 | base-out, count-base-out, LI·후반·접전 압박 상호작용 |
| `pitchmix` | 8 | 구종 구성 entropy, 주 구종 비율, 구종 간 사용률 차이 |
| `reliability` | 7 | 투수·타자·Trackman 표본 신뢰도와 정보원 개수 |
| `batter_threat_interactions` | 6 | 좌우 일치와 타자 위협도 대리변수×투수 제구 성향 |

모든 추가 블록은 현재 행의 공식 사전 정보나 `season < 예측 시즌`으로 만들어진
V1 피처만 사용합니다. 현재 투구의 정답은 사용하지 않습니다.

타자 위협도 블록은 `1 - asof_batter_success_rate`를 **제구 성공을 어렵게 하는 상대성의
대리변수**로 사용합니다. 홈런·장타 능력을 직접 측정한 값으로 해석하면 안 됩니다.
정확한 정의와 제거 실험은 `docs/batter_threat_interactions.md`에 정리돼 있습니다.

## 4개 모델 비교

`PRESET="four_models"`는 네 모델에 같은 세 가지 데이터 조건을 적용해 총 12개
실험을 실행합니다.

| 단계 | 내용 | 비교 목적 |
|---|---|---|
| `raw_minimal` | 공식 원본 열 + 실행 필수 자료형·결측 처리 | 피처 엔지니어링 전 |
| `v1_all` | V1 + asof·상황·구종·신뢰도 피처 | 피처 엔지니어링 후 |
| `v1_all_threat` | `v1_all` + 타자 위협도 교차피처 6개 | 교차피처 추가 효과 |

`raw_minimal`도 문자열을 숫자 모델에 넣기 위한 인코딩과 비정상값 처리는 수행합니다.
따라서 이는 데이터 정제 유무가 아니라 **수작업 피처 엔지니어링 전·후 비교**입니다.

네 모델 비교에서는 2019~2022로 학습하고 2023으로 최적 반복 수를 고른 뒤,
2019~2023 전체를 그 반복 수로 재학습합니다. 2024는 모델 선택에 사용하지 않고
마지막 Brier 평가에만 사용합니다. 상세 설계는 `docs/four_model_comparison.md`를
참고합니다.

## 제공 모델

| 모델 설정 | 기본 프리셋 | 특징 |
|---|---|---|
| Constant | starter | 학습 성공률 기준선 |
| LightGBM base | starter | 기존 V1과 동일한 기준 모델 |
| LightGBM regularized | extended | 더 강한 규제와 큰 leaf 최소 표본 |
| LightGBM wide | extended | 더 많은 상호작용 탐색 |
| Logistic Regression | extended | 빈도 인코딩 선형 기준선 |
| HistGradientBoosting | extended | scikit-learn 히스토그램 부스팅 |
| ExtraTrees | extended | 배깅 기반 비선형 비교군 |
| XGBoost | four_models, all | 선택 설치 모델 |
| CatBoost | four_models, all | 선택 설치, 범주형 직접 처리 |

`four_models`는 HistGradientBoosting·XGBoost·CatBoost·LightGBM만 같은 실험
행렬로 비교합니다. `starter`는 피처 블록의 효과를 같은 LightGBM으로 비교하고,
`extended`는 기존 비교 모델과 제거 실험까지 넓히며, `all`은 모든 실험을 포함합니다.

## 직접 실행

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt \
  -r requirements-optional.txt
python -m pytest -q

python -m src.experiment_runner \
  --config config/experiments.json \
  --train data/train.csv \
  --test data/test.csv \
  --trackman data/trackman_history.csv \
  --mapping resources/pitcher_trackman_mapping.csv \
  --output-dir results/experiment_lab/quick \
  --mode quick \
  --preset four_models \
  --validation-season 2024 \
  --n-jobs 2
```

정식 비교는 출력 폴더와 모드만 바꿉니다.

```bash
python -m src.experiment_runner \
  --config config/experiments.json \
  --train data/train.csv \
  --test data/test.csv \
  --trackman data/trackman_history.csv \
  --mapping resources/pitcher_trackman_mapping.csv \
  --output-dir results/experiment_lab/full \
  --mode full \
  --preset four_models \
  --validation-season 2024 \
  --n-jobs 4
```

특정 조합만 실행하려면 `--only`를 사용합니다.

```bash
python -m src.experiment_runner ... \
  --only v1__lgbm_base v1_asof__lgbm_base
```

## 실험 추가

`config/experiments.json`은 세 부분으로 나뉩니다.

1. `feature_sets`: 사용할 블록과 제거 패턴
2. `models`: 모델 종류와 하이퍼파라미터
3. `experiments`: 피처 묶음과 모델의 실제 조합

예를 들어 새 LightGBM 설정을 비교하려면 `models`에 설정을 추가한 뒤
`experiments`에 다음 형태의 항목을 추가합니다.

```json
{
  "name": "v1_asof__lgbm_custom",
  "feature_set": "v1_asof",
  "model": "lgbm_custom",
  "presets": ["extended"]
}
```

새 row-wise 피처는 `src/experiment_features.py`에 builder를 추가하고
`FEATURE_BLOCKS`에 등록합니다. 모델 실행 코드를 고치지 않고 설정에서 조합할 수
있습니다.

## 결과 구조

```text
results/experiment_lab/full/
├─ experiment_history.csv          # 성공·실패를 포함한 누적 기록
├─ leaderboard.csv                 # 동일 데이터·모드의 최신 결과
├─ experiment_report.md            # 자동 해석 보고서
├─ feature_catalog.csv             # 추가 피처 목록
├─ leaderboard_brier.png
├─ improvement_vs_baseline.png
├─ improvement_vs_comparison.png  # 동일 모델·기존 피처 대비 순수 변화
├─ preprocessing_before_after.png # 원본 최소처리 대비 전체 피처 효과
├─ four_model_stage_comparison.csv
├─ four_model_final_comparison.csv
├─ four_model_final_comparison.png
├─ score_vs_time.png
├─ cache/                          # Trackman 집계 재사용
└─ runs/
   └─ 실행시각__실험명__설정해시/
      ├─ metrics.json
      ├─ resolved_config.json
      ├─ feature_report.json
      ├─ validation_predictions.csv.gz
      ├─ calibration_bins.csv
      ├─ calibration_curve.png
      ├─ prediction_distribution.png
      ├─ feature_importance.csv
      ├─ feature_importance_top30.png
      └─ error.txt                 # 실패한 실험에만 생성
```

판정은 `brier_delta_vs_baseline`을 보면 됩니다.

- 음수: V1 LightGBM보다 개선
- 0: 동일
- 양수: 악화

4개 모델 비교에서는 다음 열을 함께 봅니다.

- `brier_delta_vs_preprocessing`: 같은 모델의 `v1_all - raw_minimal`; 음수면 전체
  피처 엔지니어링이 개선
- `brier_delta_vs_comparison`: 같은 모델의 `v1_all_threat - v1_all`; 음수면 위협도
  교차피처 6개가 개선
- `four_model_final_comparison.csv`: `v1_all_threat` 조건의 네 모델 최종 순위

quick 모드는 시즌당 일부 행만 쓰므로 점수 비교용이 아닙니다. 최종 채택은 반드시
full 모드 결과로 결정합니다.

## 대회 데이터 보호

대회 원본 데이터, 전처리 결과, 실행 결과와 학습 모델은 저장소에 포함하지 않습니다.
공개 저장소에는 재현 가능한 코드와 집계된 실험 결론만 포함합니다.
