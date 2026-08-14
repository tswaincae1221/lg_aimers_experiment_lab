# LG Aimers 팀 공유 패키지 v1.5

이 패키지는 투수의 다음 투구 제구 성공 확률을 예측하는 자동 실험실의 코드, 전처리·피처 설명, 검증 결과, 그리고 다음 실험 설정을 함께 제공합니다.

## 먼저 읽을 파일

1. `docs/TEAM_MODELING_REPORT.md`: 전체 분석과 결론
2. `reproduced_results/verified_results.csv`: 실제 실행으로 확인된 핵심 결과
3. `config/experiments.json`: 피처·모델·실험 정의
4. `notebooks/run_experiment_lab_colab.ipynb`: Colab 실행 노트북

## 핵심 결론

- 동일 데이터 full 재실행의 최고 점수는 Trackman 전체를 제거한 HGB C 137개 피처의
  Brier Score `0.24817527`입니다.
- 실용 최소안 HGB D는 88개 피처로 전체보다 54.6% 작고 51.2% 빠르며 Brier 악화는
  `0.00008423`으로 사전 허용선 안입니다.
- CatBoost도 A–E를 실행했으며 자체 최고는 E 107개 피처 `0.24978767`입니다.
- 새 4모델 표만 보면 XGBoost가 1위지만, 그 표의 HGB는 2023 단일 시즌에서 선택된 반복 수가 4회로 줄어 과소학습되었습니다.
- 기존 HGB는 최대 350회까지 학습해 Brier, AUC, ECE 모두 새 XGBoost보다 좋았습니다.
- XGBoost에서는 타자 위협도 교차피처 6개가 Brier를 `0.00097521` 개선했습니다.
- 축소 결과와 열 선정 근거는 `docs/COMPACT_FEATURE_EXPERIMENT.md`에 정리했습니다.

## 데이터 배치

다음 파일은 용량과 대회 데이터 보호를 위해 ZIP에 포함하지 않습니다.

```text
data/train.csv
data/test.csv
data/trackman_history.csv
```

투수 ID-Trackman ID 매핑은 `resources/pitcher_trackman_mapping.csv`에 포함되어 있습니다.

## Colab 권장 실행 순서

기존 노트북을 열어 사용자 설정에서 다음과 같이 지정합니다.

```python
MODE = "quick"
PRESET = "compact"
ONLY_EXPERIMENTS = []
N_JOBS = 2
RERUN = False
```

패키지 설치 조건문에는 `compact`도 선택 설치 대상으로 포함해야 합니다.

```python
if PRESET in {"four_models", "team_next", "compact", "all"}:
    install.extend(["-r", str(REPO_DIR / "requirements-optional.txt")])
```

quick이 성공하면 점수를 모델 선택에 사용하지 말고 다음처럼 바꿉니다.

```python
MODE = "full"
PRESET = "compact"
```

직접 명령어로 실행할 수도 있습니다.

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
  --n-jobs 2
```

## `team_next` 실험 7개

| 실험 | 목적 |
|---|---|
| `v1_all__hist_gbdt_fixed350` | 재현 기준 HGB |
| `v1_all_threat__hist_gbdt_fixed350` | HGB에서 위협도 6개 효과 확인 |
| `v1_all_no_ids__hist_gbdt_fixed350` | 투수·타자 ID 의존성 제거 효과 확인 |
| `v1_all_threat__xgboost_fixed25` | 기존 XGBoost 반복 수 재현 |
| `v1_all_threat__xgboost_fixed50` | 반복 수 증가 효과 확인 |
| `v1_all_threat__xgboost_fixed100` | 반복 수 증가 효과 확인 |
| `v1_all_threat__xgboost_fixed200` | 반복 수 증가 효과 확인 |

모든 고정 반복 실험은 2024 검증 정답을 early stopping에 사용하지 않습니다.

## `compact` 실험 10개

전체, ID 제거, Trackman 제거, 핵심 88개, 핵심+대표 Trackman 107개의 다섯 단계를
HGB 350회와 CatBoost에 각각 적용합니다. CatBoost는 2023에서 반복 수를 선택하고
2019~2023 전체로 재학습합니다.

## 결과 채택 기준

1. 대회 주 지표인 Brier Score를 최우선으로 봅니다.
2. AUC는 순위 구분 능력, ECE는 확률 보정 상태를 확인하는 보조 지표입니다.
3. Brier 차이가 작으면 저장된 `validation_predictions.csv.gz`로 행 단위 paired bootstrap을 수행합니다.
4. 여러 모델의 앙상블 가중치는 2024가 아니라 과거 시즌 OOF 예측으로 결정합니다.

## 주의사항

- `raw_minimal`도 문자열·결측 때문에 모델 실행에 필요한 최소 변환은 수행합니다. 완전한 무처리 데이터가 아닙니다.
- `brier_skill_score`는 v1.4에서 표준 퍼센트 단위로 수정했습니다. 과거 결과의 `618.0`은 실제로 `0.618%`입니다.
- 검증 결과와 매핑 파일은 대회 데이터에 해당할 수 있으므로 팀 내부에서만 공유하고 공개 저장소에는 올리지 않는 것을 권장합니다.
