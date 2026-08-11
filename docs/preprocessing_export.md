# 1차 모델 전처리 전체 결과 생성 가이드

## 생성되는 데이터

기본 실행은 다음 두 파일을 만든다.

- `train_features.parquet`: `row_id` + `control_success` + 모델 피처 155개
- `test_features.parquet`: `row_id` + 모델 피처 155개

`--asof-trends`를 추가하면 V1.1의 추세·대조 피처 13개가 더해져 모델 피처가
168개가 된다.

추가로 `feature_schema.csv`, 100행 미리보기 CSV, 실행 요약 JSON이 생성된다.

시즌 `s` 행에 들어가는 성공률과 Trackman 집계는 반드시 `season < s`인 기록만 이용한다.
따라서 같은 시즌의 정답과 Trackman 기록이 피처에 섞이지 않는다.

## Windows에서 실행

프로젝트 최상위 폴더에서 다음을 실행한다.

```powershell
python -m pip install pandas numpy pyarrow
python -m src.export_preprocessed --train data/train.csv --test data/test.csv --trackman data/trackman_history.csv --mapping resources/pitcher_trackman_mapping.csv --output-dir results/preprocessed_full
```

전체 결과를 만들 때는 `--max-rows-per-season`, `--max-trackman-rows`를 붙이지 않는다.

## macOS 또는 Linux에서 실행

```bash
make setup
make preprocess-all \
  TRAIN=data/train.csv \
  TEST=data/test.csv \
  TRACKMAN=data/trackman_history.csv \
  MAPPING=resources/pitcher_trackman_mapping.csv \
  ARGS="--output-dir results/preprocessed_full"
```

## 결과 읽기

```python
import pandas as pd

train_features = pd.read_parquet("results/preprocessed_full/train_features.parquet")
test_features = pd.read_parquet("results/preprocessed_full/test_features.parquet")

print(train_features.shape)  # (1475092, 157): ID 1 + 정답 1 + 피처 155
print(test_features.shape)   # 현재 5행 test라면 (5, 156): ID 1 + 피처 155
print(train_features.head())
```

`pyarrow` 설치가 어려우면 실행 명령 끝에 `--format pickle`을 추가한다.

```python
train_features = pd.read_pickle("results/preprocessed_full/train_features.pkl.gz")
test_features = pd.read_pickle("results/preprocessed_full/test_features.pkl.gz")
```

Pickle은 신뢰할 수 있는 파일만 읽어야 한다. 이 프로젝트가 직접 생성한 파일에는 사용할 수
있지만, 출처가 불분명한 Pickle 파일을 내려받아 열면 안 된다.

## 확인해야 할 파일

- `feature_schema.csv`: 155개 피처의 순서, 자료형, 결측 수와 비율
- `preprocessing_summary.json`: 실제 행·열 수와 실행 조건
- `train_features_preview.csv`: Excel에서 빠르게 보는 학습 피처 100행
- `test_features_preview.csv`: Excel에서 빠르게 보는 테스트 피처

전체 결과는 Excel의 최대 행 수 1,048,576을 넘으므로 Excel 한 시트로 저장하지 않는다.
