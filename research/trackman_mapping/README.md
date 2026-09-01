# Deterministic Trackman mapping

공식 train과 Trackman 데이터의 직접적인 선수 키가 없을 때, 경기 fingerprint와 정렬된
투구 전 상태열만으로 선수 crosswalk를 만드는 연구 코드입니다. 타깃은 매핑에 사용하지
않으며, 역사적 holdout에서는 평가 시즌 이전 경기만 사용합니다.

실행 순서:

```bash
python research/trackman_mapping/deterministic_trackman_mapping.py
python research/trackman_mapping/build_train_trackman_combined.py
python research/trackman_mapping/build_holdout_crosswalk.py
python research/trackman_mapping/evaluate_mapping_ablation.py
```

원본 `data/train.csv`와 `data/trackman_history.csv`가 로컬에 있어야 합니다. 생성되는
crosswalk, overlay, 예측값과 모델 파일은 공개 저장소에 커밋하지 않습니다.
