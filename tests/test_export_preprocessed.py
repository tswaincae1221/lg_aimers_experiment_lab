import numpy as np
import pandas as pd

from src.export_preprocessed import build_export_frame, build_feature_schema
from src.first_model_features import FeatureBundle


def sample_bundle() -> FeatureBundle:
    features = pd.DataFrame(
        {
            "numeric": pd.Series([1.0, np.nan, 3.0], dtype="float32"),
            "category": pd.Series(
                pd.Categorical(["a", "__MISSING__", "b"])
            ),
        }
    )
    return FeatureBundle(
        features=features,
        target=pd.Series([1.0, 0.0, np.nan]),
        row_ids=pd.Series(["TRAIN_1", "TRAIN_2", "TEST_1"], dtype="string"),
        seasons=pd.Series([2024, 2024, 2025], dtype="int16"),
        train_rows=np.array([0, 1]),
        test_rows=np.array([2]),
        categorical_features=["category"],
    )


def test_build_export_frame_preserves_order_and_target_rules() -> None:
    bundle = sample_bundle()
    train = build_export_frame(bundle, bundle.train_rows, include_target=True)
    test = build_export_frame(bundle, bundle.test_rows, include_target=False)

    assert train.columns.tolist() == ["row_id", "control_success", "numeric", "category"]
    assert test.columns.tolist() == ["row_id", "numeric", "category"]
    assert train["row_id"].tolist() == ["TRAIN_1", "TRAIN_2"]
    assert test["row_id"].tolist() == ["TEST_1"]


def test_feature_schema_counts_numeric_and_logical_category_missing() -> None:
    schema = build_feature_schema(sample_bundle()).set_index("feature")

    assert schema.loc["numeric", "train_missing_n"] == 1
    assert schema.loc["category", "train_missing_n"] == 1
    assert schema.loc["category", "is_categorical"]
