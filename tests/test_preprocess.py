import pandas as pd
from sklearn.compose import ColumnTransformer
from fraud_detection.pipeline.preprocess import build_preprocessor, clean_data
from fraud_detection.data.features import TARGET_COLUMN


def test_clean_data():
    raw_data = pd.DataFrame(
        {
            "_c39": [1, 2, 3],
            "policy_number": [123, 456, 789],
            "age": [25, 30, "?"],
            "fraud_reported": ["Y", "N", "Y"],
        }
    )

    cleaned = clean_data(raw_data)

    # Check dropped columns
    assert "_c39" not in cleaned.columns
    assert "policy_number" not in cleaned.columns

    # Check "?" replacement
    assert pd.isna(cleaned.loc[2, "age"])


def test_build_preprocessor():
    raw_data = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "incident_type": ["Collision", "Theft", "Collision"],
            "fraud_reported": ["Y", "N", "Y"],
        }
    )

    preprocessor, feature_names = build_preprocessor(raw_data, TARGET_COLUMN)

    assert isinstance(preprocessor, ColumnTransformer)
    assert len(feature_names) == 2
    assert "age" in feature_names
    assert "incident_type" in feature_names
