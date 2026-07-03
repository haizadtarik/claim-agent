import pandas as pd
from fraud_detection.data.load import load_data


def test_load_data():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "fraud_reported" in df.columns
    # Basic shape check, should have 1000 rows typically for this dataset
    assert len(df) > 0
