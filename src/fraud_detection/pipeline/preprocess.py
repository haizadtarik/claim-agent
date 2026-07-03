import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.data.features import (
    DEFAULT_DROP_COLUMNS,
    TARGET_COLUMN,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data by dropping irrelevant columns and handling placeholders."""
    cleaned = df.replace("?", np.nan)
    # Drop columns starting with _c
    cleaned = cleaned.drop(
        columns=[c for c in cleaned.columns if c.startswith("_c")], errors="ignore"
    )
    # Drop default identifiers
    cleaned = cleaned.drop(columns=list(DEFAULT_DROP_COLUMNS), errors="ignore")
    return cleaned


def build_preprocessor(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[ColumnTransformer, list[str]]:
    """
    Build a scikit-learn ColumnTransformer for preprocessing numeric and categorical features.
    Returns the preprocessor and a list of feature names.
    """
    cleaned_df = clean_data(df)

    candidate_cols = [c for c in cleaned_df.columns if c != target_column]

    numeric_features = []
    categorical_features = []

    for col in candidate_cols:
        series = cleaned_df[col]
        # Identify numeric vs categorical similarly to features.py
        if pd.api.types.is_numeric_dtype(series):
            numeric_features.append(col)
        else:
            numeric_attempt = pd.to_numeric(series, errors="coerce")
            numeric_coverage = (
                float(numeric_attempt.notna().mean()) if len(series) else 0.0
            )
            if numeric_coverage >= 0.8:
                numeric_features.append(col)
            else:
                categorical_features.append(col)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop anything else that might have slipped through
    )

    return preprocessor, numeric_features + categorical_features
