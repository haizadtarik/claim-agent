"""Feature selection utilities for the insurance claim fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.data.load import load_data

TARGET_COLUMN = "fraud_reported"
DEFAULT_TOP_K = 12
MISSING_TOKEN = "__MISSING__"

DEFAULT_DROP_COLUMNS = {
    "policy_number",
    "policy_bind_date",
    "incident_date",
    "incident_location",
    "insured_zip",
}


@dataclass(frozen=True)
class FeatureSelectionResult:
    ranking: pd.DataFrame
    selected_features: list[str]


def clean_for_selection(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace("?", np.nan)
    cleaned = cleaned.drop(
        columns=[column for column in cleaned.columns if column.startswith("_c")],
        errors="ignore",
    )
    return cleaned


def _to_binary_target(target: pd.Series) -> pd.Series:
    mapping = {"N": 0, "Y": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    mapped = target.map(mapping)
    if mapped.isna().any():
        values = sorted({value for value in target.dropna().unique().tolist()})
        raise ValueError(f"Unsupported target values for feature selection: {values}")
    return mapped.astype(int)


def _is_date_like(column_name: str, series: pd.Series) -> bool:
    lowered = column_name.lower()
    if "date" in lowered:
        return True
    if not pd.api.types.is_object_dtype(series):
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def _is_identifier_like(column_name: str, series: pd.Series, row_count: int) -> bool:
    if column_name in DEFAULT_DROP_COLUMNS:
        return True
    lowered = column_name.lower()
    if any(token in lowered for token in ("id", "number", "zip", "location")):
        unique_ratio = series.nunique(dropna=True) / row_count if row_count else 0.0
        return unique_ratio >= 0.5 or series.nunique(dropna=True) > 50
    unique_ratio = series.nunique(dropna=True) / row_count if row_count else 0.0
    return unique_ratio >= 0.95


def _candidate_features(df: pd.DataFrame, target_column: str) -> list[str]:
    candidates: list[str] = []
    row_count = len(df)
    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        if series.dropna().empty:
            continue
        if _is_date_like(column, series):
            continue
        if _is_identifier_like(column, series, row_count):
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        candidates.append(column)
    return candidates


def _point_biserial_score(series: pd.Series, target: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & target.notna()
    if valid.sum() < 3:
        return 0.0
    values = numeric[valid]
    label = target[valid].astype(float)
    if values.nunique() < 2:
        return 0.0
    score = values.corr(label)
    return 0.0 if pd.isna(score) else abs(float(score))


def _cramers_v_score(series: pd.Series, target: pd.Series) -> float:
    working = pd.DataFrame(
        {
            "feature": series.fillna(MISSING_TOKEN).astype(str),
            "target": target.fillna(-1),
        }
    )
    table = pd.crosstab(working["feature"], working["target"])
    if table.empty or min(table.shape) < 2:
        return 0.0

    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0:
        return 0.0

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0).sum()

    phi2 = chi2 / total
    rows, cols = table.shape
    denom = min(rows - 1, cols - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2 / denom))


def score_features(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> pd.DataFrame:
    cleaned = clean_for_selection(df)
    if target_column not in cleaned.columns:
        raise KeyError(
            f"Target column {target_column!r} was not found in the input dataframe."
        )

    target = _to_binary_target(cleaned[target_column])
    rows: list[dict[str, object]] = []

    for column in _candidate_features(cleaned, target_column):
        series = cleaned[column]
        non_null_count = int(series.notna().sum())
        missing_rate = float(series.isna().mean())
        cardinality = int(series.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(series):
            feature_type = "numeric"
            score = _point_biserial_score(series, target)
        else:
            numeric_attempt = pd.to_numeric(series, errors="coerce")
            numeric_coverage = (
                float(numeric_attempt.notna().mean()) if len(series) else 0.0
            )
            if numeric_coverage >= 0.8:
                feature_type = "numeric"
                score = _point_biserial_score(numeric_attempt, target)
            else:
                feature_type = "categorical"
                score = _cramers_v_score(series, target)

        rows.append(
            {
                "feature": column,
                "type": feature_type,
                "score": score,
                "missing_rate": missing_rate,
                "cardinality": cardinality,
                "non_null_count": non_null_count,
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        ranking = pd.DataFrame(
            columns=[
                "feature",
                "type",
                "score",
                "missing_rate",
                "cardinality",
                "non_null_count",
            ]
        )
        return ranking

    ranking = ranking.sort_values(
        by=["score", "non_null_count", "cardinality"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranking


def select_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.05,
) -> FeatureSelectionResult:
    ranking = score_features(df, target_column=target_column)
    if ranking.empty:
        return FeatureSelectionResult(ranking=ranking, selected_features=[])

    selected = (
        ranking.loc[ranking["score"] >= min_score, "feature"].head(top_k).tolist()
    )
    if not selected:
        selected = ranking.head(top_k)["feature"].tolist()
    return FeatureSelectionResult(ranking=ranking, selected_features=selected)


def selected_feature_frame(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.05,
) -> pd.DataFrame:
    result = select_features(
        df, target_column=target_column, top_k=top_k, min_score=min_score
    )
    columns = [column for column in result.selected_features if column in df.columns]
    if target_column in df.columns:
        columns.append(target_column)
    return df.loc[:, columns].copy()


def load_and_select_features(
    path: Path | None = None,
    target_column: str = TARGET_COLUMN,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.05,
) -> FeatureSelectionResult:
    df = load_data(path) if path is not None else load_data()
    return select_features(
        df, target_column=target_column, top_k=top_k, min_score=min_score
    )


def format_feature_ranking(ranking: pd.DataFrame, limit: int | None = None) -> str:
    if ranking.empty:
        return "No candidate features were found."

    display = ranking.copy()
    if limit is not None:
        display = display.head(limit)
    return display.to_string(
        index=False,
        formatters={"score": "{:.4f}".format, "missing_rate": "{:.2%}".format},
    )


def main() -> None:
    result = load_and_select_features()
    print("Feature ranking:\n")
    print(format_feature_ranking(result.ranking, limit=20))
    print("\nSelected features:\n")
    print(
        ", ".join(result.selected_features)
        if result.selected_features
        else "No features selected."
    )


if __name__ == "__main__":
    main()
