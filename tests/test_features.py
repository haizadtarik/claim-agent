import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.features import score_features


@pytest.fixture
def synthetic_claims() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    target = np.tile([0, 1], n // 2)
    return pd.DataFrame(
        {
            "months_as_customer": target * 10 + rng.integers(0, 5, n),
            "total_claim_amount": rng.integers(10, 30, n) * 100,
            "incident_severity": np.where(target == 1, "Major Damage", "Minor Damage"),
            "auto_make": rng.choice(["Saab", "Dodge", "Suburu"], n),
            "fraud_reported": np.where(target == 1, "Y", "N"),
        }
    )


def test_numeric_features_scored_with_pearson(synthetic_claims):
    ranking = score_features(synthetic_claims).set_index("feature")

    assert ranking.loc["months_as_customer", "method"] == "pearson"
    expected = abs(
        synthetic_claims["months_as_customer"].corr(
            synthetic_claims["fraud_reported"].map({"N": 0, "Y": 1}).astype(float)
        )
    )
    assert ranking.loc["months_as_customer", "score"] == pytest.approx(expected)


def test_categorical_features_scored_with_permutation_importance(synthetic_claims):
    ranking = score_features(synthetic_claims).set_index("feature")
    categorical = ranking[ranking["type"] == "categorical"]

    assert set(categorical.index) == {"incident_severity", "auto_make"}
    assert (categorical["method"] == "permutation_importance").all()
    # Negative permutation importances are clipped to zero.
    assert (categorical["score"] >= 0).all()
    # Shuffling a perfect predictor drops held-out AUC from ~1.0 toward ~0.5.
    assert categorical.loc["incident_severity", "score"] > 0.1
    # A noise category earns ~zero importance on held-out data.
    assert categorical.loc["auto_make", "score"] < 0.05
    assert (
        categorical.loc["incident_severity", "score"]
        > categorical.loc["auto_make", "score"]
    )


def test_scoring_is_deterministic(synthetic_claims):
    first = score_features(synthetic_claims)
    second = score_features(synthetic_claims)

    pd.testing.assert_frame_equal(first, second)
