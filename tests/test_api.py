import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_model():
    import numpy as np

    # Create a dummy model that returns fake predictions
    model = MagicMock()
    model.feature_names_in_ = np.array(
        ["age", "incident_type", "total_claim_amount", "umbrella_limit"]
    )
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    return model


@pytest.fixture
def client(dummy_model):
    # We need to mock mlflow to prevent it from failing on startup
    # if the model isn't registered locally during CI/CD.
    with patch("mlflow.sklearn.load_model", return_value=dummy_model):
        from fraud_detection.api.app import app

        # Using the `with` block triggers the FastAPI startup event
        with TestClient(app) as c:
            yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model": "FraudDetectionModel"}


def test_predict(client):
    payload = {
        "claims": [
            {"age": 35, "incident_type": "Collision", "total_claim_amount": 5000}
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1

    pred = data["predictions"][0]
    assert "fraud_prediction" in pred
    assert "fraud_probability" in pred

    assert pred["fraud_prediction"] == 1
    assert pred["fraud_probability"] == 0.9


def test_predict_imputes_missing_model_columns(client, dummy_model):
    payload = {"claims": [{"age": 35, "bogus_field": "x"}]}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    df = dummy_model.predict.call_args[0][0]
    assert list(df.columns) == list(dummy_model.feature_names_in_)
    assert df["umbrella_limit"].isna().all()
    assert "bogus_field" not in df.columns
