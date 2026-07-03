import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    # We need to mock mlflow to prevent it from failing on startup
    # if the model isn't registered locally during CI/CD.
    with patch("mlflow.sklearn.load_model") as mock_load_model:
        import numpy as np

        # Create a dummy model that returns fake predictions
        dummy_model = MagicMock()
        dummy_model.predict.return_value = np.array([1])
        dummy_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_load_model.return_value = dummy_model

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
