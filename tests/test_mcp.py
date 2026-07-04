import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def server():
    from fraud_detection.mcp import server as mcp_server

    # Reset the lazy-loaded model between tests
    mcp_server._model_pipeline = None
    yield mcp_server
    mcp_server._model_pipeline = None


@pytest.fixture
def dummy_model(server):
    with patch("mlflow.sklearn.load_model") as mock_load_model:
        model = MagicMock()
        model.feature_names_in_ = np.array(
            ["age", "incident_type", "total_claim_amount", "umbrella_limit"]
        )
        model.predict.return_value = np.array([1, 0])
        model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        mock_load_model.return_value = model
        yield model


SAMPLE_CLAIMS = [
    {"age": 35, "incident_type": "Collision", "total_claim_amount": 5000},
    {"age": 42, "incident_type": "Vehicle Theft", "total_claim_amount": 1000},
]


def test_tools_are_registered(server):
    tools = asyncio.run(server.mcp.list_tools())
    tool_names = {tool.name for tool in tools}
    assert "predict_fraud" in tool_names
    assert "model_info" in tool_names


def test_predict_fraud_returns_predictions(server, dummy_model):
    result = server.predict_fraud(SAMPLE_CLAIMS)

    assert len(result["predictions"]) == 2
    first = result["predictions"][0]
    assert first["fraud_prediction"] == 1
    assert first["fraud_probability"] == 0.9
    second = result["predictions"][1]
    assert second["fraud_prediction"] == 0
    assert second["fraud_probability"] == 0.2


def test_predict_fraud_aligns_columns_with_model_features(server, dummy_model):
    server.predict_fraud([{"age": 35, "bogus_field": "x"}, {"age": 42}])

    df = dummy_model.predict.call_args[0][0]
    assert list(df.columns) == list(dummy_model.feature_names_in_)
    assert df["umbrella_limit"].isna().all()
    assert "bogus_field" not in df.columns


def test_predict_fraud_rejects_empty_claims(server, dummy_model):
    with pytest.raises(ValueError, match="at least one claim"):
        server.predict_fraud([])


def test_predict_fraud_loads_model_once(server, dummy_model):
    with patch("mlflow.sklearn.load_model") as mock_load_model:
        mock_load_model.return_value = dummy_model
        server.predict_fraud(SAMPLE_CLAIMS)
        server.predict_fraud(SAMPLE_CLAIMS)
        assert mock_load_model.call_count == 1


def test_model_info_returns_latest_version_metadata(server):
    def make_version(number, run_id):
        version = MagicMock()
        version.version = str(number)
        version.run_id = run_id
        version.status = "READY"
        version.creation_timestamp = 1700000000000 + number
        return version

    run = MagicMock()
    run.data.metrics = {"recall": 0.9, "precision": 0.8}

    with patch("fraud_detection.mcp.server.MlflowClient") as mock_client_cls:
        client = mock_client_cls.return_value
        client.search_model_versions.return_value = [
            make_version(4, "old456"),
            make_version(5, "abc123"),
        ]
        client.get_run.return_value = run

        info = server.model_info()

    client.search_model_versions.assert_called_once_with("name='FraudDetectionModel'")
    assert info["name"] == "FraudDetectionModel"
    assert info["version"] == "5"
    assert info["run_id"] == "abc123"
    assert info["metrics"] == {"recall": 0.9, "precision": 0.8}
