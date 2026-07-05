# ruff: noqa: E402
import os
import sys
from pathlib import Path

# Add src to sys.path
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.append(src_path)

import mlflow
import pandas as pd
from mcp.server.fastmcp import FastMCP
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "FraudDetectionModel"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

# MCP clients launch this server from an arbitrary working directory, so the
# tracking URI must be absolute unless overridden.
mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
)

mcp = FastMCP("fraud-detection")

_model_pipeline = None


def _get_model():
    global _model_pipeline
    if _model_pipeline is None:
        _model_pipeline = mlflow.sklearn.load_model(MODEL_URI)
    return _model_pipeline


@mcp.tool()
def predict_fraud(claims: list[dict]) -> dict:
    """
    Predict whether vehicle insurance claims are fraudulent.

    Each claim is a flat record of claim features, e.g. months_as_customer,
    age, policy_annual_premium, incident_type, incident_severity,
    authorities_contacted, incident_hour_of_the_day,
    number_of_vehicles_involved, bodily_injuries, witnesses, injury_claim,
    property_claim, vehicle_claim, total_claim_amount, insured_hobbies.
    Unknown fields are ignored; fields the model expects but the claim omits
    are imputed by the pipeline.

    Returns one result per claim with fraud_prediction (1 = fraud, 0 = not
    fraud) and fraud_probability (probability of fraud between 0 and 1).
    """
    if not claims:
        raise ValueError("Provide at least one claim record.")

    model = _get_model()
    df = pd.DataFrame(claims)

    # Align with the training columns: the pipeline imputes NaNs but errors
    # on absent columns.
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        df = df.reindex(columns=list(expected))

    predictions = model.predict(df)
    try:
        probabilities = model.predict_proba(df)[:, 1].tolist()
    except AttributeError:
        probabilities = [None] * len(predictions)

    results = [
        {
            "fraud_prediction": int(pred),
            "fraud_probability": float(prob) if prob is not None else None,
        }
        for pred, prob in zip(predictions, probabilities)
    ]
    return {"predictions": results}


@mcp.tool()
def model_info() -> dict:
    """
    Return metadata about the registered fraud detection model: name, latest
    version, run id, status, and evaluation metrics (accuracy, precision,
    recall, f1_score, pr_auc) from the training run.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No registered versions found for '{MODEL_NAME}'.")

    latest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)
    return {
        "name": MODEL_NAME,
        "version": latest.version,
        "run_id": latest.run_id,
        "status": latest.status,
        "creation_timestamp": latest.creation_timestamp,
        "metrics": run.data.metrics,
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
