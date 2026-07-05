# ruff: noqa: E402
import sys
from pathlib import Path

# Add src to sys.path
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.append(src_path)

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List

app = FastAPI(
    title="Fraud Detection API",
    description="API for serving the fraud detection ML pipeline.",
    version="1.0.0",
)


# Define a flexible request schema
class ClaimData(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictRequest(BaseModel):
    claims: List[ClaimData]


# Global variable to hold the model pipeline
model_pipeline = None


@app.on_event("startup")
def load_model():
    global model_pipeline
    model_uri = "models:/FraudDetectionModel/latest"
    print(f"Loading model from {model_uri}...")
    try:
        model_pipeline = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        # Not raising an exception here to allow the server to start,
        # but endpoints will fail if model isn't loaded.


@app.get("/health")
def health_check():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "healthy", "model": "FraudDetectionModel"}


@app.post("/predict")
def predict(request: PredictRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        # Convert pydantic models to dictionaries
        records = [claim.model_dump() for claim in request.claims]

        # Convert to Pandas DataFrame
        df = pd.DataFrame(records)

        # Align with the training columns: the pipeline imputes NaNs but
        # errors on absent columns.
        expected = getattr(model_pipeline, "feature_names_in_", None)
        if expected is not None:
            df = df.reindex(columns=list(expected))

        # Run prediction
        predictions = model_pipeline.predict(df)

        # Try to get probabilities if the model supports it
        try:
            probabilities = model_pipeline.predict_proba(df)[:, 1].tolist()
        except AttributeError:
            probabilities = [None] * len(predictions)

        # Format the response
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(
                {
                    "fraud_prediction": int(pred),
                    "fraud_probability": float(prob) if prob is not None else None,
                }
            )

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
