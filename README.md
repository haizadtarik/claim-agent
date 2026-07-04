# Vehicle Claim Fraud Detection

## Overview
This project implements an end-to-end machine learning pipeline for detecting fraudulent vehicle insurance claims. It handles everything from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, and serving.

The pipeline trains multiple machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost), handles class imbalance through class weighting, tracks experiments locally using MLflow, and automatically registers the best-performing model based on the Recall score. The best model is then served via a FastAPI application.

## Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.12+ installed. We recommend using a virtual environment (such as `conda` or `venv`).

### 2. Installation
Install the required dependencies:

```bash
# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install pre-commit hooks for code formatting and linting
pre-commit install
```

> **Tip:** Every step below is also available as a Make target — run `make help` to list them (e.g. `make train`, `make serve`, `make test`, or `make pipeline` for the full EDA → feature selection → training run).

### 3. Train the Models
Run the training pipeline. This script will load the data, train multiple models, track their performance using MLflow, and automatically register the best model to the local MLflow Model Registry.

```bash
python src/fraud_detection/pipeline/train.py
```

*To view the MLflow UI and compare model runs, execute `mlflow ui` in your terminal and visit `http://127.0.0.1:5000`.*

### 4. Serve as REST API
Start the FastAPI server. On startup, the application dynamically loads the best registered model from MLflow.

```bash
uvicorn src.fraud_detection.api.app:app --reload --port 8080
```
*You can view the interactive API documentation at `http://127.0.0.1:8080/docs`.*

### 5. Test the Endpoints
To verify the API is working correctly, run the provided test script in a new terminal window:

```bash
python src/fraud_detection/api/test_api.py
```

Alternatively, you can run the full Pytest testing suite:

```bash
pytest tests/
```

### 6. Serve as MCP Server
The fraud detector is also available as a [Model Context Protocol](https://modelcontextprotocol.io) server, so AI assistants such as Claude can call it directly as a tool:

```bash
make mcp-serve  # or: python src/fraud_detection/mcp/server.py
```

The server communicates over stdio and exposes two tools:

- `predict_fraud(claims)` — score one or more claim records; returns `fraud_prediction` (1 = fraud) and `fraud_probability` per claim. Missing fields are imputed and unknown fields are ignored.
- `model_info()` — name, latest registered version, and evaluation metrics of the model being served.

To register it with an MCP client (e.g. Claude Code or Claude Desktop), use a config like:

```json
{
  "mcpServers": {
    "fraud-detection": {
      "command": "python",
      "args": ["/absolute/path/to/claim-agent/src/fraud_detection/mcp/server.py"]
    }
  }
}
```

The server loads `models:/FraudDetectionModel/latest` from the local MLflow registry (`sqlite:///mlflow.db` in the repo root by default; override with the `MLFLOW_TRACKING_URI` environment variable), so train and register a model first.
