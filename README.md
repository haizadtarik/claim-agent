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

#### Happy & sad path examples

Two ready-made payloads in [`data/examples/`](data/examples/) exercise both decision paths. Each is a real record from `data/insurance_claims.csv` that the registered model classifies confidently, so they double as a sanity check that the served model behaves as expected. With the API running (step 4):

```bash
# Happy path — legitimate claim, expect fraud_prediction 0 (claim approved)
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d @data/examples/happy_path_claim.json

# Sad path — fraudulent claim, expect fraud_prediction 1 (claim disapproved)
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d @data/examples/sad_path_claim.json
```

Expected responses (probabilities vary with the trained model version):

```json
{"predictions": [{"fraud_prediction": 0, "fraud_probability": 0.019}]}
{"predictions": [{"fraud_prediction": 1, "fraud_probability": 0.999}]}
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

### 7. Chat with the Claim Agent
A conversational agent built on [LangChain](https://python.langchain.com) and a local [Ollama](https://ollama.com) `gemma4` model handles claim approvals end to end: it interviews the claimant about their vehicle insurance claim, scores the claim with the registered fraud detection model, then **approves** the claim when it is predicted legitimate or **disapproves** it when it is flagged as fraud. The approval decision is made deterministically from the model prediction — the LLM only gathers details and relays the outcome.

Prerequisites: a trained model in the registry (step 3) and Ollama running with the model pulled (`ollama pull gemma4`). Then start the chat:

```bash
make agent  # or: PYTHONPATH=src python -m claim_agent.agent
```

Example session:

```
You: Hi, I'd like to file a claim for a minor collision, about $5,000 total.
Agent: ... (asks follow-up questions, then runs the fraud check)
Agent: Good news — your claim has been approved. ...
```

Type `quit` to end the session.

Prefer a browser? The same agent is available as a [Streamlit](https://streamlit.io) chat app:

```bash
make ui  # or: PYTHONPATH=src streamlit run src/claim_agent/ui.py
```

This opens a chat page at `http://localhost:8501` with the conversation history, a spinner while the agent thinks, and a **New conversation** button in the sidebar to start over. It has the same prerequisites as the terminal chat (registered model + Ollama serving `gemma4`).

The official decision is shown in a banner above the chat that is rendered directly from the fraud model's tool result — the LLM's prose can never produce or override it. Until an assessment has actually run, the page explicitly says no decision is official yet.
