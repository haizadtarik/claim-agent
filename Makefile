.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install eda features train serve mcp-serve agent ui smoke-test mlflow-ui test lint format

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and pre-commit hooks
	pip install -r requirements.txt
	pre-commit install

eda: ## Generate the EDA report at reports/eda_report.html
	PYTHONPATH=src $(PYTHON) -m fraud_detection.data.eda

features: ## Print the feature-selection ranking
	PYTHONPATH=src $(PYTHON) -m fraud_detection.data.features

train: ## Tune models with Optuna (override trials: N_TRIALS=5 make train), track with MLflow, register the best
	$(PYTHON) src/fraud_detection/pipeline/train.py

serve: ## Serve the best registered model with FastAPI on port 8080
	uvicorn src.fraud_detection.api.app:app --reload --port 8080

mcp-serve: ## Serve the best registered model as an MCP server over stdio
	$(PYTHON) src/fraud_detection/mcp/server.py

agent: ## Chat with the claim approval agent (needs Ollama serving gemma4)
	PYTHONPATH=src $(PYTHON) -m claim_agent.agent

ui: ## Chat with the claim approval agent in the browser via Streamlit
	PYTHONPATH=src streamlit run src/claim_agent/ui.py

smoke-test: ## Hit the running API with sample requests (needs `make serve`)
	API_URL=http://127.0.0.1:8080/predict $(PYTHON) src/fraud_detection/api/test_api.py

mlflow-ui: ## Open the MLflow tracking UI on port 5000
	mlflow ui

test: ## Run the pytest suite
	pytest tests/

lint: ## Check code style with ruff
	ruff check src tests

format: ## Auto-format and fix lint issues with ruff
	ruff format src tests
	ruff check --fix src tests
