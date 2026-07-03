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

### 3. Train the Models
Run the training pipeline. This script will load the data, train multiple models, track their performance using MLflow, and automatically register the best model to the local MLflow Model Registry.

```bash
python src/fraud_detection/pipeline/train.py
```

*To view the MLflow UI and compare model runs, execute `mlflow ui` in your terminal and visit `http://127.0.0.1:5000`.*

### 4. Serve the API
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
