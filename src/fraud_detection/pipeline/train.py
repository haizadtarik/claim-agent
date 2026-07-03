# ruff: noqa: E402
import sys
from pathlib import Path

# Add src to sys.path so we can import from fraud_detection
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.append(src_path)

from fraud_detection.data.load import load_data
from fraud_detection.model.models import get_models
from fraud_detection.pipeline.preprocess import build_preprocessor
from fraud_detection.data.features import TARGET_COLUMN, _to_binary_target

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main():
    print("Loading data...")
    df = load_data()

    print("Building preprocessor...")
    preprocessor, feature_names = build_preprocessor(df, TARGET_COLUMN)

    # Split data
    X = df.drop(columns=[TARGET_COLUMN])
    y = _to_binary_target(df[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    scale_pos_weight = (
        float(num_negatives / num_positives) if num_positives > 0 else 1.0
    )

    models = get_models(scale_pos_weight=scale_pos_weight)

    mlflow.set_experiment("fraud_detection_vehicle_claims")

    print("Training models...")
    best_run_id = None
    best_recall = -1.0

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            print(f"Training {model_name}...")

            # Create a full pipeline with preprocessor and model
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", model)]
            )

            # Train the model
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)
            y_prob = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline, "predict_proba")
                else y_pred
            )

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log model
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            print(
                f"{model_name} Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
            )

            if recall > best_recall:
                best_recall = recall
                best_run_id = run.info.run_id

    print(f"\nTraining complete. Best Recall: {best_recall:.4f}")
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "FraudDetectionModel"
        print(f"Registering best model ({model_uri}) as '{model_name}'...")
        mlflow.register_model(model_uri, model_name)
        print("Model registration successful.")


if __name__ == "__main__":
    main()
