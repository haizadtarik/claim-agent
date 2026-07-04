# ruff: noqa: E402
import os
import sys
from pathlib import Path

# Add src to sys.path so we can import from fraud_detection
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.append(src_path)

from fraud_detection.data.load import load_data
from fraud_detection.model.models import MODEL_NAMES, build_model, suggest_params
from fraud_detection.pipeline.preprocess import build_preprocessor
from fraud_detection.data.features import (
    TARGET_COLUMN,
    _to_binary_target,
    selected_feature_frame,
)

import mlflow
import mlflow.sklearn
import optuna
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

N_TRIALS = int(os.environ.get("N_TRIALS", "25"))


def tune_model(model_name, preprocessor, X_train, y_train, scale_pos_weight):
    """
    Run an Optuna study for one model, maximizing mean cross-validated PR-AUC
    on the training split. Returns the best params and best CV score.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = suggest_params(trial, model_name)
        model = build_model(model_name, params, scale_pos_weight=scale_pos_weight)
        pipeline = Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("classifier", model)]
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="average_precision"
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params, study.best_value


def main():
    print("Loading data...")
    df = load_data()

    print("Selecting features...")
    df = selected_feature_frame(df)
    selected = [column for column in df.columns if column != TARGET_COLUMN]
    print(f"Selected {len(selected)} features: {', '.join(selected)}")

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

    mlflow.set_experiment("fraud_detection_vehicle_claims")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Training models ({N_TRIALS} Optuna trials each)...")
    best_run_id = None
    best_recall = -1.0

    for model_name in MODEL_NAMES:
        with mlflow.start_run(run_name=model_name) as run:
            print(f"Tuning {model_name}...")
            best_params, cv_pr_auc = tune_model(
                model_name, preprocessor, X_train, y_train, scale_pos_weight
            )
            print(f"{model_name} best CV PR-AUC: {cv_pr_auc:.4f} with {best_params}")

            # Refit the best configuration on the full training split
            model = build_model(
                model_name, best_params, scale_pos_weight=scale_pos_weight
            )
            pipeline = Pipeline(
                steps=[("preprocessor", clone(preprocessor)), ("classifier", model)]
            )
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
            pr_auc = average_precision_score(y_test, y_prob)

            # Log params and metrics
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_pr_auc", cv_pr_auc)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("pr_auc", pr_auc)

            # Log model
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            print(
                f"{model_name} Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}"
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
