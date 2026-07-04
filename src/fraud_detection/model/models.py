import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODEL_NAMES = (
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
)


def build_model(
    model_name: str, params: dict | None = None, scale_pos_weight: float = 1.0
):
    """
    Build a classifier by name, applying tuned hyperparameters on top of the
    base configuration (seed, class balancing, verbosity).
    """
    params = params or {}
    if model_name == "LogisticRegression":
        kwargs = {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"}
        return LogisticRegression(**{**kwargs, **params})
    if model_name == "RandomForest":
        kwargs = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}
        return RandomForestClassifier(**{**kwargs, **params})
    if model_name == "XGBoost":
        kwargs = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "scale_pos_weight": scale_pos_weight,
        }
        return XGBClassifier(**{**kwargs, **params})
    if model_name == "LightGBM":
        kwargs = {"random_state": 42, "verbose": -1, "class_weight": "balanced"}
        return LGBMClassifier(**{**kwargs, **params})
    if model_name == "CatBoost":
        kwargs = {"verbose": 0, "random_state": 42, "auto_class_weights": "Balanced"}
        return CatBoostClassifier(**{**kwargs, **params})
    raise ValueError(f"Unknown model name: {model_name!r}")


def suggest_params(trial: optuna.trial.Trial, model_name: str) -> dict:
    """
    Sample a hyperparameter configuration for the given model from its
    Optuna search space. Keys match the estimator's constructor arguments.
    """
    if model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
    raise ValueError(f"Unknown model name: {model_name!r}")


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """
    Returns a dictionary of models to be evaluated.
    """
    return {
        model_name: build_model(model_name, scale_pos_weight=scale_pos_weight)
        for model_name in MODEL_NAMES
    }
