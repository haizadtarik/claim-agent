from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """
    Returns a dictionary of models to be evaluated.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        ),
        "LightGBM": LGBMClassifier(
            random_state=42, verbose=-1, class_weight="balanced"
        ),
        "CatBoost": CatBoostClassifier(
            verbose=0, random_state=42, auto_class_weights="Balanced"
        ),
    }
    return models
