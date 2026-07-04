import optuna
import pytest

from fraud_detection.model.models import (
    MODEL_NAMES,
    build_model,
    get_models,
    suggest_params,
)


def _new_trial() -> optuna.trial.Trial:
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    return study.ask()


def test_model_names_matches_get_models():
    assert list(get_models()) == list(MODEL_NAMES)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_suggest_params_returns_nonempty_search_space(model_name):
    params = suggest_params(_new_trial(), model_name)

    assert isinstance(params, dict)
    assert params


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_build_model_applies_suggested_params(model_name):
    params = suggest_params(_new_trial(), model_name)
    model = build_model(model_name, params=params, scale_pos_weight=2.0)

    model_params = model.get_params()
    for key, value in params.items():
        assert model_params[key] == value


def test_build_model_without_params_keeps_defaults():
    model = build_model("LogisticRegression")

    assert model.get_params()["max_iter"] == 1000
    assert model.get_params()["class_weight"] == "balanced"


def test_build_model_passes_scale_pos_weight_to_xgboost():
    model = build_model("XGBoost", scale_pos_weight=3.5)

    assert model.get_params()["scale_pos_weight"] == 3.5


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("NotAModel")
    with pytest.raises(ValueError, match="Unknown model"):
        suggest_params(_new_trial(), "NotAModel")
