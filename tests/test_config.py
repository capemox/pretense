from pathlib import Path

import pytest

from pretense import PretenseConfig


def minimum_config() -> dict:
    return {
        "model": {"model_name_or_path": "tiny"},
        "method": {"name": "retromae"},
        "data": {"data_files": "train.jsonl"},
    }


def test_minimum_config_uses_documented_defaults() -> None:
    config = PretenseConfig.from_dict(minimum_config())
    assert config.method.encoder_mlm_probability == 0.30
    assert config.method.decoder_mlm_probability == 0.50
    assert config.model.model_kwargs == {}
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.save_total_limit is None


def test_training_checkpoint_limit_round_trip() -> None:
    value = minimum_config()
    value["training"] = {"save_total_limit": 2}
    config = PretenseConfig.from_dict(value)
    assert config.training.save_total_limit == 2


def test_weights_only_checkpoint_cannot_be_resumed() -> None:
    value = minimum_config()
    value["training"] = {"save_only_model": True, "resume_from_checkpoint": True}
    with pytest.raises(ValueError, match="cannot be resumed"):
        PretenseConfig.from_dict(value)


def test_hub_push_requires_a_matching_export() -> None:
    value = minimum_config()
    value["export"] = {
        "push_to_hub": True,
        "transformers": False,
        "transformers_repo_id": "owner/model",
    }
    with pytest.raises(ValueError, match="Enable the Transformers export"):
        PretenseConfig.from_dict(value)


def test_model_kwargs_round_trip() -> None:
    value = minimum_config()
    value["model"]["model_kwargs"] = {
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
    }
    config = PretenseConfig.from_dict(value)
    assert config.to_dict()["model"]["model_kwargs"] == value["model"]["model_kwargs"]


def test_model_kwargs_rejects_duplicate_trust_remote_code() -> None:
    value = minimum_config()
    value["model"]["model_kwargs"] = {"trust_remote_code": True}
    with pytest.raises(ValueError, match="Set model.trust_remote_code directly"):
        PretenseConfig.from_dict(value)


def test_unknown_keys_fail_early() -> None:
    value = minimum_config()
    value["method"]["typo"] = True
    with pytest.raises(ValueError, match="Unknown MethodConfig keys"):
        PretenseConfig.from_dict(value)


def test_invalid_probability_is_rejected() -> None:
    value = minimum_config()
    value["method"]["mlm_probability"] = 2
    with pytest.raises(ValueError, match="Masking probabilities"):
        PretenseConfig.from_dict(value)


def test_unknown_method_is_rejected() -> None:
    value = minimum_config()
    value["method"]["name"] = "almost-retromae"
    with pytest.raises(ValueError, match="Unknown pretraining method"):
        PretenseConfig.from_dict(value)


def test_contriever_config_round_trip() -> None:
    value = minimum_config()
    value["method"] = {
        "name": "contriever",
        "momentum": 0.9995,
        "queue_size": 131_072,
        "contrastive_temperature": 0.05,
        "augmentation": "delete",
        "augmentation_probability": 0.1,
        "crop_ratio_min": 0.1,
        "crop_ratio_max": 0.5,
    }
    config = PretenseConfig.from_dict(value)
    assert config.method.name == "contriever"
    assert config.method.queue_size == 131_072
    assert config.to_dict()["method"]["momentum"] == 0.9995


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("momentum", 1.0),
        ("queue_size", 0),
        ("augmentation_probability", 1.0),
        ("crop_ratio_min", 0.0),
        ("crop_ratio_max", 1.1),
    ],
)
def test_invalid_contriever_config_is_rejected(field: str, invalid: object) -> None:
    value = minimum_config()
    value["method"] = {"name": "contriever", field: invalid}
    with pytest.raises(ValueError):
        PretenseConfig.from_dict(value)


def test_contriever_recipe_parses() -> None:
    recipe = Path(__file__).parents[1] / "recipes" / "contriever.yaml"
    config = PretenseConfig.from_yaml(recipe)
    assert config.method.name == "contriever"
    assert config.method.queue_size == 131_072
    assert config.data.text_column == "text"


def test_contrastive_recipe_parses() -> None:
    recipe = Path(__file__).parents[1] / "recipes" / "contrastive.yaml"
    config = PretenseConfig.from_yaml(recipe)
    assert config.method.name == "contrastive"
    assert config.method.contrastive_distance_metric == "cosine"
    assert config.method.contrastive_margin == 0.5
    assert config.data.text_pair_column == "sentence2"
    assert config.data.label_column == "label"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("contrastive_distance_metric", "dot"), ("contrastive_margin", 0)],
)
def test_invalid_contrastive_config_is_rejected(field: str, invalid: object) -> None:
    value = minimum_config()
    value["method"] = {"name": "contrastive", field: invalid}
    with pytest.raises(ValueError):
        PretenseConfig.from_dict(value)


@pytest.mark.parametrize("method", ["mnrl", "cmnrl"])
def test_mnrl_recipes_parse(method: str) -> None:
    recipe = Path(__file__).parents[1] / "recipes" / f"{method}.yaml"
    config = PretenseConfig.from_yaml(recipe)
    assert config.method.name == method
    assert config.method.mnrl_scale == 20.0
    assert config.data.text_pair_column == "positive"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("mnrl_scale", 0), ("mnrl_similarity", "euclidean"), ("cmnrl_mini_batch_size", 0)],
)
def test_invalid_mnrl_config_is_rejected(field: str, invalid: object) -> None:
    value = minimum_config()
    value["method"] = {"name": "mnrl", field: invalid}
    with pytest.raises(ValueError):
        PretenseConfig.from_dict(value)


def test_duplicate_mnrl_negative_columns_are_rejected() -> None:
    value = minimum_config()
    value["method"] = {"name": "mnrl"}
    value["data"]["negative_columns"] = ["negative", "negative"]
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        PretenseConfig.from_dict(value)
