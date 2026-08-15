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
