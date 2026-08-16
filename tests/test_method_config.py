import pytest

from pretense import MethodConfig


def test_method_config_uses_documented_defaults() -> None:
    config = MethodConfig(name="retromae")
    assert config.encoder_mlm_probability == 0.30
    assert config.decoder_mlm_probability == 0.50


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown pretraining method"):
        MethodConfig(name="almost-retromae")


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="Masking probabilities"):
        MethodConfig(name="retromae", mlm_probability=2)


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
def test_invalid_contriever_settings_are_rejected(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="contriever", **{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("contrastive_distance_metric", "dot"), ("contrastive_margin", 0)],
)
def test_invalid_contrastive_settings_are_rejected(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="contrastive", **{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("mnrl_scale", 0), ("mnrl_similarity", "euclidean"), ("cmnrl_mini_batch_size", 0)],
)
def test_invalid_mnrl_settings_are_rejected(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="mnrl", **{field: invalid})


def test_simcse_defaults_projection_only_to_supervised_inference() -> None:
    unsupervised = MethodConfig(name="simcse")
    supervised = MethodConfig(name="simcse", simcse_mode="supervised")
    assert unsupervised.simcse_temperature == 0.05
    assert not unsupervised.simcse_uses_projection_at_inference
    assert supervised.simcse_uses_projection_at_inference
    assert MethodConfig(
        name="simcse", simcse_mode="supervised", simcse_mlp_only_train=True
    ).simcse_uses_projection_at_inference is False


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("simcse_mode", "semi-supervised"),
        ("simcse_temperature", 0),
        ("simcse_hard_negative_weight", float("nan")),
        ("simcse_mlm_weight", -0.1),
    ],
)
def test_invalid_simcse_settings_are_rejected(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="simcse", **{field: invalid})
