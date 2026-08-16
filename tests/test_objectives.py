from copy import deepcopy

import pytest
import torch

from pretense import (
    CachedMultipleNegativesRankingLoss,
    ContrastiveLoss,
    MultipleNegativesRankingLoss,
)


def test_contrastive_loss_rewards_positive_and_separated_negative_pairs() -> None:
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    ideal_other = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    wrong_other = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([1, 0])
    objective = ContrastiveLoss()
    assert objective(anchor, ideal_other, labels) < objective(anchor, wrong_other, labels)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_contrastive_loss_reductions(reduction: str) -> None:
    objective = ContrastiveLoss(reduction=reduction)  # type: ignore[arg-type]
    loss = objective(torch.eye(2), torch.eye(2), torch.tensor([1, 0]))
    assert loss.shape == (2,) if reduction == "none" else loss.ndim == 0


def test_contrastive_loss_validates_inputs() -> None:
    objective = ContrastiveLoss()
    with pytest.raises(ValueError, match="matching shapes"):
        objective(torch.ones(2, 3), torch.ones(2, 4), torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="must be 0.*or 1"):
        objective(torch.ones(2, 3), torch.ones(2, 3), torch.tensor([1, 0.5]))


def test_legacy_objective_model_exports_remain_available() -> None:
    from pretense.objectives import RetroMAEForPretraining

    assert RetroMAEForPretraining.__name__ == "RetroMAEForPretraining"


@pytest.mark.parametrize("similarity", ["cosine", "dot"])
def test_mnrl_matches_sentence_transformers(similarity: str) -> None:
    try:
        from sentence_transformers.sentence_transformer.losses import (
            MultipleNegativesRankingLoss as STMultipleNegativesRankingLoss,
        )
        from sentence_transformers.util import cos_sim, dot_score
    except ImportError:  # Sentence Transformers 5.2-5.6
        from sentence_transformers.losses import (
            MultipleNegativesRankingLoss as STMultipleNegativesRankingLoss,
        )
        from sentence_transformers.util import cos_sim, dot_score

    torch.manual_seed(13)
    anchor = torch.randn(4, 8)
    positive = torch.randn(4, 8)
    negative = torch.randn(4, 8)
    ours = MultipleNegativesRankingLoss(scale=7.0, similarity=similarity)  # type: ignore[arg-type]
    actual = ours(anchor, positive, negative)
    reference = STMultipleNegativesRankingLoss(
        model=None,
        scale=7.0,
        similarity_fct=cos_sim if similarity == "cosine" else dot_score,
    )
    expected = reference.compute_loss_from_embeddings([anchor, positive, negative], labels=None)
    assert torch.allclose(actual, expected)


def test_cached_mnrl_matches_loss_and_parameter_gradients() -> None:
    torch.manual_seed(17)
    encoder = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh())
    cached_encoder = deepcopy(encoder)
    # Five rows with cache chunks of two exercises the final uneven mini-batch.
    columns = [torch.randn(5, 5) for _ in range(3)]
    masks = [torch.ones(5, 1) for _ in columns]

    regular = MultipleNegativesRankingLoss(scale=9.0)
    regular_loss = regular(*(encoder(column) for column in columns))
    (regular_loss * 0.25).backward()

    cached = CachedMultipleNegativesRankingLoss(scale=9.0, mini_batch_size=2)
    cached_loss = cached(
        lambda values, mask: cached_encoder(values),
        list(zip(columns, masks, strict=True)),
    )
    (cached_loss * 0.25).backward()

    assert torch.allclose(cached_loss, regular_loss.detach(), atol=1e-6)
    for expected, actual in zip(encoder.parameters(), cached_encoder.parameters(), strict=True):
        assert torch.allclose(actual.grad, expected.grad, atol=1e-6)


def test_cached_mnrl_replays_dropout_randomness() -> None:
    torch.manual_seed(23)
    dropout = torch.nn.Dropout(p=0.5)
    first_pass: list[torch.Tensor] = []
    replay: list[torch.Tensor] = []

    def encode(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        output = dropout(values)
        (replay if torch.is_grad_enabled() else first_pass).append(output.detach().clone())
        return output

    columns = [torch.randn(4, 6, requires_grad=True) for _ in range(2)]
    masks = [torch.ones(4, 1) for _ in columns]
    loss = CachedMultipleNegativesRankingLoss(mini_batch_size=2)(
        encode,
        list(zip(columns, masks, strict=True)),
    )
    loss.backward()

    assert len(first_pass) == len(replay) == 4
    for expected, actual in zip(first_pass, replay, strict=True):
        assert torch.equal(actual, expected)


def test_mnrl_validates_configuration_and_columns() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        MultipleNegativesRankingLoss(scale=0)
    with pytest.raises(ValueError, match="same shape"):
        MultipleNegativesRankingLoss()(torch.ones(2, 3), torch.ones(3, 3))
    with pytest.raises(ValueError, match="mini_batch_size"):
        CachedMultipleNegativesRankingLoss(mini_batch_size=0)
