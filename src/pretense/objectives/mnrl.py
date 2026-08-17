from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

Similarity = Literal["cosine", "dot"]
Encoder = Callable[[Tensor, Tensor], Tensor]
Features = tuple[Tensor, Tensor]
NoSync = Callable[[], AbstractContextManager[None]]


def _gather_with_grad(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    try:
        from torch.distributed import _functional_collectives

        gather = _functional_collectives.all_gather_single_autograd
    except (ImportError, AttributeError):  # pragma: no cover - older supported PyTorch
        from torch.distributed.nn.functional import all_gather

        return torch.cat(
            list(all_gather(value)),
            dim=0,
        )
    group = dist.group.WORLD
    if group is None:  # pragma: no cover - guarded by dist.is_initialized above
        raise RuntimeError("The default distributed process group is unavailable.")
    return gather(value, gather_dim=0, group=group)  # type: ignore[arg-type]


class MultipleNegativesRankingLoss(nn.Module):
    """InfoNCE over matched pairs and in-batch/explicit negatives."""

    def __init__(
        self,
        scale: float = 20.0,
        similarity: Similarity = "cosine",
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive.")
        if similarity not in {"cosine", "dot"}:
            raise ValueError(f"Unknown MNRL similarity: {similarity!r}.")
        self.scale = scale
        self.similarity = similarity
        self.gather_across_devices = gather_across_devices

    def similarity_scores(self, anchors: Tensor, candidates: Tensor) -> Tensor:
        if self.similarity == "cosine":
            anchors = F.normalize(anchors, dim=-1)
            candidates = F.normalize(candidates, dim=-1)
        return anchors @ candidates.T

    def prepare(
        self,
        anchors: Tensor,
        positive: Tensor,
        negatives: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        columns = [positive, *negatives]
        if anchors.ndim != 2 or any(column.ndim != 2 for column in columns):
            raise ValueError("MNRL embeddings must have shape [batch, dimension].")
        if any(column.shape != anchors.shape for column in columns):
            raise ValueError("Every MNRL embedding column must have the same shape.")
        batch_size = anchors.shape[0]
        offset = 0
        if self.gather_across_devices:
            columns = [_gather_with_grad(column) for column in columns]
            if dist.is_available() and dist.is_initialized():
                offset = dist.get_rank() * batch_size
        candidates = torch.cat(columns, dim=0)
        targets = torch.arange(offset, offset + batch_size, device=anchors.device)
        return anchors, candidates, targets

    def forward(self, anchors: Tensor, positive: Tensor, *negatives: Tensor) -> Tensor:
        anchors, candidates, targets = self.prepare(anchors, positive, negatives)
        scores = self.similarity_scores(anchors, candidates) * self.scale
        return F.cross_entropy(scores, targets)


class _RandContext:
    """Replay dropout and other RNG-dependent operations during GradCache re-embedding."""

    def __init__(self, *tensors: Tensor) -> None:
        self.cpu_state = torch.get_rng_state()
        non_mps = tuple(tensor for tensor in tensors if tensor.device.type != "mps")
        self.gpu_devices, self.gpu_states = get_device_states(*non_mps)
        self.mps_state = (
            torch.mps.get_rng_state()
            if any(tensor.device.type == "mps" for tensor in tensors)
            else None
        )

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.cpu_state)
        if self.mps_state is not None:
            self._outside_mps_state = torch.mps.get_rng_state()
            torch.mps.set_rng_state(self.mps_state)
        set_device_states(self.gpu_devices, self.gpu_states)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.mps_state is not None:
            torch.mps.set_rng_state(self._outside_mps_state)
        self._fork.__exit__(exc_type, exc_value, traceback)


class CachedMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    """GradCache MNRL that bounds encoder activation memory by mini-batch size."""

    uses_gradient_cache = True

    def __init__(
        self,
        scale: float = 20.0,
        similarity: Similarity = "cosine",
        mini_batch_size: int = 32,
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__(scale, similarity, gather_across_devices)
        if mini_batch_size < 1:
            raise ValueError("mini_batch_size must be positive.")
        self.mini_batch_size = mini_batch_size

    def _embed_column(
        self,
        encode: Encoder,
        features: Features,
        *,
        with_grad: bool,
        random_states: list[_RandContext] | None = None,
    ) -> tuple[list[Tensor], list[_RandContext]]:
        input_ids, attention_mask = features
        embeddings: list[Tensor] = []
        captured_states: list[_RandContext] = []
        context = nullcontext if with_grad else torch.no_grad
        for index, begin in enumerate(range(0, input_ids.shape[0], self.mini_batch_size)):
            end = begin + self.mini_batch_size
            ids = input_ids[begin:end]
            mask = attention_mask[begin:end]
            random_state = None if random_states is None else random_states[index]
            replay = nullcontext() if random_state is None else random_state
            with replay, context():
                if not with_grad:
                    captured_states.append(_RandContext(ids, mask))
                embeddings.append(encode(ids, mask))
        return embeddings, captured_states

    def _calculate_loss(
        self,
        representations: list[list[Tensor]],
        *,
        with_backward: bool,
    ) -> Tensor:
        anchors = torch.cat(representations[0])
        positive = torch.cat(representations[1])
        negatives = [torch.cat(column) for column in representations[2:]]
        anchors, candidates, targets = self.prepare(anchors, positive, negatives)
        losses: list[Tensor] = []
        batch_size = anchors.shape[0]
        for begin in range(0, batch_size, self.mini_batch_size):
            end = begin + self.mini_batch_size
            scores = self.similarity_scores(anchors[begin:end], candidates) * self.scale
            chunk = F.cross_entropy(scores, targets[begin:end])
            chunk = chunk * len(scores) / batch_size
            if with_backward:
                chunk.backward()
                chunk = chunk.detach()
            losses.append(chunk)
        return torch.stack(losses).sum()

    def forward_cached(
        self,
        encode: Encoder,
        features: Sequence[Features],
        *,
        no_sync: NoSync | None = None,
    ) -> Tensor:
        if len(features) < 2:
            raise ValueError("CMNRL requires at least anchor and positive feature columns.")
        grad_enabled = torch.is_grad_enabled()
        representations: list[list[Tensor]] = []
        random_states: list[list[_RandContext]] = []
        for column in features:
            embeddings, states = self._embed_column(encode, column, with_grad=False)
            representations.append(
                [embedding.detach().requires_grad_(grad_enabled) for embedding in embeddings]
            )
            random_states.append(states)
        if not grad_enabled:
            return self._calculate_loss(representations, with_backward=False)

        loss = self._calculate_loss(representations, with_backward=True)
        cache = [[embedding.grad for embedding in column] for column in representations]
        if any(gradient is None for column in cache for gradient in column):
            raise RuntimeError("CMNRL failed to cache a gradient for every embedding column.")
        deferred = loss.detach().requires_grad_()
        deferred.register_hook(
            partial(
                self._reembed_backward,
                encode=encode,
                features=features,
                cache=cache,
                random_states=random_states,
                no_sync=no_sync,
            )
        )
        return deferred

    def forward(  # type: ignore[override]
        self,
        encode: Encoder,
        features: Sequence[Features],
        *,
        no_sync: NoSync | None = None,
    ) -> Tensor:
        return self.forward_cached(encode, features, no_sync=no_sync)

    def _reembed_backward(
        self,
        grad_output: Tensor,
        *,
        encode: Encoder,
        features: Sequence[Features],
        cache: list[list[Tensor | None]],
        random_states: list[list[_RandContext]],
        no_sync: NoSync | None,
    ) -> None:
        replays_remaining = sum(len(column) for column in cache)
        with torch.enable_grad():
            for column, gradients, states in zip(features, cache, random_states, strict=True):
                input_ids, attention_mask = column
                for index, begin in enumerate(
                    range(0, input_ids.shape[0], self.mini_batch_size)
                ):
                    end = begin + self.mini_batch_size
                    ids = input_ids[begin:end]
                    mask = attention_mask[begin:end]
                    gradient = gradients[index]
                    assert gradient is not None
                    replays_remaining -= 1
                    sync = no_sync() if no_sync is not None and replays_remaining else nullcontext()
                    with sync, states[index]:
                        embedding = encode(ids, mask)
                        surrogate = (
                            torch.dot(
                                embedding.flatten().float(), gradient.flatten().float()
                            )
                            * grad_output
                        )
                        surrogate.backward()
