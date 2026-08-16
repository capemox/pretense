"""Two-process CPU smoke test, invoked explicitly by CI rather than pytest."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import BertConfig, BertForMaskedLM

from pretense import (
    CachedMNRLForPretraining,
    CoCondenserForPretraining,
    ContrieverForPretraining,
    MethodConfig,
    MNRLForPretraining,
    SimCSEForPretraining,
)


def tiny_encoder() -> BertForMaskedLM:
    return BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = int(os.environ["RANK"])

    def stage(message: str) -> None:
        print(f"[rank {rank}] {message}", flush=True)

    stage("process group initialized")
    torch.manual_seed(13)
    model = CoCondenserForPretraining(tiny_encoder(), MethodConfig(name="cocondenser"))
    input_ids = torch.randint(5, 30, (2, 8))
    input_ids[:, 0] = 2
    input_ids += rank % 2
    labels = torch.full_like(input_ids, -100)
    labels[:, 2] = input_ids[:, 2]
    input_ids[:, 2] = 4
    stage("coCondenser forward")
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )
    assert output.contrastive_loss is not None and torch.isfinite(output.contrastive_loss)
    output.loss.backward()
    assert model.encoder.bert.embeddings.word_embeddings.weight.grad is not None
    stage("coCondenser complete")

    contriever = ContrieverForPretraining(
        tiny_encoder(),
        MethodConfig(
            name="contriever",
            queue_size=8,
            contrastive_temperature=0.05,
            normalize_embeddings=True,
        ),
    )
    stage("Contriever DDP setup")
    distributed_contriever = DistributedDataParallel(contriever)
    query_ids = torch.randint(5, 30, (2, 8))
    key_ids = torch.randint(5, 30, (2, 8)) + rank % 2
    for _ in range(2):
        contriever_output = distributed_contriever(
            query_input_ids=query_ids,
            query_attention_mask=torch.ones_like(query_ids),
            key_input_ids=key_ids,
            key_attention_mask=torch.ones_like(key_ids),
        )
        assert torch.isfinite(contriever_output.loss)
        contriever_output.loss.backward()
    assert contriever.queue_ptr.item() == 0
    rank_zero_queue = contriever.queue.clone()
    dist.broadcast(rank_zero_queue, src=0)
    assert torch.equal(contriever.queue, rank_zero_queue)
    assert contriever.encoder.bert.embeddings.word_embeddings.weight.grad is not None
    stage("Contriever complete")

    for model_class, method_name in (
        (MNRLForPretraining, "mnrl"),
        (CachedMNRLForPretraining, "cmnrl"),
    ):
        stage(f"{method_name.upper()} DDP setup")
        ranking_model = model_class(
            tiny_encoder(),
            MethodConfig(
                name=method_name,
                mnrl_gather_across_devices=True,
                cmnrl_mini_batch_size=1,
            ),
        )
        distributed_ranking = DistributedDataParallel(ranking_model)
        anchor_ids = torch.randint(5, 30, (2, 8))
        candidate_ids = torch.randint(5, 30, (2, 2, 8)) + rank % 2
        ranking_output = distributed_ranking(
            anchor_input_ids=anchor_ids,
            anchor_attention_mask=torch.ones_like(anchor_ids),
            candidate_input_ids=candidate_ids,
            candidate_attention_mask=torch.ones_like(candidate_ids),
        )
        assert ranking_output.mnrl_loss is not None
        assert torch.isfinite(ranking_output.mnrl_loss)
        ranking_output.loss.backward()
        assert ranking_model.encoder.bert.embeddings.word_embeddings.weight.grad is not None
        stage(f"{method_name.upper()} complete")

    stage("SimCSE DDP setup")
    simcse = SimCSEForPretraining(tiny_encoder(), MethodConfig(name="simcse"))
    distributed_simcse = DistributedDataParallel(simcse)
    sentence_ids = torch.randint(5, 30, (2, 2, 8)) + rank % 2
    sentence_ids[1] = sentence_ids[0]
    simcse_output = distributed_simcse(
        input_ids=sentence_ids,
        attention_mask=torch.ones_like(sentence_ids),
    )
    assert simcse_output.contrastive_loss is not None
    assert torch.isfinite(simcse_output.contrastive_loss)
    simcse_output.loss.backward()
    assert simcse.projection.weight.grad is not None
    assert simcse.encoder.bert.embeddings.word_embeddings.weight.grad is not None
    stage("SimCSE complete")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
