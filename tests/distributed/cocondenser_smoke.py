"""Two-process CPU smoke test, invoked explicitly by CI rather than pytest."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import BertConfig, BertForMaskedLM

from pretense import CoCondenserForPretraining, ContrieverForPretraining, MethodConfig


def main() -> None:
    dist.init_process_group("gloo")
    rank = int(os.environ["RANK"])
    torch.manual_seed(13)
    encoder = BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    model = CoCondenserForPretraining(encoder, MethodConfig(name="cocondenser"))
    input_ids = torch.randint(5, 30, (2, 8))
    input_ids[:, 0] = 2
    input_ids += rank % 2
    labels = torch.full_like(input_ids, -100)
    labels[:, 2] = input_ids[:, 2]
    input_ids[:, 2] = 4
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )
    assert output.contrastive_loss is not None and torch.isfinite(output.contrastive_loss)
    output.loss.backward()
    assert model.encoder.bert.embeddings.word_embeddings.weight.grad is not None

    contriever = ContrieverForPretraining(
        encoder,
        MethodConfig(
            name="contriever",
            queue_size=8,
            contrastive_temperature=0.05,
            normalize_embeddings=True,
        ),
    )
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
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
