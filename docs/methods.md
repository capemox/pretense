# Method notes

## RetroMAE

The encoder receives a moderately corrupted view. A shallow decoder reconstructs content tokens
from the encoder's final first-token representation and a separately, aggressively restricted view
of the original tokens. Defaults are 30% encoder masking and 50% decoder attention masking.

## DupMAE

DupMAE adds a second representation path. Vocabulary logits from ordinary encoder tokens are
max-pooled, then trained against the normalized bag of words. Its default loss weight is 0.1.

## Condenser

The final sentence token is joined to ordinary token states from the midpoint of the encoder. Two
additional Transformer layers perform MLM, forcing information required by token prediction into
the sentence representation. Late backbone MLM is enabled by default.

## coCondenser

Two spans are sampled from each document and kept adjacent in the batch. Their final sentence
representations are positives; all other spans in the global distributed batch are negatives.
Similarity is an unnormalized dot product with temperature 1.0, matching the reference objective.

## Contriever

Two independently cropped and augmented views are sampled from every document. The query view is
mean-pooled by the online encoder; the key view is mean-pooled by an exponential-moving-average
momentum encoder. The positive key is contrasted against a persistent MoCo queue of keys from
earlier batches. The queue and momentum encoder are included in resumable Pretense checkpoints but
omitted from clean exports.

The `delete`, `mask`, `replace`, and `shuffle` token augmentations from the reference implementation
are supported, along with `none`. The included recipe uses the published English setup: deletion at
probability 0.1, crop ratios from 0.1 to 0.5, momentum 0.9995, queue size 131,072, and temperature
0.05. These are expensive paper-scale settings; smaller queues and fewer steps are useful for
development but change the objective's negative distribution.

Contriever uses attention-mask-aware mean pooling. Its Sentence Transformers export preserves that
pooling choice. Optional embedding normalization is applied consistently during training and
included as a `Normalize` export module.

## Contrastive

The supervised pairwise objective implements the same classic margin equation as Sentence
Transformers' `ContrastiveLoss`:

```text
0.5 * (label * distance² + (1 - label) * max(margin - distance, 0)²)
```

Labels are `1` for similar pairs and `0` for dissimilar pairs. Cosine distance is the default;
Euclidean and Manhattan distances are also available. The default margin is `0.5`. Both texts are
encoded by the same backbone and attention-mask-aware mean pooling is used for training and the
Sentence Transformers export.

Configure the input column names independently:

```yaml
method:
  name: contrastive
  contrastive_distance_metric: cosine
  contrastive_margin: 0.5
data:
  text_column: sentence1
  text_pair_column: sentence2
  label_column: label
```

This method is useful for labeled or weakly labeled pairs. For unlabeled documents, Contriever is
the native unsupervised contrastive option. For retrieval-positive pairs where every other item in
the batch should act as a negative, use MNRL or CMNRL.

The embedding-level loss is also public for custom workflows:

```python
from pretense import ContrastiveLoss

objective = ContrastiveLoss(distance_metric="cosine", margin=0.5)
loss = objective(anchor_embeddings, other_embeddings, labels)
```

## MNRL and CMNRL

Multiple Negatives Ranking Loss (MNRL) trains aligned query/positive pairs with a sampled-softmax
objective. For each query, its aligned positive is the target and every other positive in the batch
acts as a negative. Any configured explicit negative columns are appended to that candidate pool.
The default is cosine similarity scaled by `20.0`; unnormalized dot-product similarity is also
available.

```yaml
method:
  name: mnrl
  mnrl_scale: 20.0
  mnrl_similarity: cosine
  mnrl_gather_across_devices: false
data:
  text_column: query
  text_pair_column: positive
  negative_columns: [hard_negative]
```

Set `mnrl_gather_across_devices: true` under distributed training to include candidate columns from
every process. All processes must then use the same local batch size; dropping the last incomplete
batch is recommended. Cross-device gathering increases communication and the effective number of
negatives.

Cached MNRL (CMNRL) has the same scores and gradients but uses GradCache to encode each large
Trainer batch in smaller chunks:

```yaml
method:
  name: cmnrl
  mnrl_scale: 20.0
  cmnrl_mini_batch_size: 32
training:
  per_device_train_batch_size: 256
```

Here `256` determines the local in-batch candidate pool, while `32` bounds encoder activation
memory. CMNRL performs an extra no-gradient encoding pass and is therefore slower than ordinary
MNRL when both fit in memory. It preserves dropout randomness during the gradient replay.

The data should not repeat a positive (or a query that is also another row's positive) within the
same batch unless that collision is genuinely negative. MNRL cannot distinguish such false
negatives. Pretense does not silently deduplicate rows because doing so would change user-supplied
sampling and distributed batch sizes.

The embedding-level ordinary loss is also public:

```python
from pretense import MultipleNegativesRankingLoss

objective = MultipleNegativesRankingLoss(scale=20.0, similarity="cosine")
loss = objective(anchor_embeddings, positive_embeddings, hard_negative_embeddings)
```

## Compatibility

The BERT and RoBERTa paths reproduce the paper-era model behavior. ModernBERT and DeBERTa-v3 use
their native embeddings and MLM heads with family-neutral auxiliary Transformer layers. These are
objective-faithful adaptations, not claims that those later architectures appeared in the papers.
Token-level masking is used consistently across tokenizers; whole-word masking can be added in a
later release without changing model interfaces. Contriever, pairwise contrastive training, MNRL,
and CMNRL only require an MLM head because that is the common model interface used by Pretense;
their objectives consume backbone hidden states and do not compute MLM loss.
