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

Input can take any of three forms:

- a list of spans in `spans_column`
- individual span rows grouped by `document_id_column`
- full documents in `text_column`, which Pretense splits into non-overlapping maximum-length spans

Every document must provide at least two spans. Gradient accumulation must be `1` because separate
microbatches cannot reproduce a single global in-batch negative pool. Pretense also enables
`dataloader_drop_last` automatically so distributed processes receive equal batch sizes.

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

## Exported sentence representations

- RetroMAE, DupMAE, Condenser, and coCondenser use the first token (the CLS-equivalent).
- Contriever, pairwise contrastive training, MNRL, and CMNRL use attention-mask-aware mean pooling.
- Contriever exports include normalization when `normalize_embeddings` was enabled during
  pretraining.

## Architecture compatibility

BERT, RoBERTa, ModernBERT, and DeBERTa-v3 (the Transformers `deberta-v2` model type) are certified.
Other masked language models can be supported with `BackboneAdapter`; unpublished models can be
passed directly through `create_pretraining_model`. See the [custom-model guide](custom-models.md).

The BERT and RoBERTa implementations reproduce the paper-era architecture. ModernBERT and
DeBERTa-v3 use their native embeddings and MLM heads with family-neutral auxiliary Transformer
layers. These are objective-faithful adaptations; the later architectures did not appear in the
original papers.

Pretense uses token-level masking across tokenizers. Contriever, pairwise contrastive training,
MNRL, and CMNRL do not compute MLM loss, although their models still use Pretense's common
masked-language-model interface.
