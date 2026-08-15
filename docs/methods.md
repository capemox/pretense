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

Contriever uses attention-mask-aware mean pooling. Its Sentence Transformers export therefore uses
mean pooling instead of the CLS pooling used by the other Pretense methods. Optional embedding
normalization is applied consistently during training and included as a `Normalize` export module.

## Compatibility

The BERT and RoBERTa paths reproduce the paper-era model behavior. ModernBERT and DeBERTa-v3 use
their native embeddings and MLM heads with family-neutral auxiliary Transformer layers. These are
objective-faithful adaptations, not claims that those later architectures appeared in the papers.
Token-level masking is used consistently across tokenizers; whole-word masking can be added in a
later release without changing model interfaces. Contriever only requires an MLM head because that
is the common model interface used by Pretense; its contrastive objective itself consumes backbone
hidden states and does not compute MLM loss.
