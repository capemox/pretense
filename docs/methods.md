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

## Compatibility

The BERT and RoBERTa paths reproduce the paper-era model behavior. ModernBERT and DeBERTa-v3 use
their native embeddings and MLM heads with family-neutral auxiliary Transformer layers. These are
objective-faithful adaptations, not claims that those later architectures appeared in the papers.
Token-level masking is used consistently across tokenizers; whole-word masking can be added in a
later release without changing model interfaces.
