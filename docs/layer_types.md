# Compression for Different Layer Types

## Verified feature matrix

Based on passing e2e tests (`tests/e2e-codex/test_target_layers.py`,
`tests/e2e/test_e2e_target_layers.py`).
`✓` means verified by passing test, `—` means not tested.

| Method       | MLP | Attention | Embedding (token-level) |
|--------------|:------------:|:------------------:|:-----------------------:|
| SVD          | ✓            | ✓                  | ✓                       |
| Tucker       | ✓            | ✓                  | ✓                       |
| CP           | ✓            | ✓                  | ✓                       |
| Tensor Train | ✓            | ✓                  | ✓                       |

---

## Embedding Layer Compression

### Token-level Embedding Layer Compression

The codebase currently does **not** support embedding layers compressed with
matrix-level compression approach. 
Embedding layers are compressed through each token vector, which is reshaped to matrix or higher-dimensional tensors, such that it can easily map to token embeddings during the forward pass.  
However, a good forward strategy for
mapping token ids back hasn't been decided, and
matrix-level embeddings compression is unsupported for now. 


### Weight Tying

Many causal LMs (including Gemma) tie the embedding layer and the language model
head (`lm_head`) so they share the same weight tensor. PyTorch's
`model.parameters()` deduplicates shared parameters, so the tied weight is
counted once.

#### Impact on embedding compression for this codebase

When `ModelConsolidator` replaces `model.embed_tokens` with a `FactorLinear`,
the tie is broken. The old weight tensor is no longer shared — it remains in
`lm_head` as a standalone parameter. The result:

- The compressed embedding factors are added (small).
- The untied `lm_head.weight` is now counted separately (large).
- Total parameter count **increases** instead of decreasing.

To avoid this, either compress both `model.embed_tokens` and `lm_head`
together, or re-tie the output head to the reconstructed embedding after
compression.

#### Is Weight Tying Always a Good Choice?

Not always. Weight tying is not ideal when the input and output distributions need to learn
different representations:

1. **Sequence-to-sequence / translation** — The source and target vocabularies
   may overlap but serve different roles. The encoder embedding and decoder
   output head benefit from independent specialization.

2. **Multi-task heads** — When `lm_head` is replaced with task-specific heads
   (classification, regression, token labeling), those heads have different
   output dimensions and semantics than the embedding lookup.

3. **Retrieval / contrastive learning** — The embedding is trained to produce
   query representations while the output head may need to produce key
   representations in a different metric space.

4. **Knowledge distillation** — The student's embedding and output head may
   need to adapt independently to match different teacher signals at the input
   vs output side.

5. **Vocabulary extension / domain adaptation** — When you add new tokens for a
   specialized domain, the embedding for new tokens needs training from scratch
   while the output projection for existing tokens should stay stable. Tying
   them forces both to move together.

6. **Asymmetric architectures** — When `embed_dim != hidden_dim` at the output
   (e.g., with a projection layer before the head), tying is structurally
   impossible.

7. **Post-compression** — After SVD compresses the embedding to low-rank
   factors, the reconstruction `U @ S @ Vt` is a lossy approximation. Tying
   `lm_head` to this approximation forces the output logits through the same
   low-rank bottleneck, which can degrade generation quality more than
   compressing them independently at different ranks would.

In practice, for standard causal LM pretraining/fine-tuning on a single
language, weight tying works well and saves memory (one copy of the vocab-sized matrix instead of two). The cases above are where untying becomes worth the extra parameters.
