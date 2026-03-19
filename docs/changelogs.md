# Changelogs

## 2026-03-18

- Initial public release.
- Compression: SVD, ASVD, SVD-LLM, Tucker, CP, Tensor-Train, pruning, KV-cache.
- Analysis: activation metrics, weight metrics, Fisher information, layer sensitivity.
- Evaluation: `lm_eval` integration.
- Configuration: YAML-driven pipelines with defaults.
- Runtime: H100/H200 scripts, CPU scripts, memory profiling, latency measurement.
- Integrations: CoLA, TensorLy, `information_flow`.
- Docs: remote runs guide and benchmark results.

## 2026-03-19

- `LMHarness` now defaults to safe in-memory evaluation instead of silently reloading checkpoints.
- `WorkflowExecutor` now separates constructor config from runtime kwargs and forwards `context`, `model`, and `tokenizer` correctly.
- Evaluation and compression plugins were brought back onto the framework `do_execute()` lifecycle.
- Package-safe imports were repaired, including the import-time break in `compressed_eval.py`.
- The main Hugging Face loader path now normalizes auth kwargs and uses `AutoModelForCausalLM`.
- Critical paths now have pytest coverage, including a toy end-to-end workflow.
