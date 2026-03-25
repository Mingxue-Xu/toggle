# Algebra Backends

`ModelConsolidator` exposes two different backend knobs:

| Compression methods | Kwarg | Supported values |
|---------------------|-------|------------------|
| SVD | `svd_backend` | `"torch"` / `"pytorch"`, `"cola"` |
| Tucker / CP / Tensor Train | `backend` | `"pytorch"` / `"torch"` |

In practice, the code currently uses three algebra implementations:

- PyTorch SVD
- CoLA SVD
- TensorLy on the PyTorch backend

In `.yaml` config files under `config/`, the backend settings usually look like this:

```yaml
# SVD backend choice (resolved internally into svd_backend / svd_backend_config)
compression:
  svd:
    backend: cola
    cola:
      algorithm: lanczos

# TensorLy-backed decomposition choice
factorization:
  func_name: tucker
  tucker_ranks: [64, 64]
  device: cuda
  backend: pytorch
```



## PyTorch (default)

Uses `torch.linalg.svd`. No extra algebra dependency beyond PyTorch itself.

```python
# From tests/e2e/test_e2e_algebra_backends.py::TestPyTorchBackend
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=["model.layers[0].mlp.gate_proj"],
    rank=32,
    svd_backend="torch",                          # default if omitted
)
```

Can also be used directly via `build_svd_backend` in `svd_backend.py`:

```python
from goldcrest.plugins.compression.svd_backend import build_svd_backend, TorchSVDBackend

backend = build_svd_backend("torch", {})          # returns TorchSVDBackend
U, S, Vh = backend.compute_svd(weight, k=16)      # truncated SVD
```

## CoLA

Requires an importable `cola` package, typically via `pip install cola-ml`.
Select the CoLA algorithm with `svd_backend_config["algorithm"]`:

| Algorithm | Config | Backend class used |
|-----------|--------|--------------------|
| `dense` | `{"algorithm": "dense"}` | `DenseSVD()` |
| `lanczos` | `{"algorithm": "lanczos", "max_iters": 100, "tol": 1e-6}` | `Lanczos(...)` |
| `lobpcg` | `{"algorithm": "lobpcg", "max_iters": 50}` | `LOBPCG(max_iters=...)` |
| `auto` | `{"algorithm": "auto"}` | `Auto(...)` |

```python
# Dense, from tests/e2e/test_e2e_algebra_backends.py::TestCoLABackend
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=["model.layers[0].mlp.gate_proj"],
    rank=32,
    svd_backend="cola",
    svd_backend_config={"algorithm": "dense"},
)

# Lanczos, from tests/e2e/test_e2e_algebra_backends.py::TestLanczosBackend
consolidator = ModelConsolidator(
    compression_method="svd", target_modules=[...], rank=32,
    svd_backend="cola",
    svd_backend_config={"algorithm": "lanczos", "max_iters": 100, "tol": 1e-6},
)

# LOBPCG, from tests/e2e/test_e2e_algebra_backends.py::TestLOBPCGBackend
consolidator = ModelConsolidator(
    compression_method="svd", target_modules=[...], rank=32,
    svd_backend="cola",
    svd_backend_config={"algorithm": "lobpcg", "max_iters": 50},
)
```

`ColaSVDBackend` accepts `algorithm` and `which`.
From `svd_backend_config`, it only forwards the allowlisted keys
`start_vector`, `max_iters`, `tol`, `pbar`, and `key` into CoLA.
For `lobpcg`, only `max_iters` is used.

## TensorLy

Used by Tucker, CP, and Tensor Train decompositions. 
Requires `pip install tensorly tensorly-torch`.

For direct TensorLy usage, you can set the backend explicitly:

```python
# From tests/e2e/test_e2e_algebra_backends.py::TestTensorLyBackend
from goldcrest.plugins.compression.tensorly_backend import set_tensorly_backend
set_tensorly_backend("pytorch")                   # "pytorch" or "torch" accepted
```

`set_tensorly_backend()` lowercases the name, normalizes `"torch"` to
`"pytorch"`, calls `tl.set_backend()`, and raises `NotImplementedError`
for `"cola"`.

When using `ModelConsolidator`, you do not need to call
`set_tensorly_backend()` manually. The consolidator and the TensorLy-based
compression classes already do that internally.
Do not pass `backend="cola"` for Tucker/CP/TT; that path raises
`NotImplementedError`.

When using `ModelConsolidator` with Tucker/CP/TT, pass `backend="pytorch"`
(or `backend="torch"`, which is normalized to `"pytorch"`):

```python
# Tucker -- from tests/e2e/test_e2e_compression_algorithms.py::TestTuckerCompression
consolidator = ModelConsolidator(
    compression_method="tucker",
    target_modules=["model.layers[0].mlp.gate_proj"],
    tucker_ranks=[32, 32],
    backend="pytorch",
)

# CP -- from tests/e2e/test_e2e_compression_algorithms.py::TestCPCompression
consolidator = ModelConsolidator(
    compression_method="cp",
    target_modules=["model.layers[0].self_attn.q_proj"],
    cp_rank=16,
    backend="pytorch",
)

# Tensor Train -- from tests/e2e/test_e2e_compression_algorithms.py::TestTensorTrainCompression
consolidator = ModelConsolidator(
    compression_method="tensor_train",
    target_modules=["model.layers[0].mlp.gate_proj"],
    tensor_ranks=[1, 4, 1],
    backend="pytorch",
)
```

Important Tensor Train caveat:
for 2D weight matrices, the current `TensorTrain` implementation reshapes the
matrix to 3D and derives working TT ranks internally. That means the runtime
TT ranks are not guaranteed to match the exact `tensor_ranks` argument you
passed in.

## Summary

| Method | Backend kwarg | Config kwarg | Implementation |
|--------|---------------|--------------|----------------|
| SVD | `svd_backend="torch"` | `rank=...` or `preserve_energy=...` | `torch.linalg.svd` |
| SVD | `svd_backend="cola"` | `svd_backend_config={"algorithm": ...}` | CoLA `DenseSVD` / `Lanczos` / `LOBPCG` / `Auto` |
| Tucker | `backend="pytorch"` or `backend="torch"` | `tucker_ranks=[r1, r2]` | `tl.decomposition.tucker` |
| CP | `backend="pytorch"` or `backend="torch"` | `cp_rank=r` | `tl.decomposition.parafac` |
| TT | `backend="pytorch"` or `backend="torch"` | `tensor_ranks=[...]` | `tl.decomposition.tensor_train` after internal reshape/rank derivation for 2D matrices |
