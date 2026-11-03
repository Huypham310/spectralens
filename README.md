# SpectraLens

A linear-algebra lens on vision networks.

SpectraLens is a research-style project that looks at image classifiers through their
matrices: the singular values of their weights, the eigenvalues of their loss Hessian, and how
much of each layer can be thrown away with a low-rank approximation before accuracy drops.

All of the numerical linear algebra is written from scratch in PyTorch (power iteration,
Lanczos, randomized SVD, stochastic trace estimation, Hessian-vector products) and checked
against `torch.linalg` / SciPy before it is used on real models.

> Status: early work in progress. The sections below describe the plan; results will be added
> as each part lands.

## Questions

1. **Weight spectra.** How do the singular-value spectra of conv and linear layers evolve during
   training, and how far do they drift from a random (Marchenko-Pastur) baseline?
2. **Curvature.** What does the Hessian eigenspectrum of a trained CNN look like, and how do the
   top eigenvalues change with batch size and with sharpness-aware training (SAM)?
3. **Compression.** How much accuracy is lost per FLOP saved when layers are replaced by their
   truncated SVD (linear) or Tucker (conv) factorizations, and does the spectrum predict which
   layers are safe to compress?

## Linear algebra toolkit

`spectralens.linalg` works on dense tensors or on any matrix-free `LinearOperator`
(a function `v -> A @ v`), so the same code runs on a weight matrix or a network Hessian.

| Routine | What it does |
|---|---|
| `power_iteration` | top-k eigenpairs by magnitude, with implicit deflation |
| `lanczos`, `lanczos_eigsh` | Krylov tridiagonalization with full reorthogonalization and breakdown detection; extreme Ritz pairs (`LA` / `SA` / `LM`) |
| `randomized_svd` | Halko-Martinsson-Tropp range finder with oversampling and power iterations |
| `hutchinson`, `hutchpp` | stochastic trace estimation (Rademacher / Gaussian probes), Hutch++ |
| `hessian_operator` | Hessian-vector products by double backprop, never forming the Hessian |

Every routine is tested against `torch.linalg` or SciPy (including `scipy.sparse.linalg.eigsh`
on a sparse Laplacian), and the randomized SVD error is checked against the HMT 2011 bound
with Hypothesis-generated matrices.

Benchmark on a 4096 x 4096 float32 matrix, RTX 5070 Ti Laptop GPU
(`python scripts/bench_linalg.py --device cuda`):

| Method | Time (ms) | Error |
|---|---:|---:|
| full SVD (`torch.linalg.svd`) | 5633.6 | 1 |
| randomized SVD, rank 64, q=0 | 5.5 | 2.67 |
| randomized SVD, rank 64, q=2 | 8.6 | 1.02 |
| dense `torch.linalg.eigvalsh` | 248.0 | 4.8e-6 |
| Lanczos top-10, 60 steps | 28.4 | 3.8e-6 |

SVD error is the spectral-norm residual divided by the optimal one, sigma_65 (1.0 means
optimal), on a spectrum sigma_i = 1/i. Eigenvalue error is the largest absolute error of the
top 10. Two power iterations bring the randomized SVD within 2% of optimal while staying
about 650x faster than the full decomposition.

## Planned layout

```
src/spectralens/
  linalg/     power iteration, Lanczos, randomized SVD, trace estimators, HVPs
  data/       EuroSAT + CIFAR-10 loaders
  models/     small CNN, ResNet-18, tiny ViT
  analysis/   weight spectra, Hessian spectra, rank metrics
  compress/   SVD / Tucker factorized layers, LoRA
configs/      Hydra experiment configs
paper/        LaTeX write-up
tests/        unit + property-based tests
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
python -m spectralens info
```

For GPU training install a CUDA build of PyTorch first, following the selector on
[pytorch.org](https://pytorch.org/get-started/locally/).

## License

MIT
