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
