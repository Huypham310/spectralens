"""Singular-value spectra of weight matrices, compared with a random-matrix baseline.

For a weight matrix W (m x n, m <= n after transposing) the empirical spectral density (ESD) is the
distribution of the eigenvalues of W W^T / n, i.e. squared singular values / n. If W were pure
i.i.d. noise with variance sigma^2, Marchenko-Pastur says that density lives on
[sigma^2 (1 - sqrt(q))^2, sigma^2 (1 + sqrt(q))^2] with q = m / n. Training adds structure, which
shows up as eigenvalues beyond that edge and as a heavy (power-law) tail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass
class LayerSpectrum:
    name: str
    shape: tuple[int, int]
    spectral_norm: float
    stable_rank: float  # ||W||_F^2 / ||W||_2^2, between 1 and min(m, n)
    effective_rank: float  # exp(entropy of normalized singular values), Roy & Vetterli 2007
    outlier_fraction: float  # eigenvalues above the Marchenko-Pastur bulk edge
    alpha: float  # Hill estimate of the ESD tail exponent (smaller = heavier tail)

    def as_dict(self) -> dict:
        return asdict(self)


def layer_matrices(state_dict: dict[str, Tensor], min_dim: int = 8) -> dict[str, Tensor]:
    """The 2-D view of every weight that acts like a matrix.

    Conv kernels (out, in, kh, kw) become (out, in * kh * kw); attention's packed in_proj is split
    into its q / k / v blocks. Biases, norm gains and positional embeddings are skipped, and so
    are matrices with a side shorter than `min_dim` (a spectrum of 3 values says nothing).
    """
    out: dict[str, Tensor] = {}
    for name, w in state_dict.items():
        if not name.endswith("weight") or w.ndim < 2:
            continue
        if w.ndim == 4:
            w = w.reshape(w.shape[0], -1)
        if name.endswith("in_proj_weight"):
            # nn.MultiheadAttention stores [W_q; W_k; W_v] stacked along dim 0
            for part, block in zip("qkv", w.chunk(3, dim=0), strict=True):
                out[f"{name}.{part}"] = block
            continue
        if min(w.shape) >= min_dim:
            out[name] = w
    return {k: v for k, v in out.items() if min(v.shape) >= min_dim}


def mp_edge(sigma2: float, m: int, n: int) -> float:
    """Upper edge of the Marchenko-Pastur bulk for eigenvalues of W W^T / n (W is m x n, m <= n)."""
    q = min(m, n) / max(m, n)
    return sigma2 * (1.0 + q**0.5) ** 2


def hill_alpha(eigs: Tensor, k: int | None = None) -> float:
    """Hill estimator of the power-law exponent of the top-`k` eigenvalues.

    For a density p(x) ~ x^-alpha, alpha = 1 + k / sum(log(x_i / x_k)). Uses the largest 10% of the
    eigenvalues by default (at least 5).
    """
    x = torch.sort(eigs.double(), descending=True).values
    x = x[x > 0]
    if x.numel() < 6:
        return float("nan")
    k = k or max(5, x.numel() // 10)
    k = min(k, x.numel() - 1)
    logs = torch.log(x[:k] / x[k])
    s = float(logs.sum())
    return 1.0 + k / s if s > 0 else float("nan")


def spectrum_of(name: str, w: Tensor) -> LayerSpectrum:
    w = w.detach().double()
    if w.shape[0] > w.shape[1]:
        w = w.T  # m <= n
    m, n = w.shape
    s = torch.linalg.svdvals(w)
    eigs = s**2 / n
    fro2 = float((s**2).sum())
    spec = float(s[0])
    p = s / s.sum()
    entropy = float(-(p * torch.log(p.clamp_min(1e-300))).sum())

    # sigma^2 estimated from the whole matrix: for pure noise it's the true variance; with a few
    # big outliers it's slightly inflated, which only makes the outlier count conservative
    sigma2 = fro2 / (m * n)
    edge = mp_edge(sigma2, m, n)
    return LayerSpectrum(
        name=name,
        shape=(m, n),
        spectral_norm=spec,
        stable_rank=fro2 / spec**2,
        effective_rank=float(torch.exp(torch.tensor(entropy))),
        outlier_fraction=float((eigs > edge * 1.0001).double().mean()),
        alpha=hill_alpha(eigs),
    )


def network_spectra(state_dict: dict[str, Tensor]) -> list[LayerSpectrum]:
    return [spectrum_of(name, w) for name, w in layer_matrices(state_dict).items()]


def summarize(spectra: list[LayerSpectrum]) -> dict[str, float]:
    """Layer-averaged numbers for one checkpoint (median alpha: a few layers can blow it up)."""

    def mean(key: str) -> float:
        return float(sum(getattr(s, key) for s in spectra) / len(spectra))

    alphas = torch.tensor([s.alpha for s in spectra if s.alpha == s.alpha])
    return {
        "layers": len(spectra),
        "stable_rank": mean("stable_rank"),
        # relative to the layer's size, so big and small layers count the same
        "stable_rank_ratio": float(
            sum(s.stable_rank / min(s.shape) for s in spectra) / len(spectra)
        ),
        "effective_rank_ratio": float(
            sum(s.effective_rank / min(s.shape) for s in spectra) / len(spectra)
        ),
        "outlier_fraction": mean("outlier_fraction"),
        "alpha_median": float(alphas.median()) if alphas.numel() else float("nan"),
    }
