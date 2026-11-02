"""Randomized SVD (Halko, Martinsson & Tropp, 2011).

Idea: if A has (approximately) rank k, then A @ Omega for a random Gaussian Omega with
k + p columns almost surely spans the top-k range of A. Orthonormalize that sample to get Q,
project B = Q^T A (small), and take the exact SVD of B. Only 2(q + 1) passes over A are
needed, each a product with a tall-skinny matrix, which is why this beats a full SVD on
big weight matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from spectralens.linalg.operators import LinearOperator, aslinearoperator


@dataclass
class SVDResult:
    """Truncated SVD A ~ U diag(S) Vh with U (m, k), S (k,), Vh (k, n)."""

    U: Tensor
    S: Tensor
    Vh: Tensor

    def reconstruct(self) -> Tensor:
        return (self.U * self.S) @ self.Vh


def _orth(y: Tensor) -> Tensor:
    q, _ = torch.linalg.qr(y, mode="reduced")
    return q


def randomized_range_finder(
    a: Tensor | LinearOperator,
    size: int,
    n_iter: int = 2,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Orthonormal Q (m, size) whose range approximates the top-``size`` range of A.

    ``n_iter`` subspace (power) iterations replace A by (A A^T)^q A, which has the same
    singular vectors but singular values raised to 2q + 1, so a slowly decaying spectrum
    becomes a fast decaying one. We re-orthonormalize after every product; without that
    the columns collapse onto the top singular vector in floating point.
    """
    op = aslinearoperator(a)
    m, n = op.shape
    # Gaussian test matrix, drawn on CPU so a seed gives the same result on any device
    omega = torch.randn(n, size, dtype=op.dtype, generator=generator).to(op.device)
    q = _orth(op.matmat(omega))
    for _ in range(n_iter):
        q = _orth(op.rmatmat(q))
        q = _orth(op.matmat(q))
    return q


def randomized_svd(
    a: Tensor | LinearOperator,
    rank: int,
    oversample: int = 10,
    n_iter: int = 2,
    generator: torch.Generator | None = None,
) -> SVDResult:
    """Rank-``rank`` approximate SVD of A.

    ``oversample`` extra columns make the range capture robust (p = 5-10 is the usual
    advice); ``n_iter`` power iterations sharpen it for flat spectra.
    """
    op = aslinearoperator(a)
    m, n = op.shape
    if not 1 <= rank <= min(m, n):
        raise ValueError(f"rank must be in [1, {min(m, n)}], got {rank}")
    size = min(rank + oversample, min(m, n))

    q = randomized_range_finder(op, size, n_iter=n_iter, generator=generator)
    # B = Q^T A, computed as (A^T Q)^T so we only need the transpose action
    b = op.rmatmat(q).T
    ub, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ ub
    return SVDResult(U=u[:, :rank], S=s[:rank], Vh=vh[:rank])


def halko_error_bound(rank: int, oversample: int, n_iter: int, min_dim: int) -> float:
    """Factor C such that E ||A - Q Q^T A||_2 <= C * sigma_{rank+1} (HMT 2011, Cor. 10.10).

    C = (1 + 4 sqrt(k + p) / (p - 1) * sqrt(min(m, n)))^(1 / (2q + 1)). Needs p >= 2.
    """
    if oversample < 2:
        raise ValueError("the bound needs oversample >= 2")
    base = 1 + 4 * math.sqrt(rank + oversample) / (oversample - 1) * math.sqrt(min_dim)
    return base ** (1 / (2 * n_iter + 1))
