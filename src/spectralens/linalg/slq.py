"""Stochastic Lanczos quadrature (SLQ) for the spectral density of a symmetric operator.

For a unit probe v, m Lanczos steps give T = S diag(theta) S^T, and the Gauss quadrature
rule with nodes theta_i and weights tau_i = S[0, i]^2 integrates any f against the
"local" spectral measure of v:  v^T f(A) v ~ sum_i tau_i f(theta_i). Averaging over random
probes (E[v v^T] = I / n) turns that into the full density
    phi(t) = (1/n) sum_j delta(t - lambda_j),
which we smooth with a Gaussian kernel for plotting (Ghorbani, Krishnan & Xiao, 2019;
Lin, Saad & Yang, 2016).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from spectralens.linalg.lanczos import lanczos
from spectralens.linalg.operators import LinearOperator, aslinearoperator
from spectralens.linalg.trace import probes


@dataclass
class SpectralDensity:
    """Quadrature nodes and weights, one row per probe. Each row of ``weights`` sums to 1."""

    nodes: Tensor
    weights: Tensor
    dim: int

    @property
    def num_probes(self) -> int:
        return self.nodes.shape[0]

    def trace(self) -> float:
        """tr(A) = n * E[v^T A v]: the quadrature applied to f(t) = t."""
        return self.dim * (self.weights * self.nodes).sum(dim=1).mean().item()

    def moment(self, k: int) -> float:
        """(1/n) tr(A^k), the k-th moment of the density."""
        return (self.weights * self.nodes**k).sum(dim=1).mean().item()

    def evaluate(self, grid: Tensor, sigma: float) -> Tensor:
        """Density smoothed with a Gaussian of width ``sigma``, on the points in ``grid``.

        Integrates to ~1 over a grid that covers all nodes by a few sigma.
        """
        grid = grid.to(self.nodes)
        diff = grid[:, None, None] - self.nodes[None]  # (g, probes, m)
        kernel = torch.exp(-0.5 * (diff / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        return (kernel * self.weights[None]).sum(dim=2).mean(dim=1)

    def as_dict(self) -> dict:
        return {"dim": self.dim, "nodes": self.nodes.tolist(), "weights": self.weights.tolist()}


def slq(
    a: Tensor | LinearOperator,
    num_steps: int,
    num_probes: int = 4,
    generator: torch.Generator | None = None,
) -> SpectralDensity:
    """Run ``num_probes`` Lanczos runs of ``num_steps`` steps from Rademacher starts.

    m steps make the quadrature exact for polynomials up to degree 2m - 1, so the extreme
    eigenvalues (which Lanczos finds first) and the low moments are both resolved well. Runs
    that break down early are padded with zero-weight nodes so every row has the same length.
    """
    op = aslinearoperator(a)
    n = op.shape[0]
    if op.shape[0] != op.shape[1]:
        raise ValueError(f"SLQ needs a square operator, got {op.shape}")
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1")
    num_steps = min(num_steps, n)

    nodes = torch.zeros(num_probes, num_steps, dtype=op.dtype)
    weights = torch.zeros(num_probes, num_steps, dtype=op.dtype)
    for i in range(num_probes):
        v = probes(n, 1, "rademacher", op.dtype, op.device, generator)[:, 0]
        res = lanczos(op, num_steps, v0=v)
        theta, s = torch.linalg.eigh(res.tridiagonal())
        m = theta.numel()
        nodes[i, :m] = theta.cpu()
        # first component of each Ritz vector, squared: sums to 1 since S is orthogonal
        weights[i, :m] = (s[0] ** 2).cpu()
    return SpectralDensity(nodes=nodes, weights=weights, dim=n)
