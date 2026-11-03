"""Test matrices with a known spectrum."""

import torch


def random_orthogonal(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    # fix column signs so Q is Haar-distributed, not biased by the QR convention
    return q * torch.sign(torch.diagonal(r))


def sym_with_spectrum(eigs: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Symmetric matrix Q diag(eigs) Q^T with a random orthogonal Q."""
    q = random_orthogonal(eigs.numel(), seed)
    return (q * eigs.to(torch.float64)) @ q.T


def with_singular_values(m: int, n: int, sv: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """m x n matrix U diag(sv) V^T with random orthonormal U, V."""
    k = sv.numel()
    u = random_orthogonal(m, seed)[:, :k]
    v = random_orthogonal(n, seed + 1)[:, :k]
    return (u * sv.to(torch.float64)) @ v.T
