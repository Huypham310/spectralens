"""Lanczos tridiagonalization and Ritz eigenvalue estimates for symmetric operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from spectralens.linalg.operators import LinearOperator, aslinearoperator, random_start
from spectralens.linalg.power import EigResult

Which = Literal["LA", "SA", "LM"]


@dataclass
class LanczosResult:
    """Output of ``m`` Lanczos steps: A Q = Q T + (residual) e_m^T.

    ``alpha`` is the diagonal of T (length m), ``beta`` the off-diagonal (length m - 1) and
    ``basis`` the orthonormal Krylov basis Q with shape (n, m).
    """

    alpha: Tensor
    beta: Tensor
    basis: Tensor

    @property
    def steps(self) -> int:
        return self.alpha.numel()

    def tridiagonal(self) -> Tensor:
        """The m x m tridiagonal matrix T = Q^T A Q."""
        t = torch.diag(self.alpha)
        if self.beta.numel():
            t = t + torch.diag(self.beta, 1) + torch.diag(self.beta, -1)
        return t


def lanczos(
    a: Tensor | LinearOperator,
    num_steps: int,
    v0: Tensor | None = None,
    reorth: bool = True,
    generator: torch.Generator | None = None,
) -> LanczosResult:
    """Run ``num_steps`` steps of the Lanczos process on a symmetric operator.

    In exact arithmetic the three-term recurrence keeps Q orthonormal. In floating point it
    does not: once a Ritz value converges, Q loses orthogonality and copies of that eigenvalue
    ("ghosts") appear in T. ``reorth=True`` fixes this with full reorthogonalization against
    every previous vector, done twice (classical Gram-Schmidt, "twice is enough").
    That costs O(n m) memory and O(n m^2) time on top of the m matvecs.

    If the Krylov space becomes invariant before ``num_steps`` (breakdown), the run stops
    early and the result has fewer steps.
    """
    op = aslinearoperator(a)
    n = op.shape[0]
    if op.shape[0] != op.shape[1]:
        raise ValueError(f"Lanczos needs a square operator, got {op.shape}")
    if not 1 <= num_steps <= n:
        raise ValueError(f"num_steps must be in [1, {n}], got {num_steps}")

    if v0 is None:
        q = random_start(n, op.dtype, op.device, generator)
    else:
        q = v0.to(dtype=op.dtype, device=op.device)
        q = q / torch.linalg.vector_norm(q)

    # TODO: với Hessian của ResNet (n ~ 1e7) basis n x m không vừa GPU, cần option để trên CPU
    basis = torch.zeros(n, num_steps, dtype=op.dtype, device=op.device)
    alpha = torch.zeros(num_steps, dtype=op.dtype, device=op.device)
    beta = torch.zeros(max(num_steps - 1, 0), dtype=op.dtype, device=op.device)

    # Breakdown threshold. Ritz value errors scale like beta^2 / gap, so stopping once
    # beta < sqrt(eps) * ||A|| still leaves eigenvalues accurate to about eps * ||A||^2 / gap.
    # a pure eps-based test is too strict: an "exactly" rank-deficient matrix built in
    # floating point has null-space eigenvalues ~1e-16, and beta only drops to ~1e-13
    tol = torch.finfo(op.dtype).eps ** 0.5
    anorm = 0.0  # running estimate of ||A||_2 from the entries of T
    steps = num_steps
    q_prev = torch.zeros_like(q)
    b_prev = torch.zeros((), dtype=op.dtype, device=op.device)
    for j in range(num_steps):
        basis[:, j] = q
        w = op.matvec(q)
        alpha[j] = torch.dot(q, w)
        w = w - alpha[j] * q - b_prev * q_prev
        if reorth:
            qj = basis[:, : j + 1]
            for _ in range(2):
                w = w - qj @ (qj.T @ w)
        if j == num_steps - 1:
            break
        b = torch.linalg.vector_norm(w)
        anorm = max(anorm, abs(alpha[j].item()) + b.item() + b_prev.item())
        if b.item() <= tol * anorm:
            # breakdown: the Krylov space is invariant (e.g. A = I, or rank(A) < m), so T
            # already holds exact eigenvalues. dividing by b would fill Q with NaN/noise
            steps = j + 1
            break
        beta[j] = b
        q_prev, q = q, w / b
        b_prev = b

    return LanczosResult(alpha=alpha[:steps], beta=beta[: steps - 1], basis=basis[:, :steps])


def lanczos_eigsh(
    a: Tensor | LinearOperator,
    k: int = 1,
    num_steps: int | None = None,
    which: Which = "LA",
    generator: torch.Generator | None = None,
) -> EigResult:
    """``k`` extreme eigenpairs from the Ritz pairs of a Lanczos run.

    ``which`` is "LA" (largest algebraic), "SA" (smallest algebraic) or "LM" (largest
    magnitude), same names as ``scipy.sparse.linalg.eigsh``. Extreme eigenvalues converge
    first, so ``num_steps`` a few times ``k`` is usually enough when the spectrum has gaps.

    After an early breakdown fewer than ``k`` pairs may come back: a single start vector only
    sees as many distinct eigenvalues as the Krylov space it spans.
    """
    op = aslinearoperator(a)
    n = op.shape[0]
    if num_steps is None:
        num_steps = min(n, max(2 * k + 1, 30))
    if not 1 <= k <= num_steps:
        raise ValueError(f"k must be in [1, num_steps={num_steps}], got {k}")

    res = lanczos(op, num_steps, generator=generator)
    theta, s = torch.linalg.eigh(res.tridiagonal())

    if which == "LA":
        order = torch.argsort(theta, descending=True)
    elif which == "SA":
        order = torch.argsort(theta)
    elif which == "LM":
        order = torch.argsort(theta.abs(), descending=True)
    else:
        raise ValueError(f"unknown which={which!r}")
    idx = order[: min(k, res.steps)]

    vecs = res.basis @ s[:, idx]
    # residual norm of a Ritz pair is |beta_m * s_{m,i}|; ta không có beta_m nên check trực tiếp
    resid = torch.linalg.vector_norm(op.matmat(vecs) - vecs * theta[idx], dim=0)
    scale = theta[idx].abs().clamp_min(1.0)
    return EigResult(
        eigenvalues=theta[idx],
        eigenvectors=vecs,
        iterations=[res.steps] * idx.numel(),
        converged=(resid <= 1e-6 * scale).tolist(),
    )
