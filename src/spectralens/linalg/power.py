"""Power iteration for the dominant eigenpairs of a symmetric operator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from spectralens.linalg.operators import LinearOperator, aslinearoperator, random_start


@dataclass
class EigResult:
    """Eigenvalues (descending by magnitude), eigenvectors as columns, and convergence info."""

    eigenvalues: Tensor
    eigenvectors: Tensor
    iterations: list[int]
    converged: list[bool]


def power_iteration(
    a: Tensor | LinearOperator,
    k: int = 1,
    max_iters: int = 200,
    tol: float = 1e-6,
    generator: torch.Generator | None = None,
) -> EigResult:
    """Top-``k`` eigenpairs of a symmetric operator by power iteration with deflation.

    Converges to the eigenvalues of largest *magnitude*, so for an indefinite matrix (e.g. a
    Hessian away from a minimum) the result can be negative. The rate for the i-th pair is
    |lambda_{i+1} / lambda_i| per step, which is why Lanczos is used when the gap is small.

    Deflation is done implicitly: each product is A v - sum_j lambda_j u_j (u_j^T v), so the
    operator itself is never modified.
    """
    op = aslinearoperator(a)
    n = op.shape[0]
    if op.shape[0] != op.shape[1]:
        raise ValueError(f"power iteration needs a square operator, got {op.shape}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    vals: list[Tensor] = []
    vecs: list[Tensor] = []
    iters: list[int] = []
    conv: list[bool] = []

    for _ in range(k):

        def deflated(v: Tensor) -> Tensor:
            out = op.matvec(v)
            for lam, u in zip(vals, vecs, strict=True):
                out = out - lam * u * torch.dot(u, v)
            return out

        v = random_start(n, op.dtype, op.device, generator)
        lam = torch.zeros((), dtype=op.dtype, device=op.device)
        done = False
        used = 0
        while used < max_iters:
            used += 1
            w = deflated(v)
            new_lam = torch.dot(v, w)  # Rayleigh quotient, v is unit norm
            norm = torch.linalg.vector_norm(w)
            if norm == 0:
                # v nằm trong null space của toán tử đã deflate -> trị riêng còn lại bằng 0
                lam, done = new_lam, True
                break
            v = w / norm
            # relative change, with an absolute floor so lambda ~ 0 can still converge
            if torch.abs(new_lam - lam) <= tol * max(torch.abs(new_lam).item(), 1e-12):
                lam, done = new_lam, True
                break
            lam = new_lam
        vals.append(lam)
        vecs.append(v)
        iters.append(used)
        conv.append(done)

    return EigResult(
        eigenvalues=torch.stack(vals),
        eigenvectors=torch.stack(vecs, dim=1),
        iterations=iters,
        converged=conv,
    )
