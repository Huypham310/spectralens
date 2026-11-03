"""Stochastic trace estimation for matrix-free operators.

The Hessian of a network is only available through Hessian-vector products, but its trace
(sum of eigenvalues, a common sharpness measure) can still be estimated from a handful of
them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from spectralens.linalg.operators import LinearOperator, aslinearoperator

Probe = Literal["rademacher", "gaussian"]


@dataclass
class TraceEstimate:
    """Point estimate plus the standard error of the mean over probe vectors."""

    estimate: float
    stderr: float
    num_matvecs: int


def probes(
    n: int,
    k: int,
    kind: Probe,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``k`` random probe vectors with E[z z^T] = I, as columns of an (n, k) matrix."""
    if kind == "rademacher":
        z = torch.randint(0, 2, (n, k), generator=generator).to(dtype) * 2 - 1
    elif kind == "gaussian":
        z = torch.randn(n, k, dtype=dtype, generator=generator)
    else:
        raise ValueError(f"unknown probe kind {kind!r}")
    return z.to(device)


def hutchinson(
    a: Tensor | LinearOperator,
    num_samples: int,
    kind: Probe = "rademacher",
    batch_size: int = 32,
    generator: torch.Generator | None = None,
) -> TraceEstimate:
    """Hutchinson's estimator tr(A) ~ (1/s) sum_i z_i^T A z_i.

    Unbiased for any probe with E[z z^T] = I. The single-sample variance is
    2 (||A||_F^2 - sum_i A_ii^2) for Rademacher probes and 2 ||A||_F^2 for Gaussian ones
    (symmetric A), so Rademacher is never worse and is exact on diagonal matrices.
    """
    op = aslinearoperator(a)
    n = op.shape[0]
    if op.shape[0] != op.shape[1]:
        raise ValueError(f"trace needs a square operator, got {op.shape}")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    vals: list[Tensor] = []
    done = 0
    # probes in batches: one matmat is much faster than many matvecs on GPU
    while done < num_samples:
        b = min(batch_size, num_samples - done)
        z = probes(n, b, kind, op.dtype, op.device, generator)
        vals.append((z * op.matmat(z)).sum(dim=0))
        done += b
    v = torch.cat(vals)
    std = v.std(unbiased=True).item() if v.numel() > 1 else math.nan
    return TraceEstimate(
        estimate=v.mean().item(),
        stderr=std / math.sqrt(v.numel()),
        num_matvecs=num_samples,
    )


def hutchpp(
    a: Tensor | LinearOperator,
    num_matvecs: int,
    kind: Probe = "rademacher",
    generator: torch.Generator | None = None,
) -> float:
    """Hutch++ (Meyer, Musco, Musco & Woodruff, 2021).

    Splits the budget in three: m/3 products find an approximate top eigenspace Q, m/3 compute
    tr(Q^T A Q) exactly, and m/3 run Hutchinson on the deflated rest (I - QQ^T) A (I - QQ^T).
    For PSD A this needs O(1/eps) products for relative error eps instead of Hutchinson's
    O(1/eps^2), and it is exact when rank(A) <= m/3.
    """
    op = aslinearoperator(a)
    n = op.shape[0]
    if op.shape[0] != op.shape[1]:
        raise ValueError(f"trace needs a square operator, got {op.shape}")
    k = num_matvecs // 3
    if k < 1:
        raise ValueError("Hutch++ needs at least 3 matvecs")

    s = probes(n, k, kind, op.dtype, op.device, generator)
    g = probes(n, k, kind, op.dtype, op.device, generator)
    q, _ = torch.linalg.qr(op.matmat(s), mode="reduced")
    top = torch.trace(q.T @ op.matmat(q))
    g = g - q @ (q.T @ g)  # chiếu probe ra khỏi không gian đã tính chính xác
    rest = (g * op.matmat(g)).sum() / k
    return (top + rest).item()
