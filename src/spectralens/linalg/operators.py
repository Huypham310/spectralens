"""Matrix-free linear operators.

The algorithms in this package only ever touch a matrix through products ``A @ x`` (and
sometimes ``A.T @ x``). Wrapping that in a small object lets the same code run on a dense
weight matrix, a sparse matrix, or a Hessian we can never afford to materialize.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

MatVec = Callable[[Tensor], Tensor]


@dataclass
class LinearOperator:
    """A real linear map R^n -> R^m given only by its action.

    ``matvec`` maps a vector of shape (n,) to (m,). ``matmat`` (optional) maps (n, k) to (m, k)
    in one call; if it is missing we fall back to one matvec per column, which is correct but
    slow. ``rmatvec`` / ``rmatmat`` are the same for the transpose and are only needed by
    algorithms that use A^T (randomized SVD).
    """

    shape: tuple[int, int]
    matvec: MatVec
    rmatvec: MatVec | None = None
    matmat_fn: MatVec | None = None
    rmatmat_fn: MatVec | None = None
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cpu")

    def matmat(self, x: Tensor) -> Tensor:
        """Apply the operator to every column of ``x`` (shape (n, k))."""
        if self.matmat_fn is not None:
            return self.matmat_fn(x)
        return torch.stack([self.matvec(col) for col in x.T], dim=1)

    def rmatmat(self, x: Tensor) -> Tensor:
        """Apply the transpose to every column of ``x`` (shape (m, k))."""
        if self.rmatmat_fn is not None:
            return self.rmatmat_fn(x)
        if self.rmatvec is None:
            raise NotImplementedError("this operator has no transpose action")
        return torch.stack([self.rmatvec(col) for col in x.T], dim=1)

    @property
    def T(self) -> LinearOperator:  # noqa: N802 - mirrors Tensor.T
        """The transpose as another operator (needs ``rmatvec``)."""
        if self.rmatvec is None:
            raise NotImplementedError("this operator has no transpose action")
        m, n = self.shape
        return LinearOperator(
            shape=(n, m),
            matvec=self.rmatvec,
            rmatvec=self.matvec,
            matmat_fn=self.rmatmat_fn,
            rmatmat_fn=self.matmat_fn,
            dtype=self.dtype,
            device=self.device,
        )

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matvec(x) if x.ndim == 1 else self.matmat(x)

    def to_dense(self) -> Tensor:
        """Materialize the matrix by applying it to the identity. Only for small n (tests)."""
        eye = torch.eye(self.shape[1], dtype=self.dtype, device=self.device)
        return self.matmat(eye)


def aslinearoperator(a: Tensor | LinearOperator) -> LinearOperator:
    """Wrap a dense 2-D tensor (or pass an existing operator through unchanged)."""
    if isinstance(a, LinearOperator):
        return a
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(a.shape)}")
    return LinearOperator(
        shape=(a.shape[0], a.shape[1]),
        matvec=lambda v: a @ v,
        rmatvec=lambda v: a.T @ v,
        matmat_fn=lambda x: a @ x,
        rmatmat_fn=lambda x: a.T @ x,
        dtype=a.dtype,
        device=a.device,
    )


def random_start(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Unit-norm Gaussian vector, the usual starting point for Krylov methods."""
    # tạo trên CPU rồi mới chuyển, để cùng seed thì CPU/GPU ra cùng vector
    v = torch.randn(n, dtype=dtype, generator=generator).to(device)
    return v / torch.linalg.vector_norm(v)
