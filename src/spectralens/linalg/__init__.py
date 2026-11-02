"""From-scratch numerical linear algebra used by the analysis code.

Everything here works on dense tensors or on matrix-free operators (a callable v -> A @ v),
so the same routines can be pointed at a weight matrix or at a network's Hessian.
"""

from spectralens.linalg.hvp import hessian_operator, model_hessian_operator
from spectralens.linalg.lanczos import LanczosResult, lanczos, lanczos_eigsh
from spectralens.linalg.operators import LinearOperator, aslinearoperator
from spectralens.linalg.power import EigResult, power_iteration
from spectralens.linalg.rsvd import SVDResult, randomized_range_finder, randomized_svd
from spectralens.linalg.trace import TraceEstimate, hutchinson, hutchpp

__all__ = [
    "EigResult",
    "LanczosResult",
    "LinearOperator",
    "SVDResult",
    "TraceEstimate",
    "aslinearoperator",
    "hessian_operator",
    "hutchinson",
    "hutchpp",
    "lanczos",
    "lanczos_eigsh",
    "model_hessian_operator",
    "power_iteration",
    "randomized_range_finder",
    "randomized_svd",
]
