"""From-scratch numerical linear algebra used by the analysis code.

Everything here works on dense tensors or on matrix-free operators (a callable v -> A @ v),
so the same routines can be pointed at a weight matrix or at a network's Hessian.
"""
