"""Hessian-vector products via double backprop (Pearlmutter, 1994).

For a scalar loss L(theta) with gradient g = dL/dtheta, H v = d(g^T v)/dtheta. Autograd gives
that for the cost of about two backward passes, without ever forming the P x P Hessian
(P is 11M for ResNet-18, so the dense Hessian would be ~500 TB in float32).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from spectralens.linalg.operators import LinearOperator


def flatten(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors into one 1-D vector."""
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(vec: Tensor, like: Sequence[Tensor]) -> list[Tensor]:
    """Split a flat vector back into tensors shaped like ``like``."""
    out, offset = [], 0
    for t in like:
        n = t.numel()
        out.append(vec[offset : offset + n].view_as(t))
        offset += n
    if offset != vec.numel():
        raise ValueError(f"vector has {vec.numel()} entries, parameters need {offset}")
    return out


def hessian_operator(loss: Tensor, params: Sequence[Tensor]) -> LinearOperator:
    """Hessian of ``loss`` w.r.t. ``params`` as a symmetric matrix-free operator.

    The gradient graph is built once here (``create_graph=True``) and reused by every
    product, so ``loss`` must stay alive as long as the operator is used.
    """
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError("no parameters require grad")
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    # unused params (e.g. a frozen head) get a zero gradient; we keep them so the vector
    # layout always matches the full parameter list
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params, strict=True)]
    n = sum(p.numel() for p in params)
    ref = params[0]

    def matvec(v: Tensor) -> Tensor:
        vs = unflatten_like(v.to(dtype=ref.dtype, device=ref.device), params)
        dot = sum((g * vi).sum() for g, vi in zip(grads, vs, strict=True))
        if not dot.requires_grad:
            # gradient is constant in theta (e.g. linear loss) -> Hessian is zero
            return torch.zeros(n, dtype=ref.dtype, device=ref.device)
        hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
        hv = [torch.zeros_like(p) if h is None else h for h, p in zip(hv, params, strict=True)]
        return flatten(hv).detach()

    # TODO: batch HVP bằng torch.func (vmap over jvp-of-grad) để matmat nhanh hơn vòng for
    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        rmatvec=matvec,  # H is symmetric
        dtype=ref.dtype,
        device=ref.device,
    )


def model_hessian_operator(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
) -> LinearOperator:
    """Hessian of ``loss_fn(model(inputs), targets)`` over all trainable parameters.

    Uses a single batch. The model should be in eval mode (or have no BatchNorm/Dropout),
    otherwise the "function" being differentiated changes between calls.
    """
    loss = loss_fn(model(inputs), targets)
    return hessian_operator(loss, list(model.parameters()))


def batched_hessian_operator(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    batches: Sequence[tuple[Tensor, Tensor]],
) -> LinearOperator:
    """Hessian of the mean loss over several batches, rebuilt batch by batch on every product.

    ``model_hessian_operator`` keeps one double-backward graph alive, which for a few thousand
    images through ResNet-18 does not fit in 12 GB. Here each matvec redoes forward + grad per
    batch and frees the graph right away: ~1.5x slower per product, but memory stays at one
    batch. ``loss_fn`` must use mean reduction; batches are weighted by their size so the
    result equals the Hessian of the mean over all samples.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no parameters require grad")
    if not batches:
        raise ValueError("need at least one batch")
    total = sum(int(y.shape[0]) for _, y in batches)
    n = sum(p.numel() for p in params)
    ref = params[0]

    def matvec(v: Tensor) -> Tensor:
        vs = unflatten_like(v.to(dtype=ref.dtype, device=ref.device), params)
        out = torch.zeros(n, dtype=ref.dtype, device=ref.device)
        for x, y in batches:
            loss = loss_fn(model(x), y) * (y.shape[0] / total)
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            dot = sum((g * vi).sum() for g, vi in zip(grads, vs, strict=True) if g is not None)
            if not isinstance(dot, Tensor) or not dot.requires_grad:
                continue
            hv = torch.autograd.grad(dot, params, allow_unused=True)
            out += flatten(
                [torch.zeros_like(p) if h is None else h for h, p in zip(hv, params, strict=True)]
            )
        return out

    return LinearOperator(
        shape=(n, n), matvec=matvec, rmatvec=matvec, dtype=ref.dtype, device=ref.device
    )
