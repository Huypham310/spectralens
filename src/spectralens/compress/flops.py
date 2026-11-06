"""Multiply-accumulate counts from one forward pass (forward hooks on the compute layers)."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """MACs of one forward pass of a single input of shape `input_shape` (no batch dim).

    Counts Conv2d, Linear and nn.MultiheadAttention (its projections are called functionally,
    so the inner Linear hooks never fire). Norms, activations and pooling are ignored: they're
    a rounding error next to the matrix products.
    """
    total = 0

    def conv_hook(m: nn.Conv2d, _inp, out):
        nonlocal total
        kh, kw = m.kernel_size
        total += out[0].numel() * (m.in_channels // m.groups) * kh * kw

    def linear_hook(m: nn.Linear, inp, _out):
        nonlocal total
        rows = inp[0].numel() // m.in_features
        total += rows * m.in_features * m.out_features

    def mha_hook(m: nn.MultiheadAttention, inp, _out):
        nonlocal total
        q = inp[0]
        tokens = q.shape[1] if m.batch_first else q.shape[0]
        d = m.embed_dim
        # q/k/v projections + QK^T + attention-weighted V + output projection
        total += tokens * d * 3 * d + 2 * tokens * tokens * d + tokens * d * d

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(mha_hook))
        elif isinstance(mod, nn.Linear) and not _inside_mha(model, mod):
            hooks.append(mod.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    try:
        p = next(model.parameters())
        model(torch.zeros(1, *input_shape, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return int(total)


def _inside_mha(model: nn.Module, linear: nn.Linear) -> bool:
    return any(
        isinstance(m, nn.MultiheadAttention) and linear is m.out_proj for m in model.modules()
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
