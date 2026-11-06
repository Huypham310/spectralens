"""Low-rank replacements for Linear and Conv2d layers.

- Truncated SVD (Eckart-Young-Mirsky: the best rank-r approximation in Frobenius and spectral
  norm) turns a Linear W (out x in) into two Linears through an r-dim bottleneck, and a Conv2d
  with its kernel flattened to (out, in*k*k) into a k x k conv to r channels + a 1x1 conv.
- Tucker-2 (HOSVD on the output and input channel modes) turns a k x k conv into
  1x1 (in -> r_in), k x k (r_in -> r_out), 1x1 (r_out -> out), the scheme of Kim et al. (2016).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rank_for_energy(s: Tensor, energy: float) -> int:
    """Smallest r with sum(s[:r]^2) >= energy * sum(s^2), s sorted descending."""
    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0, 1]")
    e = (s.double() ** 2).cumsum(0)
    # clamp: cumsum in float can end a hair below energy * total even at r = len(s)
    r = int(torch.searchsorted(e, energy * e[-1]).item()) + 1
    return min(max(r, 1), s.numel())


def _svd(m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(m.double(), full_matrices=False)
    return u, s, vh


def svd_linear(layer: nn.Linear, rank: int) -> nn.Sequential:
    """Linear(in, out) -> Linear(in, r, no bias) -> Linear(r, out, bias)."""
    w = layer.weight.detach()
    u, s, vh = _svd(w)
    r = min(rank, s.numel())
    root = s[:r].sqrt()
    # split sqrt(S) between the two factors so neither ends up badly scaled
    first = nn.Linear(layer.in_features, r, bias=False)
    second = nn.Linear(r, layer.out_features, bias=layer.bias is not None)
    first.weight.data = (root[:, None] * vh[:r]).to(w.dtype)
    second.weight.data = (u[:, :r] * root[None, :]).to(w.dtype)
    if layer.bias is not None:
        second.bias.data = layer.bias.detach().clone()
    return nn.Sequential(first, second).to(w.device)


def _check_conv(layer: nn.Conv2d) -> None:
    if layer.groups != 1:
        raise ValueError("grouped / depthwise convolutions are not decomposed")
    if isinstance(layer.padding, str):
        raise ValueError("string padding modes are not supported")


def svd_conv(layer: nn.Conv2d, rank: int) -> nn.Sequential:
    """k x k conv (in -> out) -> k x k conv (in -> r) + 1x1 conv (r -> out)."""
    _check_conv(layer)
    w = layer.weight.detach()
    out_c, in_c, kh, kw = w.shape
    u, s, vh = _svd(w.reshape(out_c, -1))
    r = min(rank, s.numel())
    root = s[:r].sqrt()
    first = nn.Conv2d(in_c, r, (kh, kw), layer.stride, layer.padding, layer.dilation, bias=False)
    second = nn.Conv2d(r, out_c, 1, bias=layer.bias is not None)
    first.weight.data = (root[:, None] * vh[:r]).reshape(r, in_c, kh, kw).to(w.dtype)
    second.weight.data = (u[:, :r] * root[None, :]).reshape(out_c, r, 1, 1).to(w.dtype)
    if layer.bias is not None:
        second.bias.data = layer.bias.detach().clone()
    return nn.Sequential(first, second).to(w.device)


def tucker2_factors(w: Tensor, r_out: int, r_in: int) -> tuple[Tensor, Tensor, Tensor]:
    """HOSVD on modes 0 (out) and 1 (in): W ~ core x_0 U_out x_1 U_in.

    Returns U_out (out, r_out), U_in (in, r_in) and core (r_out, r_in, kh, kw).
    """
    out_c, in_c, kh, kw = w.shape
    w = w.double()
    u_out = _svd(w.reshape(out_c, -1))[0][:, :r_out]
    # mode-1 unfolding: in-channels become the rows
    u_in = _svd(w.transpose(0, 1).reshape(in_c, -1))[0][:, :r_in]
    core = torch.einsum("oikl,or,is->rskl", w, u_out, u_in)
    return u_out, u_in, core


def tucker2_conv(layer: nn.Conv2d, r_out: int, r_in: int) -> nn.Sequential:
    """k x k conv -> 1x1 (in -> r_in), k x k (r_in -> r_out), 1x1 (r_out -> out)."""
    _check_conv(layer)
    w = layer.weight.detach()
    out_c, in_c, kh, kw = w.shape
    r_out, r_in = min(r_out, out_c), min(r_in, in_c)
    u_out, u_in, core = tucker2_factors(w, r_out, r_in)
    first = nn.Conv2d(in_c, r_in, 1, bias=False)
    middle = nn.Conv2d(
        r_in, r_out, (kh, kw), layer.stride, layer.padding, layer.dilation, bias=False
    )
    last = nn.Conv2d(r_out, out_c, 1, bias=layer.bias is not None)
    first.weight.data = u_in.T.reshape(r_in, in_c, 1, 1).to(w.dtype)
    middle.weight.data = core.to(w.dtype)
    last.weight.data = u_out.reshape(out_c, r_out, 1, 1).to(w.dtype)
    if layer.bias is not None:
        last.bias.data = layer.bias.detach().clone()
    return nn.Sequential(first, middle, last).to(w.device)


def tucker2_ranks(w: Tensor, energy: float) -> tuple[int, int]:
    """Per-mode ranks keeping `energy` of each unfolding's squared singular values."""
    out_c, in_c = w.shape[:2]
    s_out = torch.linalg.svdvals(w.double().reshape(out_c, -1))
    s_in = torch.linalg.svdvals(w.double().transpose(0, 1).reshape(in_c, -1))
    return rank_for_energy(s_out, energy), rank_for_energy(s_in, energy)
