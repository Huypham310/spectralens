"""Swap a trained network's layers for their low-rank versions."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from spectralens.compress.decompose import (
    rank_for_energy,
    svd_conv,
    svd_linear,
    tucker2_conv,
    tucker2_ranks,
)


@dataclass
class LayerChoice:
    name: str
    shape: tuple[int, ...]
    ranks: tuple[int, ...]
    params_before: int
    params_after: int


def _mha_out_projs(model: nn.Module) -> set[int]:
    return {id(m.out_proj) for m in model.modules() if isinstance(m, nn.MultiheadAttention)}


def eligible_layers(model: nn.Module, min_dim: int = 16) -> list[tuple[str, nn.Module]]:
    """Layers worth factorizing: Linear and (non-grouped) Conv2d whose matrix view has both
    sides >= `min_dim`. The RGB stem (3 input channels) and a 10-way classifier head are
    excluded by that rule; attention's out_proj is skipped because MHA calls it functionally."""
    skip = _mha_out_projs(model)
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and id(m) not in skip:
            if min(m.in_features, m.out_features) >= min_dim:
                out.append((name, m))
        elif isinstance(m, nn.Conv2d) and m.groups == 1 and m.in_channels > 3:
            kh, kw = m.kernel_size
            if min(m.out_channels, m.in_channels * kh * kw) >= min_dim:
                out.append((name, m))
    return out


def _set_module(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def _rank(full: int, s: torch.Tensor | None, energy: float | None, ratio: float | None) -> int:
    if ratio is not None:
        return max(1, math.ceil(ratio * full))
    return rank_for_energy(s, energy)


@torch.no_grad()
def compress_model(
    model: nn.Module,
    method: str = "svd",
    energy: float | None = 0.9,
    rank_ratio: float | None = None,
    only: set[str] | None = None,
) -> tuple[nn.Module, list[LayerChoice]]:
    """Copy of `model` with eligible layers replaced by low-rank factorizations.

    The rank comes from `rank_ratio` (fraction of the full rank) if given, else from the
    `energy` kept in the singular values. A layer is only replaced when the factorization has
    fewer parameters than the original (at high energy it often wouldn't). `method` "tucker"
    applies to k x k convs; Linear layers and 1x1 convs always use SVD.
    """
    if method not in ("svd", "tucker"):
        raise ValueError(f"unknown method {method!r}")
    model = copy.deepcopy(model)
    choices = []
    for name, m in eligible_layers(model):
        if only is not None and name not in only:
            continue
        before = m.weight.numel()
        if isinstance(m, nn.Linear):
            s = None if rank_ratio else torch.linalg.svdvals(m.weight.double())
            r = _rank(min(m.weight.shape), s, energy, rank_ratio)
            after, new, ranks = r * (m.in_features + m.out_features), None, (r,)
            if after < before:
                new = svd_linear(m, r)
        else:
            out_c, in_c, kh, kw = m.weight.shape
            if method == "tucker" and kh * kw > 1:
                if rank_ratio is not None:
                    r_out = max(1, math.ceil(rank_ratio * out_c))
                    r_in = max(1, math.ceil(rank_ratio * in_c))
                else:
                    r_out, r_in = tucker2_ranks(m.weight, energy)
                ranks = (r_out, r_in)
                after = in_c * r_in + r_in * r_out * kh * kw + r_out * out_c
                new = tucker2_conv(m, r_out, r_in) if after < before else None
            else:
                mat = m.weight.reshape(out_c, -1)
                s = None if rank_ratio else torch.linalg.svdvals(mat.double())
                r = _rank(min(mat.shape), s, energy, rank_ratio)
                ranks = (r,)
                after = r * (in_c * kh * kw + out_c)
                new = svd_conv(m, r) if after < before else None
        if new is None:
            continue
        _set_module(model, name, new)
        choices.append(LayerChoice(name, tuple(m.weight.shape), ranks, before, after))
    return model, choices
