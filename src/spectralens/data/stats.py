"""Per-channel normalization statistics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def channel_stats(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int = 512,
    num_workers: int = 0,
) -> tuple[list[float], list[float]]:
    """Mean and std of every channel over the given samples (images already as [0, 1] tensors).

    Accumulates sum and sum of squares in float64: with ~20k images of 64x64 pixels a float32
    running sum loses digits and the std drifts noticeably.
    """
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size, num_workers=num_workers
    )
    total = 0
    s = s2 = None
    for x, _ in loader:
        x = x.to(torch.float64)
        # (N, C, H, W) -> per-channel sums over N, H, W
        if s is None:
            s = torch.zeros(x.shape[1], dtype=torch.float64)
            s2 = torch.zeros_like(s)
        s += x.sum(dim=(0, 2, 3))
        s2 += (x * x).sum(dim=(0, 2, 3))
        total += x.shape[0] * x.shape[2] * x.shape[3]
    if s is None:
        raise ValueError("no samples to compute statistics on")
    mean = s / total
    # population std; clamp guards against tiny negative values from rounding
    std = (s2 / total - mean**2).clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()
