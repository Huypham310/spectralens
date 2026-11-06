"""Loss-surface curvature of a trained network: top Hessian eigenvalues, trace and density.

All three come from Hessian-vector products on a fixed training subset, so nothing of size
P x P is ever formed. The model is evaluated in eval mode: with BatchNorm in train mode the
"function" would change with every forward pass (batch statistics + running-stat updates).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from spectralens.linalg import SpectralDensity, batched_hessian_operator, lanczos_eigsh, slq


@dataclass
class HessianStats:
    loss: float
    top_eigenvalues: list[float]
    trace: float
    density: SpectralDensity
    seconds: float

    @property
    def lambda_max(self) -> float:
        return self.top_eigenvalues[0]

    def as_dict(self, with_density: bool = False) -> dict:
        out = {
            "loss": self.loss,
            "lambda_max": self.lambda_max,
            "top_eigenvalues": self.top_eigenvalues,
            "trace": self.trace,
            "seconds": round(self.seconds, 2),
        }
        if with_density:
            out["density"] = self.density.as_dict()
        return out


def load_batches(ds: Dataset, batch_size: int, device: torch.device) -> list[tuple[Tensor, Tensor]]:
    """Whole (small) dataset as a list of batches already on ``device``."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return [(x.to(device), torch.as_tensor(y).to(device)) for x, y in loader]


@torch.no_grad()
def mean_loss(model: nn.Module, batches: Iterable[tuple[Tensor, Tensor]], loss_fn) -> float:
    total = count = 0.0
    for x, y in batches:
        total += loss_fn(model(x), y).item() * y.shape[0]
        count += y.shape[0]
    return total / count


def hessian_stats(
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    top_k: int = 5,
    lanczos_steps: int = 40,
    slq_probes: int = 3,
    slq_steps: int = 60,
    seed: int = 0,
) -> HessianStats:
    """Top-``top_k`` eigenvalues (Lanczos), trace and density (SLQ) of the training-loss
    Hessian on ``batches``.

    Cost is about ``lanczos_steps + slq_probes * slq_steps`` Hessian-vector products, each
    one forward + double backward over every batch.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter()
    op = batched_hessian_operator(model, loss_fn, batches)
    gen = torch.Generator().manual_seed(seed)
    top = lanczos_eigsh(op, k=top_k, num_steps=lanczos_steps, which="LA", generator=gen)
    density = slq(op, num_steps=slq_steps, num_probes=slq_probes, generator=gen)
    return HessianStats(
        loss=mean_loss(model, batches, loss_fn),
        top_eigenvalues=[float(v) for v in top.eigenvalues],
        trace=density.trace(),
        density=density,
        seconds=time.perf_counter() - start,
    )
