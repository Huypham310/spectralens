"""Training loop pieces, kept free of Hydra / MLflow so they can be tested directly."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    samples: int


def warmup_cosine(total_steps: int, warmup_steps: int):
    """LR multiplier: linear warm-up from ~0 to 1, then cosine decay to 0."""

    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return f


def make_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float):
    """SGD + Nesterov; no weight decay on biases and norm layers (decaying a LayerNorm gain
    toward 0 only fights the normalization)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def _autocast(device: torch.device, amp: bool):
    # bf16 needs no loss scaling, unlike fp16
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    amp: bool = True,
) -> EpochStats:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = seen = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        n = y.shape[0]
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == y).sum().item()
        seen += n
    return EpochStats(total_loss / seen, correct / seen, int(seen))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = correct = seen = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x)
        total_loss += loss_fn(logits.float(), y).item()
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.shape[0]
    return EpochStats(total_loss / seen, correct / seen, int(seen))
