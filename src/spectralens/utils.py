"""Small helpers shared by experiments: seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs).

    With ``deterministic=True`` cuDNN autotuning is turned off so reruns match bit for bit,
    at the cost of some speed.
    """
    random.seed(seed)
    # global legacy RNG on purpose: torchvision / sklearn helpers still read from it
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # cuBLAS cần biến này thì mới deterministic được trên GPU
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def pick_device(preferred: str = "auto") -> torch.device:
    """Return the device to run on. ``"auto"`` means CUDA if available, else CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
