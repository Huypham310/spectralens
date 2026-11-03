"""Command line entry point (``spectralens`` or ``python -m spectralens``).

Experiments themselves are Hydra apps; this CLI only holds small utility commands.
"""

from __future__ import annotations

import argparse
import os
import platform

import torch

from spectralens import __version__
from spectralens.utils import pick_device


def _info() -> int:
    """Print versions and the device experiments would run on."""
    device = pick_device()
    print(f"spectralens {__version__}")
    print(f"python      {platform.python_version()}")
    print(f"torch       {torch.__version__} (cuda {torch.version.cuda or 'none'})")
    print(f"device      {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"gpu         {props.name}, {props.total_memory / 2**30:.1f} GiB")
    return 0


def _data(name: str, root: str, download: bool, seed: int) -> int:
    """Build the splits once (downloads if asked) and print sizes, class balance and stats."""
    import time

    import numpy as np

    from spectralens.data import build_datasets, make_loaders
    from spectralens.data.datasets import labels_of

    t0 = time.perf_counter()
    splits = build_datasets(name, root, seed=seed, download=download)
    print(f"{name}: built in {time.perf_counter() - t0:.1f}s")
    for split_name in ("train", "val", "test"):
        part = getattr(splits, split_name)
        counts = np.bincount(labels_of(part.base)[part.indices])
        print(f"  {split_name:5s} {len(part):6d} samples, per class {counts.min()}-{counts.max()}")
    print(f"  mean {[round(m, 4) for m in splits.mean]}")
    print(f"  std  {[round(s, 4) for s in splits.std]}")

    # one pass over the train loader to see if data loading will be a bottleneck
    loader = make_loaders(splits, batch_size=256, num_workers=0, seed=seed)["train"]
    t0 = time.perf_counter()
    n = sum(x.shape[0] for x, _ in loader)
    print(f"  train epoch (1 worker): {n / (time.perf_counter() - t0):.0f} images/s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spectralens", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="show versions and the selected device")
    data = sub.add_parser("data", help="build dataset splits and print their statistics")
    data.add_argument("name", choices=["eurosat", "cifar10"])
    data.add_argument("--root", default=os.environ.get("SPECTRALENS_DATA", "data"))
    data.add_argument("--download", action="store_true")
    data.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "info":
        return _info()
    if args.command == "data":
        return _data(args.name, args.root, args.download, args.seed)
    parser.print_help()
    return 0
