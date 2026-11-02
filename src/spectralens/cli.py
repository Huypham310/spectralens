"""Command line entry point (``spectralens`` or ``python -m spectralens``).

Experiments themselves are Hydra apps; this CLI only holds small utility commands.
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spectralens", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="show versions and the selected device")

    args = parser.parse_args(argv)
    if args.command == "info":
        return _info()
    parser.print_help()
    return 0
