"""Benchmark the from-scratch solvers against the dense ones in torch.linalg.

    python scripts/bench_linalg.py --n 4096 --device cuda

Prints a markdown table (time is the median of --repeats runs after one warmup).
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from spectralens.linalg import lanczos_eigsh, randomized_svd
from spectralens.utils import pick_device


def timed(fn: Callable[[], object], device: torch.device, repeats: int) -> tuple[float, object]:
    """Median wall time in ms, plus the result of the last call."""

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    out = fn()  # warmup (cuSOLVER / cuBLAS handle creation, kernel caching)
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), out


def orthogonal(n: int, k: int, g: torch.Generator, device, dtype) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(n, k, generator=g, dtype=dtype).to(device))
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    n, k = args.n, args.rank
    g = torch.Generator().manual_seed(0)
    rows = []

    # --- randomized SVD vs full SVD, polynomially decaying spectrum sigma_i = 1/i ---
    sv = 1.0 / torch.arange(1, n + 1, dtype=dtype, device=device)
    u = orthogonal(n, n, g, device, dtype)
    v = orthogonal(n, n, g, device, dtype)
    a = (u * sv) @ v.T
    del u, v

    t_full, (uf, sf, vhf) = timed(
        lambda: torch.linalg.svd(a, full_matrices=False), device, args.repeats
    )
    err_opt = sv[k].item()  # Eckart-Young: best possible rank-k error
    rows.append((f"full SVD ({n}x{n})", t_full, 1.0))
    del uf, sf, vhf

    for q in (0, 2):
        gen = torch.Generator().manual_seed(1)
        t, res = timed(
            lambda q=q, gen=gen: randomized_svd(a, k, n_iter=q, generator=gen), device, args.repeats
        )
        err = torch.linalg.matrix_norm(a - res.reconstruct(), ord=2).item()
        rows.append((f"randomized SVD, rank {k}, q={q}", t, err / err_opt))

    # --- Lanczos top-10 vs dense eigvalsh on a symmetric matrix with a known spectrum ---
    eigs = torch.cat(
        [
            torch.linspace(10, 5, 10, dtype=dtype, device=device),
            torch.linspace(1, -1, n - 10, dtype=dtype, device=device),
        ]
    )
    qm = orthogonal(n, n, g, device, dtype)
    s = (qm * eigs) @ qm.T
    s = (s + s.T) / 2
    del qm

    t_eigh, w = timed(lambda: torch.linalg.eigvalsh(s), device, args.repeats)
    rows.append(
        (f"dense eigvalsh ({n}x{n})", t_eigh, (w[-10:].flip(0) - eigs[:10]).abs().max().item())
    )
    gen = torch.Generator().manual_seed(2)
    t_lz, res = timed(
        lambda: lanczos_eigsh(s, k=10, num_steps=60, generator=gen), device, args.repeats
    )
    rows.append(
        ("Lanczos top-10 (60 steps)", t_lz, (res.eigenvalues - eigs[:10]).abs().max().item())
    )

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"device: {name}, dtype: {args.dtype}\n")
    print("| method | time (ms) | error |")
    print("|---|---:|---:|")
    for label, t, e in rows:
        print(f"| {label} | {t:.1f} | {e:.3g} |")
    print("\nSVD error = ||A - A_k||_2 / sigma_{k+1} (1.0 is optimal)")
    print("eig error = max |lambda - true|")


if __name__ == "__main__":
    main()
