"""Hessian curvature (lambda_max, trace, SLQ density) across training checkpoints.

python -m spectralens.hessian --run cnn-eurosat-s0-bs32 --run cnn-eurosat-s0-bs512 --name batch
python -m spectralens.hessian --epochs last          # final checkpoint of every run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from spectralens.analysis.hessian import hessian_stats, load_batches
from spectralens.data import SPECS, build_datasets, fixed_subset
from spectralens.models import build_model
from spectralens.utils import pick_device


def load_model(ckpt: dict, device: torch.device) -> torch.nn.Module:
    spec = SPECS[ckpt["data"]]
    model = build_model(ckpt["model"], spec.num_classes, spec.image_size, spec.in_channels)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def analyze_run(run_dir: Path, args, device, batches_cache: dict) -> dict:
    ckpts = sorted(run_dir.glob("epoch_*.pt"))
    if args.epochs == "last":
        ckpts = ckpts[-1:]
    epochs = []
    density = None
    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        data = ckpt["data"]
        if data not in batches_cache:
            # no augmentation: the Hessian is of the plain training loss
            splits = build_datasets(data, args.data, seed=0, augment=False)
            subset = fixed_subset(splits.train, args.samples, seed=0)
            batches_cache[data] = load_batches(subset, args.batch_size, device)
        model = load_model(ckpt, device)
        stats = hessian_stats(
            model,
            batches_cache[data],
            top_k=args.top_k,
            lanczos_steps=args.steps,
            slq_probes=args.probes,
            slq_steps=args.slq_steps,
        )
        epochs.append(
            {"epoch": ckpt["epoch"], "val_accuracy": ckpt["val_accuracy"], **stats.as_dict()}
        )
        density = stats.density
        print(
            f"{run_dir.name:24s} epoch {ckpt['epoch']:3d}  loss {stats.loss:.4f}  "
            f"lambda_max {stats.lambda_max:9.3f}  trace {stats.trace:10.1f}  ({stats.seconds:.0f}s)"
        )
        del model
    return {
        "run": run_dir.name,
        "samples": args.samples,
        "epochs": epochs,
        "final_density": density.as_dict() if density is not None else None,
    }


def plot(results: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectralens.linalg import SpectralDensity

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for r in results:
        xs = [max(e["epoch"], 0.5) for e in r["epochs"]]
        axes[0].plot(xs, [e["lambda_max"] for e in r["epochs"]], marker="o", ms=3, label=r["run"])
        axes[1].plot(xs, [e["trace"] for e in r["epochs"]], marker="o", ms=3, label=r["run"])
        d = r["final_density"]
        dens = SpectralDensity(torch.tensor(d["nodes"]), torch.tensor(d["weights"]), d["dim"])
        lo, hi = dens.nodes.min().item(), dens.nodes.max().item()
        # kernel width ~1% of the spectrum: the bulk near 0 and the outliers both stay visible
        sigma = max((hi - lo) / 100, 1e-3)
        grid = torch.linspace(lo - 5 * sigma, hi + 5 * sigma, 2000)
        axes[2].semilogy(grid, dens.evaluate(grid, sigma).clamp_min(1e-8), label=r["run"])
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epoch (0 = random init)")
        ax.grid(alpha=0.3)
    axes[0].set_title("top Hessian eigenvalue", fontsize=10)
    axes[1].set_title("Hessian trace (SLQ)", fontsize=10)
    axes[2].set_title("eigenvalue density, final checkpoint", fontsize=10)
    axes[2].set_xlabel("eigenvalue")
    axes[2].set_ylim(1e-6, None)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--data", default=os.environ.get("SPECTRALENS_DATA", "data"))
    p.add_argument("--run", action="append", help="run name(s); default: every run")
    p.add_argument("--epochs", choices=["all", "last"], default="all")
    # a conv double-backward costs ~1.2 ms per 64x64 image on a laptop 5070 Ti (cuDNN has no
    # fast kernels for it), so 512 images x ~110 products is already ~1 min per checkpoint
    p.add_argument("--samples", type=int, default=512, help="training images in the Hessian")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--steps", type=int, default=30, help="Lanczos steps for the top eigenvalues")
    p.add_argument("--probes", type=int, default=2, help="SLQ probe vectors")
    p.add_argument("--slq-steps", type=int, default=40)
    p.add_argument("--out", default="reports")
    p.add_argument("--name", default="hessian", help="figure name prefix")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    # fixed batch shapes, so autotuning the conv algorithms pays off (~10% here)
    torch.backends.cudnn.benchmark = True
    root = Path(args.checkpoints)
    runs = (
        [root / r for r in args.run]
        if args.run
        else sorted(d for d in root.iterdir() if d.is_dir())
    )
    out = Path(args.out)
    (out / "hessian").mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    results = []
    for run_dir in runs:
        res = analyze_run(run_dir, args, device, cache)
        results.append(res)
        (out / "hessian" / f"{res['run']}.json").write_text(json.dumps(res, indent=2))
    plot(results, out / "figures" / f"{args.name}.png")


if __name__ == "__main__":
    main()
