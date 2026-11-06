"""Accuracy vs MACs of low-rank compressed networks, and which layers tolerate it.

python -m spectralens.compress.run                       # cnn, resnet18, vit on EuroSAT
python -m spectralens.compress.run --run cnn-eurosat-s0 --finetune-epochs 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from scipy.stats import spearmanr

from spectralens.analysis.spectra import spectrum_of
from spectralens.compress.apply import compress_model, eligible_layers
from spectralens.compress.flops import count_macs, count_params
from spectralens.data import SPECS, build_datasets, make_loaders
from spectralens.hessian import load_model
from spectralens.training import evaluate, make_optimizer, train_epoch, warmup_cosine
from spectralens.utils import pick_device, seed_everything

ENERGIES = (0.80, 0.90, 0.95, 0.99)


def finetune(model, loaders, device, epochs: int, lr: float) -> None:
    """Short recovery training: same SGD recipe as the original run, small lr, cosine to 0."""
    if epochs <= 0:
        return
    opt = make_optimizer(model, lr, 0.9, 5e-4)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(steps, 0))
    for _ in range(epochs):
        train_epoch(model, loaders["train"], opt, sched, device)


def sensitivity(model, loaders, device, base_acc: float, ratio: float) -> dict:
    """Compress one layer at a time to `ratio` of its rank (no fine-tuning) and record the
    accuracy drop next to that layer's stable rank and discarded energy."""
    rows = []
    for name, layer in eligible_layers(model):
        w = layer.weight.detach()
        mat = w.reshape(w.shape[0], -1)
        spec = spectrum_of(name, mat)
        s2 = torch.linalg.svdvals(mat.double()) ** 2
        r = max(1, int(round(ratio * s2.numel())))
        small, _ = compress_model(model, "svd", rank_ratio=ratio, only={name})
        acc = evaluate(small.to(device), loaders["test"], device).accuracy
        rows.append(
            {
                "layer": name,
                "shape": list(mat.shape),
                "stable_rank_ratio": spec.stable_rank / min(mat.shape),
                "discarded_energy": float(s2[r:].sum() / s2.sum()),
                "accuracy_drop": base_acc - acc,
            }
        )
    drops = [r["accuracy_drop"] for r in rows]
    return {
        "rank_ratio": ratio,
        "layers": rows,
        "spearman_drop_vs_stable_rank": float(
            spearmanr([r["stable_rank_ratio"] for r in rows], drops).statistic
        ),
        "spearman_drop_vs_discarded_energy": float(
            spearmanr([r["discarded_energy"] for r in rows], drops).statistic
        ),
    }


def run_one(run_dir: Path, args, device, data_cache: dict) -> dict:
    ckpt_path = sorted(run_dir.glob("epoch_*.pt"))[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    spec = SPECS[ckpt["data"]]
    if ckpt["data"] not in data_cache:
        splits = build_datasets(ckpt["data"], args.data, seed=0)
        data_cache[ckpt["data"]] = make_loaders(splits, 256, 0, seed=0)
    loaders = data_cache[ckpt["data"]]
    shape = (spec.in_channels, spec.image_size, spec.image_size)

    base = load_model(ckpt, device)
    base_acc = evaluate(base, loaders["test"], device).accuracy
    base_macs, base_params = count_macs(base, shape), count_params(base)
    print(
        f"{run_dir.name}: test {base_acc:.4f}, {base_macs / 1e6:.1f} MMACs, {base_params:,} params"
    )

    methods = ["svd"] if ckpt["model"] == "vit" else ["svd", "tucker"]
    points = []
    for method in methods:
        for energy in args.energies:
            start = time.perf_counter()
            seed_everything(0)
            small, choices = compress_model(base, method, energy=energy)
            small = small.to(device)
            acc0 = evaluate(small, loaders["test"], device).accuracy
            finetune(small, loaders, device, args.finetune_epochs, args.lr)
            acc1 = evaluate(small, loaders["test"], device).accuracy
            p = {
                "method": method,
                "energy": energy,
                "layers_replaced": len(choices),
                "macs_ratio": count_macs(small, shape) / base_macs,
                "params_ratio": count_params(small) / base_params,
                "accuracy": acc0,
                "accuracy_finetuned": acc1,
            }
            points.append(p)
            print(
                f"  {method:6s} energy {energy:.2f}: {p['layers_replaced']:2d} layers, "
                f"MACs {p['macs_ratio']:6.1%}, params {p['params_ratio']:6.1%}, "
                f"acc {acc0:.4f} -> {acc1:.4f} after {args.finetune_epochs} ep "
                f"({time.perf_counter() - start:.0f}s)",
                flush=True,
            )

    sens = sensitivity(base, loaders, device, base_acc, args.sensitivity_ratio)
    print(
        f"  per-layer (rank {args.sensitivity_ratio:.0%}): spearman(drop, stable rank) "
        f"{sens['spearman_drop_vs_stable_rank']:+.2f}, spearman(drop, discarded energy) "
        f"{sens['spearman_drop_vs_discarded_energy']:+.2f}"
    )
    return {
        "run": run_dir.name,
        "checkpoint": ckpt_path.name,
        "base": {"accuracy": base_acc, "macs": base_macs, "params": base_params},
        "points": points,
        "sensitivity": sens,
    }


def plot(results: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 3.6), squeeze=False)
    for ax, r in zip(axes[0], results, strict=True):
        for method, marker in (("svd", "o"), ("tucker", "s")):
            pts = [p for p in r["points"] if p["method"] == method]
            if not pts:
                continue
            xs = [100 * p["macs_ratio"] for p in pts]
            ax.plot(xs, [100 * p["accuracy"] for p in pts], marker, ls="--", alpha=0.5,
                    label=f"{method}, no fine-tune")  # fmt: skip
            ax.plot(xs, [100 * p["accuracy_finetuned"] for p in pts], marker, ls="-",
                    label=f"{method}, fine-tuned")  # fmt: skip
        ax.axhline(100 * r["base"]["accuracy"], color="k", lw=0.8, label="dense")
        ax.set_title(r["run"], fontsize=10)
        ax.set_xlabel("MACs (% of dense)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_ylim(max(0, 100 * r["base"]["accuracy"] - 30), 100)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "compress_tradeoff.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 3.4), squeeze=False)
    for ax, r in zip(axes[0], results, strict=True):
        rows = r["sensitivity"]["layers"]
        ax.scatter([x["stable_rank_ratio"] for x in rows], [100 * x["accuracy_drop"] for x in rows])
        ax.set_title(
            f"{r['run']}  (rho = {r['sensitivity']['spearman_drop_vs_stable_rank']:+.2f})",
            fontsize=9,
        )
        ax.set_xlabel("stable rank / min dim")
        ax.set_ylabel("accuracy drop (pts), one layer at 25% rank")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "compress_sensitivity.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--data", default=os.environ.get("SPECTRALENS_DATA", "data"))
    p.add_argument(
        "--run",
        action="append",
        help="run name(s); default: cnn, resnet18 and vit on EuroSAT",
    )
    p.add_argument("--energies", type=float, nargs="+", default=list(ENERGIES))
    p.add_argument("--finetune-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--sensitivity-ratio", type=float, default=0.25)
    p.add_argument("--out", default="reports")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    runs = args.run or ["cnn-eurosat-s0", "resnet18-eurosat-s0", "vit-eurosat-s0"]
    out = Path(args.out)
    (out / "compress").mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    results = []
    for name in runs:
        res = run_one(Path(args.checkpoints) / name, args, device, cache)
        results.append(res)
        (out / "compress" / f"{name}.json").write_text(json.dumps(res, indent=2))
    plot(results, out / "figures")


if __name__ == "__main__":
    main()
