"""Weight spectra across the training checkpoints of every run.

python -m spectralens.spectra                # all runs in checkpoints/
python -m spectralens.spectra --run resnet18-eurosat-s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spectralens.analysis.spectra import network_spectra, summarize


def checkpoints_of(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("epoch_*.pt"))


def analyze_run(run_dir: Path) -> dict:
    epochs = []
    final_layers = []
    for ckpt in checkpoints_of(run_dir):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        spectra = network_spectra(state["state_dict"])
        epochs.append(
            {"epoch": state["epoch"], "val_accuracy": state["val_accuracy"], **summarize(spectra)}
        )
        final_layers = [s.as_dict() for s in spectra]
    return {"run": run_dir.name, "epochs": epochs, "final_layers": final_layers}


def plot(results: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    metrics = [
        ("stable_rank_ratio", "stable rank / min dim"),
        ("outlier_fraction", "eigenvalues beyond MP edge"),
        ("alpha_median", "tail exponent alpha (median)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4))
    for r in results:
        xs = [max(e["epoch"], 0.5) for e in r["epochs"]]  # epoch 0 on a log axis
        for ax, (key, _) in zip(axes, metrics, strict=True):
            ax.plot(xs, [e[key] for e in r["epochs"]], marker="o", ms=3, label=r["run"])
    for ax, (_, title) in zip(axes, metrics, strict=True):
        ax.set_xscale("log")
        ax.set_xlabel("epoch (0 = random init)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--run", action="append", help="run name(s); default: every run")
    p.add_argument("--out", default="reports")
    args = p.parse_args(argv)

    root = Path(args.checkpoints)
    runs = (
        [root / r for r in args.run]
        if args.run
        else sorted(d for d in root.iterdir() if d.is_dir())
    )
    out = Path(args.out)
    results = []
    for run_dir in runs:
        res = analyze_run(run_dir)
        results.append(res)
        (out / "spectra").mkdir(parents=True, exist_ok=True)
        (out / "spectra" / f"{res['run']}.json").write_text(json.dumps(res, indent=2))
        first, last = res["epochs"][0], res["epochs"][-1]
        print(
            f"{res['run']:22s} layers {last['layers']:3d}  "
            f"stable-rank ratio {first['stable_rank_ratio']:.3f} -> "
            f"{last['stable_rank_ratio']:.3f}  "
            f"outliers {first['outlier_fraction']:.3f} -> {last['outlier_fraction']:.3f}  "
            f"alpha {first['alpha_median']:.2f} -> {last['alpha_median']:.2f}"
        )

    for dataset in sorted({r["run"].split("-")[1] for r in results}):
        plot(
            [r for r in results if r["run"].split("-")[1] == dataset],
            out / "figures" / f"spectra_{dataset}.png",
        )


if __name__ == "__main__":
    main()
