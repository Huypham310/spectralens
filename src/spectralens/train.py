"""Train a model and save checkpoints for the spectral analysis.

python -m spectralens.train model=resnet18 data=eurosat
python -m spectralens.train model=cnn,resnet18,vit data=eurosat,cifar10   # sweep: all six runs
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from spectralens.data import build_datasets, make_loaders
from spectralens.models import build_model, count_parameters
from spectralens.training import evaluate, make_optimizer, train_epoch, warmup_cosine
from spectralens.utils import pick_device, seed_everything

log = logging.getLogger(__name__)


def save_checkpoint(path: Path, model, epoch: int, cfg: DictConfig, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": cfg.model,
            "data": cfg.data,
            "state_dict": model.state_dict(),
            "val_accuracy": val_acc,
        },
        path,
    )


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(overrides: list[str]) -> DictConfig:
    # Compose API instead of @hydra.main: hydra 1.3's CLI passes a non-string `help` to argparse,
    # which Python 3.14 rejects ("badly formed help string")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def expand_sweep(args: list[str]) -> list[list[str]]:
    """``model=cnn,vit data=eurosat`` -> [[model=cnn, data=eurosat], [model=vit, data=eurosat]].

    Only plain ``key=a,b`` values are expanded; lists in brackets are left alone.
    """
    options = []
    for arg in args:
        key, sep, value = arg.partition("=")
        if sep and "," in value and not value.startswith("["):
            options.append([f"{key}={v}" for v in value.split(",")])
        else:
            options.append([arg])
    return [list(combo) for combo in itertools.product(*options)]


def run(cfg: DictConfig) -> float:
    seed_everything(cfg.seed, cfg.deterministic)
    device = pick_device(cfg.device)

    splits = build_datasets(cfg.data, cfg.paths.data, seed=cfg.seed)
    loaders = make_loaders(splits, cfg.batch_size, cfg.num_workers, cfg.seed)
    model = build_model(cfg.model, splits.spec.num_classes, splits.spec.image_size).to(device)
    model = model.to(memory_format=torch.channels_last)

    steps = cfg.epochs * len(loaders["train"])
    optimizer = make_optimizer(model, cfg.lr, cfg.momentum, cfg.weight_decay)
    warmup = int(cfg.warmup_epochs * len(loaders["train"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(steps, warmup))

    run_name = f"{cfg.model}-{cfg.data}-s{cfg.seed}" + (f"-{cfg.tag}" if cfg.tag else "")
    ckpt_dir = Path(cfg.paths.checkpoints) / run_name
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            OmegaConf.to_container(cfg, resolve=True) | {"params": count_parameters(model)}
        )
        log.info("%s: %d params, %d steps", run_name, count_parameters(model), steps)

        # epoch 0 = the random initialization, the baseline for the weight spectra
        val = evaluate(model, loaders["val"], device, cfg.amp)
        save_checkpoint(ckpt_dir / "epoch_000.pt", model, 0, cfg, val.accuracy)
        start = time.perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            tr = train_epoch(
                model, loaders["train"], optimizer, scheduler, device, cfg.amp, cfg.sam_rho
            )
            val = evaluate(model, loaders["val"], device, cfg.amp)
            mlflow.log_metrics(
                {
                    "train_loss": tr.loss,
                    "train_acc": tr.accuracy,
                    "val_loss": val.loss,
                    "val_acc": val.accuracy,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            log.info(
                "epoch %3d  train %.4f / %.3f  val %.4f / %.3f",
                epoch,
                tr.loss,
                tr.accuracy,
                val.loss,
                val.accuracy,
            )
            if epoch in cfg.checkpoint_epochs or epoch == cfg.epochs:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, epoch, cfg, val.accuracy)

        test = evaluate(model, loaders["test"], device, cfg.amp)
        minutes = (time.perf_counter() - start) / 60
        mlflow.log_metrics(
            {"test_acc": test.accuracy, "test_loss": test.loss, "train_minutes": minutes}
        )
        summary = {
            "run": run_name,
            "params": count_parameters(model),
            "val_acc": val.accuracy,
            "test_acc": test.accuracy,
            "minutes": round(minutes, 2),
        }
        (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info("done: %s", summary)
    return test.accuracy


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for overrides in expand_sweep(sys.argv[1:] if argv is None else argv):
        run(load_config(overrides))


if __name__ == "__main__":
    main()
