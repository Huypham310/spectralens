"""End to end on CPU with a fake EuroSAT: train -> spectra / hessian / compression CLIs.

Every experiment script takes real data paths; here `build_datasets` is swapped for one that
builds the same splits from torchvision's FakeData, so the whole chain runs in seconds and the
scripts' wiring (arguments, checkpoints, report files) is actually exercised.
"""

import json

import pytest
from torchvision.datasets import FakeData

import spectralens.compress.run as compress_run
import spectralens.data
import spectralens.hessian as hessian_cli
import spectralens.spectra as spectra_cli
import spectralens.train as train_cli
from spectralens import cli
from spectralens.data import build_datasets as real_build


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="module", autouse=True)
def fake_data(workdir):
    # module scope on purpose: the module-scoped `trained` fixture is set up before any
    # function-scoped one, so a function-scoped patch silently trained on the real EuroSAT
    # (132 s on CPU here, and a crash on a machine without the data)
    def build(name, root=None, seed=0, download=False, augment=True, **_):
        pool = FakeData(size=160, image_size=(3, 64, 64), num_classes=10, random_offset=seed)
        return real_build(name, workdir / "data", seed=seed, augment=augment, pool=pool)

    with pytest.MonkeyPatch.context() as mp:
        for module in (train_cli, hessian_cli, compress_run, spectralens.data):
            mp.setattr(module, "build_datasets", build)
        mp.setenv("CUDA_VISIBLE_DEVICES", "")  # same behaviour on a GPU machine and in CI
        yield


@pytest.fixture(scope="module")
def trained(workdir):
    """One tiny CNN run with checkpoints at epochs 0, 1 and 2."""
    ckpts = workdir / "checkpoints"
    train_cli.main(
        [
            "model=cnn",
            "data=eurosat",
            "epochs=2",
            "batch_size=32",
            "checkpoint_epochs=[1]",
            "device=cpu",
            "amp=false",
            "tag=e2e",
            f"paths.checkpoints={ckpts.as_posix()}",
            f"mlflow.tracking_uri=sqlite:///{(workdir / 'mlflow.db').as_posix()}",
        ]
    )
    return ckpts, "cnn-eurosat-s0-e2e"


def test_training_writes_checkpoints_and_a_summary(trained):
    ckpts, run = trained
    names = sorted(p.name for p in (ckpts / run).glob("epoch_*.pt"))
    assert names == ["epoch_000.pt", "epoch_001.pt", "epoch_002.pt"]
    summary = json.loads((ckpts / run / "summary.json").read_text())
    assert summary["run"] == run and 0.0 <= summary["test_acc"] <= 1.0


def test_spectra_cli_on_the_trained_run(trained, workdir):
    ckpts, run = trained
    out = workdir / "reports"
    spectra_cli.main(["--checkpoints", str(ckpts), "--run", run, "--out", str(out)])
    report = json.loads((out / "spectra" / f"{run}.json").read_text())
    assert [e["epoch"] for e in report["epochs"]] == [0, 1, 2]
    assert (out / "figures" / "spectra_eurosat.png").stat().st_size > 0


def test_hessian_cli_on_the_trained_run(trained, workdir):
    ckpts, run = trained
    out = workdir / "reports"
    args = ["--checkpoints", str(ckpts), "--run", run, "--out", str(out), "--device", "cpu",
            "--samples", "20", "--batch-size", "20", "--top-k", "2", "--steps", "4",
            "--probes", "1", "--slq-steps", "4", "--name", "hessian_e2e"]  # fmt: skip
    hessian_cli.main(args)
    report = json.loads((out / "hessian" / f"{run}.json").read_text())
    assert len(report["epochs"]) == 3
    assert all(e["lambda_max"] >= e["top_eigenvalues"][-1] for e in report["epochs"])
    assert len(report["final_density"]["nodes"]) == 1  # one SLQ probe
    assert (out / "figures" / "hessian_e2e.png").stat().st_size > 0

    hessian_cli.main([*args, "--epochs", "last"])  # only the final checkpoint
    report = json.loads((out / "hessian" / f"{run}.json").read_text())
    assert [e["epoch"] for e in report["epochs"]] == [2]


def test_compression_cli_on_the_trained_run(trained, workdir):
    ckpts, run = trained
    out = workdir / "reports"
    compress_run.main(["--checkpoints", str(ckpts), "--run", run, "--out", str(out),
                       "--device", "cpu", "--energies", "0.8", "--finetune-epochs", "1",
                       "--batch-size", "32"])  # fmt: skip
    report = json.loads((out / "compress" / f"{run}.json").read_text())
    methods = {p["method"] for p in report["points"]}
    assert methods == {"svd", "tucker"}
    assert all(p["macs_ratio"] <= 1.0 for p in report["points"])
    sens = report["sensitivity"]
    assert len(sens["layers"]) == 5  # the small CNN's five factorizable convs
    assert {"stable_rank_ratio", "accuracy_drop"} <= set(sens["layers"][0])
    for fig in ("compress_tradeoff.png", "compress_sensitivity.png"):
        assert (out / "figures" / fig).stat().st_size > 0


def test_cli_info_and_data(capsys, workdir):
    assert cli.main(["info"]) == 0
    assert cli.main(["data", "eurosat", "--root", str(workdir / "data")]) == 0
    out = capsys.readouterr().out
    assert "train" in out and "per class" in out and "images/s" in out
    assert cli.main([]) == 0  # no command: prints help


def test_expand_sweep_runs_every_combination(monkeypatch):
    seen = []
    monkeypatch.setattr(train_cli, "run", lambda cfg: seen.append((cfg.model, cfg.data)) or 0.0)
    train_cli.main(["model=cnn,vit", "data=eurosat,cifar10", "epochs=1"])
    assert seen == [("cnn", "eurosat"), ("cnn", "cifar10"), ("vit", "eurosat"), ("vit", "cifar10")]


def test_main_module_runs(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr("sys.argv", ["spectralens", "--version"])
    with pytest.raises(SystemExit):
        runpy.run_module("spectralens", run_name="__main__")
    assert capsys.readouterr().out.strip()
