import torch

from spectralens import __version__
from spectralens.cli import main
from spectralens.utils import pick_device, seed_everything


def test_version():
    assert __version__.count(".") == 2


def test_info_runs(capsys):
    assert main(["info"]) == 0
    assert "spectralens" in capsys.readouterr().out


def test_seed_is_reproducible():
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_pick_device_explicit():
    assert pick_device("cpu").type == "cpu"
