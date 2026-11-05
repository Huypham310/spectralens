import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from spectralens.analysis.hessian import hessian_stats, load_batches
from spectralens.hessian import load_model
from spectralens.linalg import model_hessian_operator
from spectralens.models import build_model


def _setup():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3)).double()
    x = torch.randn(40, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (40,))
    return model, x, y


def test_stats_match_the_dense_hessian():
    model, x, y = _setup()
    dense = model_hessian_operator(model, nn.CrossEntropyLoss(), x, y).to_dense()
    eigs = torch.linalg.eigvalsh(dense).flip(0)
    n = dense.shape[0]  # 51 parameters

    batches = load_batches(TensorDataset(x, y), 16, torch.device("cpu"))
    stats = hessian_stats(model, batches, top_k=3, lanczos_steps=n, slq_probes=40, slq_steps=n)
    torch.testing.assert_close(torch.tensor(stats.top_eigenvalues, dtype=torch.float64), eigs[:3])
    assert stats.lambda_max == pytest.approx(eigs[0].item())
    # SLQ trace is Hutchinson with 40 probes, so only roughly equal
    assert stats.trace == pytest.approx(torch.trace(dense).item(), rel=0.25)
    assert stats.loss == pytest.approx(nn.CrossEntropyLoss()(model(x), y).item())

    d = stats.as_dict(with_density=True)
    json.dumps(d)  # must be serializable for the reports
    assert len(d["density"]["nodes"]) == 40


def test_load_model_restores_a_checkpoint_in_eval_mode():
    src = build_model("cnn", 10, 64)
    ckpt = {"model": "cnn", "data": "eurosat", "state_dict": src.state_dict()}
    model = load_model(ckpt, torch.device("cpu"))
    assert not model.training
    x = torch.randn(2, 3, 64, 64)
    torch.testing.assert_close(model(x), src.eval()(x))


def test_load_batches_keeps_order_and_sizes():
    x, y = torch.arange(10.0)[:, None], torch.arange(10)
    batches = load_batches(TensorDataset(x, y), 4, torch.device("cpu"))
    assert [b[1].tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
