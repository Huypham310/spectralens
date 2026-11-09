import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectralens.models import build_model
from spectralens.train import expand_sweep, load_config
from spectralens.training import (
    evaluate,
    make_optimizer,
    sam_perturb,
    train_epoch,
    warmup_cosine,
)


def test_warmup_then_cosine():
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5)
    assert f(100) == pytest.approx(0.0, abs=1e-12)
    assert all(f(s) >= f(s + 1) for s in range(10, 100))


def test_no_weight_decay_on_biases_and_norms():
    model = build_model("cnn", 10, 32)
    opt = make_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4)
    decayed, plain = opt.param_groups
    assert decayed["weight_decay"] == 5e-4 and plain["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decayed["params"])
    assert all(p.ndim == 1 for p in plain["params"])
    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total == sum(p.numel() for p in model.parameters())


def test_training_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 4, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = build_model("cnn", 4, 32)
    opt = make_optimizer(model, lr=0.05, momentum=0.9, weight_decay=0.0)
    device = torch.device("cpu")
    first = train_epoch(model, loader, opt, None, device, amp=False)
    for _ in range(40):
        last = train_epoch(model, loader, opt, None, device, amp=False)
    assert last.loss < first.loss / 5
    assert evaluate(model, loader, device, amp=False).accuracy > 0.9


def test_sam_perturbation_has_norm_rho_along_the_gradient():
    a = torch.nn.Parameter(torch.tensor([3.0, 0.0]))
    b = torch.nn.Parameter(torch.tensor([4.0]))
    a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([4.0])  # ||g|| = 5
    steps = sam_perturb([a, b], rho=0.5)
    torch.testing.assert_close(steps[0], torch.tensor([0.3, 0.0]))
    torch.testing.assert_close(steps[1], torch.tensor([0.4]))
    torch.testing.assert_close(a.detach(), torch.tensor([3.3, 0.0]))


def test_sam_uses_the_gradient_at_the_perturbed_point():
    # one SAM step through the real loop vs the same update done by hand
    torch.manual_seed(0)
    x, y = torch.randn(8, 5), torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = torch.nn.Linear(5, 3)
    w0 = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    train_epoch(model, loader, opt, None, torch.device("cpu"), amp=False, sam_rho=0.05)

    ref = torch.nn.Linear(5, 3)
    with torch.no_grad():
        for p, w in zip(ref.parameters(), w0, strict=True):
            p.copy_(w)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn(ref(x), y).backward()
    params = list(ref.parameters())
    sam_perturb(params, 0.05)
    for p in params:
        p.grad = None
    loss_fn(ref(x), y).backward()
    for trained, p, w in zip(model.parameters(), params, w0, strict=True):
        # the step starts from w0 (not the perturbed point) but uses the perturbed gradient
        torch.testing.assert_close(trained.detach(), w - 0.1 * p.grad)


def test_empty_train_loader_is_a_clear_error():
    x, y = torch.randn(10, 3, 32, 32), torch.randint(0, 4, (10,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, drop_last=True)  # 10 < 16: no batch
    model = build_model("cnn", 4, 32)
    opt = make_optimizer(model, lr=0.05, momentum=0.9, weight_decay=0.0)
    with pytest.raises(ValueError, match="no batches"):
        train_epoch(model, loader, opt, None, torch.device("cpu"), amp=False)


def test_evaluate_counts_correctly():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            # predicts class 0 for every sample
            return torch.tensor([[2.0, 0.0]]).repeat(x.shape[0], 1)

    loader = DataLoader(TensorDataset(torch.zeros(4, 1), torch.tensor([0, 0, 1, 0])), batch_size=3)
    stats = evaluate(Fixed(), loader, torch.device("cpu"), amp=False)
    assert stats.accuracy == pytest.approx(0.75) and stats.samples == 4
    # cross-entropy of logits (2, 0): -log softmax
    expected = (3 * math.log(1 + math.exp(-2)) + math.log(1 + math.exp(2))) / 4
    assert stats.loss == pytest.approx(expected, rel=1e-5)


def test_expand_sweep():
    assert expand_sweep(["model=cnn,vit", "data=eurosat", "lr=0.1"]) == [
        ["model=cnn", "data=eurosat", "lr=0.1"],
        ["model=vit", "data=eurosat", "lr=0.1"],
    ]
    # bracketed lists are a single value
    assert expand_sweep(["checkpoint_epochs=[1,2]"]) == [["checkpoint_epochs=[1,2]"]]


def test_config_composes_with_overrides():
    cfg = load_config(["model=vit", "epochs=3"])
    assert cfg.model == "vit" and cfg.epochs == 3
    assert cfg.data == "eurosat" and 30 in cfg.checkpoint_epochs
    assert "mlflow" in cfg and "paths" in cfg  # inherited from the base config
