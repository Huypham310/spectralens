import pytest
import torch
from torch import nn
from torch.func import functional_call

from spectralens.linalg.hvp import (
    flatten,
    hessian_operator,
    model_hessian_operator,
    unflatten_like,
)
from spectralens.linalg.lanczos import lanczos_eigsh
from tests.linalg.helpers import sym_with_spectrum


def test_quadratic_hessian_is_a():
    a = sym_with_spectrum(torch.tensor([3.0, -1.0, 0.5, 2.0]), seed=0)
    b = torch.randn(4, dtype=torch.float64)
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    loss = 0.5 * x @ a @ x + b @ x
    op = hessian_operator(loss, [x])
    torch.testing.assert_close(op.to_dense(), a)


def _tiny_mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()


def test_matches_autograd_dense_hessian():
    model = _tiny_mlp()
    x = torch.randn(8, 3, dtype=torch.float64)
    y = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()

    names = [n for n, _ in model.named_parameters()]
    shapes = [p for _, p in model.named_parameters()]

    def loss_of_flat(theta):
        params = dict(zip(names, unflatten_like(theta, shapes), strict=True))
        return loss_fn(functional_call(model, params, (x,)), y)

    theta0 = flatten([p.detach() for p in model.parameters()])
    dense = torch.autograd.functional.hessian(loss_of_flat, theta0)

    op = model_hessian_operator(model, loss_fn, x, y)
    h = op.to_dense()
    torch.testing.assert_close(h, dense, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(h, h.T, rtol=1e-10, atol=1e-10)

    # and Lanczos on the operator finds the same top eigenvalue as a dense solver
    top = lanczos_eigsh(op, k=1, num_steps=op.shape[0], generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(top.eigenvalues[0], torch.linalg.eigvalsh(dense)[-1])


def test_unused_and_frozen_params():
    used = torch.randn(3, dtype=torch.float64, requires_grad=True)
    unused = torch.randn(2, dtype=torch.float64, requires_grad=True)
    frozen = torch.randn(2, dtype=torch.float64)  # requires_grad=False -> dropped
    loss = (used**3).sum() + frozen.sum()
    op = hessian_operator(loss, [used, unused, frozen])
    assert op.shape == (5, 5)
    h = op.to_dense()
    torch.testing.assert_close(h[:3, :3], torch.diag(6 * used.detach()))
    assert torch.count_nonzero(h[3:]) == 0


def test_linear_loss_has_zero_hessian():
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    op = hessian_operator((2 * x).sum(), [x])
    assert torch.count_nonzero(op.to_dense()) == 0


def test_errors():
    with pytest.raises(ValueError):
        hessian_operator(torch.tensor(1.0), [torch.zeros(2)])
    with pytest.raises(ValueError):
        unflatten_like(torch.zeros(5), [torch.zeros(2), torch.zeros(2)])
