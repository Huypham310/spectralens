import pytest
import torch

from spectralens.linalg.slq import slq
from tests.linalg.helpers import sym_with_spectrum


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_full_krylov_space_recovers_the_exact_eigenvalues():
    eigs = torch.tensor([-2.0, 0.5, 1.0, 3.0, 7.0])
    a = sym_with_spectrum(eigs, seed=1)
    d = slq(a, num_steps=5, num_probes=3, generator=_gen())
    for row in d.nodes:
        torch.testing.assert_close(row.sort().values, eigs.double())
    torch.testing.assert_close(d.weights.sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_trace_and_moments_on_a_large_matrix():
    eigs = torch.linspace(0.0, 1.0, 300).double() ** 2
    eigs[-3:] = torch.tensor([20.0, 30.0, 50.0], dtype=torch.float64)  # a few outliers
    a = sym_with_spectrum(eigs, seed=2)
    d = slq(a, num_steps=40, num_probes=30, generator=_gen(1))
    # quadrature is exact for low-degree polynomials, so the only error is probe variance
    assert d.trace() == pytest.approx(eigs.sum().item(), rel=0.1)
    assert d.moment(2) == pytest.approx((eigs**2).mean().item(), rel=0.2)
    # the top Ritz value converges first and lands on the largest eigenvalue
    assert d.nodes.max().item() == pytest.approx(50.0, rel=1e-6)


def test_smoothed_density_integrates_to_one():
    a = sym_with_spectrum(torch.tensor([0.0, 1.0, 1.0, 4.0, 9.0]), seed=3)
    d = slq(a, num_steps=5, num_probes=2, generator=_gen())
    grid = torch.linspace(-3, 12, 3001)
    dens = d.evaluate(grid, sigma=0.3)
    assert torch.trapezoid(dens, grid).item() == pytest.approx(1.0, abs=1e-3)
    assert (dens >= 0).all()


def test_early_breakdown_pads_with_zero_weights():
    d = slq(torch.eye(6, dtype=torch.float64) * 2, num_steps=4, num_probes=2, generator=_gen())
    # identity: the Krylov space is 1-D, so one node at 2 carries all the weight
    assert d.trace() == pytest.approx(12.0)
    assert (d.weights[:, 1:] == 0).all()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        slq(torch.zeros(3, 4), num_steps=2)
    with pytest.raises(ValueError):
        slq(torch.eye(3), num_steps=2, num_probes=0)
