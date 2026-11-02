import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectralens.linalg.power import power_iteration
from tests.linalg.helpers import sym_with_spectrum


def test_top_eigenpair_of_diagonal():
    a = torch.diag(torch.tensor([5.0, 2.0, 1.0, 0.5], dtype=torch.float64))
    res = power_iteration(a, tol=1e-12, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(res.eigenvalues, torch.tensor([5.0], dtype=torch.float64))
    assert res.converged == [True]
    # eigenvector is e_1 up to sign
    torch.testing.assert_close(res.eigenvectors[:, 0].abs()[0], torch.tensor(1.0).double())


@settings(max_examples=15, deadline=None)
@given(n=st.integers(6, 40), seed=st.integers(0, 10_000))
def test_deflation_recovers_top3_with_gap(n, seed):
    # well separated top eigenvalues so 500 iterations are plenty
    tail = torch.linspace(-1.0, 1.0, n - 3, dtype=torch.float64)
    eigs = torch.cat([torch.tensor([10.0, -6.0, 4.0], dtype=torch.float64), tail])
    a = sym_with_spectrum(eigs, seed)
    res = power_iteration(
        a, k=3, max_iters=500, tol=1e-12, generator=torch.Generator().manual_seed(seed)
    )
    torch.testing.assert_close(res.eigenvalues, eigs[:3], rtol=1e-6, atol=1e-6)
    # each returned pair satisfies A v = lambda v
    for i in range(3):
        v = res.eigenvectors[:, i]
        torch.testing.assert_close(a @ v, res.eigenvalues[i] * v, rtol=1e-4, atol=1e-4)


def test_zero_matrix_does_not_divide_by_zero():
    res = power_iteration(torch.zeros(4, 4, dtype=torch.float64))
    assert res.eigenvalues.item() == 0.0
    assert res.converged == [True]


def test_bad_arguments():
    with pytest.raises(ValueError):
        power_iteration(torch.ones(3, 2))
    with pytest.raises(ValueError):
        power_iteration(torch.eye(3), k=4)
