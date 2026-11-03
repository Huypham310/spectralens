import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectralens.linalg.operators import LinearOperator
from spectralens.linalg.rsvd import halko_error_bound, randomized_range_finder, randomized_svd
from tests.linalg.helpers import with_singular_values


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _spec_norm(x):
    return torch.linalg.matrix_norm(x, ord=2).item()


def test_exact_low_rank_is_recovered():
    sv = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5])
    a = with_singular_values(80, 60, sv, seed=0)
    res = randomized_svd(a, rank=5, n_iter=0, generator=_gen())
    torch.testing.assert_close(res.S, sv.double(), rtol=1e-10, atol=1e-10)
    assert _spec_norm(a - res.reconstruct()) < 1e-10


def test_factors_are_orthonormal():
    a = with_singular_values(70, 50, 1.0 / torch.arange(1, 51), seed=1)
    res = randomized_svd(a, rank=8, generator=_gen())
    eye = torch.eye(8, dtype=torch.float64)
    torch.testing.assert_close(res.U.T @ res.U, eye, atol=1e-10, rtol=0)
    torch.testing.assert_close(res.Vh @ res.Vh.T, eye, atol=1e-10, rtol=0)


@settings(max_examples=20, deadline=None)
@given(
    m=st.integers(40, 120),
    n=st.integers(40, 120),
    rank=st.integers(2, 10),
    n_iter=st.integers(0, 2),
    seed=st.integers(0, 10_000),
)
def test_error_within_halko_bound(m, n, rank, n_iter, seed):
    # slowly decaying spectrum sigma_i = 1 / i, a hard case for randomized methods
    k = min(m, n)
    sv = 1.0 / torch.arange(1, k + 1, dtype=torch.float64)
    a = with_singular_values(m, n, sv, seed=seed)
    q = randomized_range_finder(a, rank + 10, n_iter=n_iter, generator=_gen(seed))
    err = _spec_norm(a - q @ (q.T @ a))
    # the bound is for the expectation; tail bounds add a bit, but in practice the error sits
    # far below it, so no extra slack is needed
    assert err <= halko_error_bound(rank, 10, n_iter, k) * sv[rank].item()
    # and it can never beat the optimal rank-(k+p) error (Eckart-Young-Mirsky);
    # k >= 40 > rank + 10 here so sigma_{k+p+1} exists
    assert err >= sv[rank + 10].item() * (1 - 1e-9)


def test_power_iterations_help_on_flat_spectrum():
    sv = 1.0 / torch.sqrt(torch.arange(1, 201, dtype=torch.float64))
    a = with_singular_values(300, 200, sv, seed=2)
    errs = []
    for q in (0, 1, 3):
        res = randomized_svd(a, rank=10, n_iter=q, generator=_gen())
        errs.append(_spec_norm(a - res.reconstruct()))
    assert errs[0] > errs[1] > errs[2]
    # with 3 iterations we are within 5% of the optimum sigma_11
    assert errs[2] <= 1.05 * sv[10].item()


def test_matrix_free_operator_matches_dense():
    a = with_singular_values(50, 40, torch.logspace(0, -4, 40), seed=3)
    op = LinearOperator(
        shape=(50, 40),
        matvec=lambda v: a @ v,
        rmatvec=lambda v: a.T @ v,
        dtype=torch.float64,
    )
    dense = randomized_svd(a, rank=6, generator=_gen(9))
    free = randomized_svd(op, rank=6, generator=_gen(9))
    torch.testing.assert_close(free.S, dense.S)


def test_bad_arguments():
    a = torch.randn(10, 6)
    with pytest.raises(ValueError):
        randomized_svd(a, rank=0)
    with pytest.raises(ValueError):
        randomized_svd(a, rank=7)
    with pytest.raises(ValueError):
        halko_error_bound(5, 1, 0, 10)
