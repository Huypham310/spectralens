import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectralens.linalg.lanczos import lanczos, lanczos_eigsh
from spectralens.linalg.operators import LinearOperator
from tests.linalg.helpers import sym_with_spectrum


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


@settings(max_examples=20, deadline=None)
@given(n=st.integers(2, 40), seed=st.integers(0, 10_000))
def test_full_run_reproduces_spectrum(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n, generator=g, dtype=torch.float64)
    a = (x + x.T) / 2
    res = lanczos(a, num_steps=n, generator=_gen(seed))
    got = torch.linalg.eigvalsh(res.tridiagonal())
    torch.testing.assert_close(got, torch.linalg.eigvalsh(a), rtol=1e-8, atol=1e-8)


def test_basis_is_orthonormal_and_tridiagonalizes():
    eigs = torch.logspace(0, -6, 60, dtype=torch.float64)
    a = sym_with_spectrum(eigs, seed=3)
    res = lanczos(a, num_steps=40, generator=_gen())
    q = res.basis
    torch.testing.assert_close(q.T @ q, torch.eye(40, dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(q.T @ a @ q, res.tridiagonal(), atol=1e-10, rtol=0)


@pytest.mark.parametrize("which", ["LA", "SA", "LM"])
def test_eigsh_extremes(which):
    eigs = torch.cat(
        [
            # three isolated eigenvalues on each side of a dense bulk
            torch.tensor([9.0, 7.5, 6.0, -8.0, -6.5, -5.0], dtype=torch.float64),
            torch.linspace(-1, 1, 194, dtype=torch.float64),
        ]
    )
    a = sym_with_spectrum(eigs, seed=1)
    res = lanczos_eigsh(a, k=3, num_steps=60, which=which, generator=_gen())
    sorted_eigs = {
        "LA": torch.sort(eigs, descending=True).values,
        "SA": torch.sort(eigs).values,
        "LM": eigs[torch.argsort(eigs.abs(), descending=True)],
    }[which]
    torch.testing.assert_close(res.eigenvalues, sorted_eigs[:3], rtol=1e-8, atol=1e-8)
    assert all(res.converged)


def test_matches_scipy_eigsh_on_sparse_laplacian():
    # 2-D grid Laplacian, used matrix-free through a scipy sparse matvec.
    # rectangular grid on purpose: a square one has repeated eigenvalues (i, j) / (j, i), and
    # single-vector Lanczos only ever sees one copy of a repeated eigenvalue
    def lap1d(m):
        return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))

    m1, m2 = 30, 23
    lap = (sp.kron(lap1d(m1), sp.eye(m2)) + sp.kron(sp.eye(m1), lap1d(m2))).tocsr()
    n = lap.shape[0]

    op = LinearOperator(
        shape=(n, n),
        matvec=lambda v: torch.from_numpy(lap @ v.numpy()),
        dtype=torch.float64,
    )
    ours = lanczos_eigsh(op, k=4, num_steps=150, which="LA", generator=_gen())
    ref = spla.eigsh(lap, k=4, which="LA", return_eigenvectors=False)
    np.testing.assert_allclose(ours.eigenvalues.numpy(), np.sort(ref)[::-1], rtol=1e-8)


def test_bad_arguments():
    with pytest.raises(ValueError):
        lanczos(torch.ones(3, 2), num_steps=2)
    with pytest.raises(ValueError):
        lanczos(torch.eye(3), num_steps=4)
    with pytest.raises(ValueError):
        lanczos_eigsh(torch.eye(5), k=6, num_steps=5)
