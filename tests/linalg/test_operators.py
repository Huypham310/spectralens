import pytest
import torch

from spectralens.linalg.operators import LinearOperator, aslinearoperator, random_start


def _dense(m=5, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=torch.float64)


def test_dense_wrapper_matches_matmul():
    a = _dense()
    op = aslinearoperator(a)
    x = torch.randn(3, dtype=torch.float64)
    assert op.shape == (5, 3)
    torch.testing.assert_close(op @ x, a @ x)
    y = torch.ones(5, dtype=torch.float64)
    torch.testing.assert_close(op.T @ y, a.T @ y)


def test_matmat_fallback_uses_matvec_per_column():
    a = _dense()
    op = LinearOperator(shape=(5, 3), matvec=lambda v: a @ v, dtype=a.dtype)
    x = torch.randn(3, 4, dtype=torch.float64)
    torch.testing.assert_close(op.matmat(x), a @ x)
    torch.testing.assert_close(op.to_dense(), a)


def test_transpose_without_rmatvec_raises():
    op = LinearOperator(shape=(2, 2), matvec=lambda v: v)
    with pytest.raises(NotImplementedError):
        _ = op.T
    with pytest.raises(NotImplementedError):
        op.rmatmat(torch.ones(2, 1))


def test_rejects_non_matrix():
    with pytest.raises(ValueError):
        aslinearoperator(torch.ones(3))


def test_random_start_is_unit_and_seeded():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    v1 = random_start(10, torch.float64, torch.device("cpu"), g1)
    v2 = random_start(10, torch.float64, torch.device("cpu"), g2)
    torch.testing.assert_close(torch.linalg.vector_norm(v1), torch.tensor(1.0, dtype=v1.dtype))
    torch.testing.assert_close(v1, v2)
