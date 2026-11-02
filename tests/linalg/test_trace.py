import math

import pytest
import torch

from spectralens.linalg.trace import hutchinson, hutchpp, probes
from tests.linalg.helpers import sym_with_spectrum


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_rademacher_is_exact_on_diagonal():
    a = torch.diag(torch.arange(1.0, 11.0, dtype=torch.float64))
    est = hutchinson(a, num_samples=3, generator=_gen())
    assert est.estimate == pytest.approx(55.0)
    assert est.stderr == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kind", ["rademacher", "gaussian"])
def test_unbiased_and_variance_matches_theory(kind):
    a = sym_with_spectrum(torch.linspace(-2, 5, 50, dtype=torch.float64), seed=4)
    est = hutchinson(a, num_samples=20_000, kind=kind, batch_size=2000, generator=_gen(1))
    tr = torch.trace(a).item()
    # within 4 standard errors of the true trace
    assert abs(est.estimate - tr) < 4 * est.stderr

    fro2 = (a**2).sum().item()
    diag2 = (torch.diagonal(a) ** 2).sum().item()
    var = 2 * (fro2 - diag2) if kind == "rademacher" else 2 * fro2
    single_sample_std = est.stderr * math.sqrt(20_000)
    assert single_sample_std == pytest.approx(math.sqrt(var), rel=0.05)


def test_hutchpp_exact_for_low_rank_psd():
    eigs = torch.cat([torch.tensor([10.0, 4.0, 1.0, 0.5]), torch.zeros(96)]).double()
    a = sym_with_spectrum(eigs, seed=6)
    assert hutchpp(a, num_matvecs=15, generator=_gen()) == pytest.approx(15.5, rel=1e-10)


def test_hutchpp_beats_hutchinson_on_decaying_spectrum():
    # PSD with fast decay: most of the trace lives in a few directions
    eigs = 1.0 / torch.arange(1, 301, dtype=torch.float64) ** 2
    a = sym_with_spectrum(eigs, seed=7)
    tr = eigs.sum().item()
    budget = 30
    err_h, err_pp = [], []
    for seed in range(30):
        err_h.append(abs(hutchinson(a, budget, generator=_gen(seed)).estimate - tr))
        err_pp.append(abs(hutchpp(a, budget, generator=_gen(seed)) - tr))
    assert sum(err_pp) < 0.25 * sum(err_h)


def test_probe_moments():
    z = probes(5, 40_000, "rademacher", torch.float64, torch.device("cpu"), _gen())
    assert set(z.unique().tolist()) == {-1.0, 1.0}
    torch.testing.assert_close(z @ z.T / 40_000, torch.eye(5).double(), atol=0.02, rtol=0)


def test_bad_arguments():
    with pytest.raises(ValueError):
        hutchinson(torch.ones(3, 2), 4)
    with pytest.raises(ValueError):
        hutchinson(torch.eye(3), 0)
    with pytest.raises(ValueError):
        hutchpp(torch.eye(3), 2)
    with pytest.raises(ValueError):
        probes(3, 2, "uniform", torch.float32, torch.device("cpu"))  # type: ignore[arg-type]
