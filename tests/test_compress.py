import pytest
import torch
from torch import nn

from spectralens.compress import (
    compress_model,
    count_macs,
    count_params,
    eligible_layers,
    rank_for_energy,
    svd_conv,
    svd_linear,
    tucker2_conv,
    tucker2_factors,
)
from spectralens.models import build_model


def test_rank_for_energy():
    s = torch.tensor([3.0, 2.0, 1.0, 0.0])  # energies 9, 4, 1 of 14
    assert rank_for_energy(s, 0.5) == 1
    assert rank_for_energy(s, 13 / 14) == 2
    assert rank_for_energy(s, 0.99) == 3
    assert rank_for_energy(s, 1.0) == 3  # the zero singular value is never needed
    with pytest.raises(ValueError):
        rank_for_energy(s, 0.0)


def test_full_rank_svd_layers_are_exact():
    torch.manual_seed(0)
    lin = nn.Linear(12, 7)
    x = torch.randn(5, 12)
    torch.testing.assert_close(svd_linear(lin, 7)(x), lin(x), rtol=1e-5, atol=1e-5)

    conv = nn.Conv2d(6, 10, 3, stride=2, padding=1)
    img = torch.randn(2, 6, 9, 9)
    torch.testing.assert_close(svd_conv(conv, 10)(img), conv(img), rtol=1e-4, atol=1e-5)


def test_truncated_svd_error_matches_eckart_young():
    torch.manual_seed(1)
    lin = nn.Linear(20, 16, bias=False)
    s = torch.linalg.svdvals(lin.weight.double())
    approx = svd_linear(lin, 4)
    w_hat = approx[1].weight @ approx[0].weight
    err = torch.linalg.matrix_norm(lin.weight.double() - w_hat.double(), ord=2)
    # best rank-4 approximation: spectral error is exactly the 5th singular value
    assert err.item() == pytest.approx(s[4].item(), rel=1e-5)


def test_tucker2_full_ranks_are_exact_and_factors_orthonormal():
    torch.manual_seed(2)
    conv = nn.Conv2d(8, 12, 3, padding=1)
    img = torch.randn(2, 8, 7, 7)
    torch.testing.assert_close(tucker2_conv(conv, 12, 8)(img), conv(img), rtol=1e-4, atol=1e-5)

    u_out, u_in, core = tucker2_factors(conv.weight.detach(), 5, 4)
    assert core.shape == (5, 4, 3, 3)
    torch.testing.assert_close(u_out.T @ u_out, torch.eye(5, dtype=torch.float64))
    torch.testing.assert_close(u_in.T @ u_in, torch.eye(4, dtype=torch.float64))


def test_tucker2_on_a_low_multilinear_rank_kernel_is_exact():
    torch.manual_seed(3)
    core = torch.randn(3, 2, 3, 3, dtype=torch.float64)
    a = torch.linalg.qr(torch.randn(16, 3, dtype=torch.float64))[0]
    b = torch.linalg.qr(torch.randn(10, 2, dtype=torch.float64))[0]
    w = torch.einsum("rskl,or,is->oikl", core, a, b)  # multilinear rank (3, 2)
    conv = nn.Conv2d(10, 16, 3, padding=1, bias=False).double()
    conv.weight.data = w
    img = torch.randn(1, 10, 6, 6, dtype=torch.float64)
    torch.testing.assert_close(tucker2_conv(conv, 3, 2)(img), conv(img))


def test_grouped_convs_are_rejected():
    with pytest.raises(ValueError):
        svd_conv(nn.Conv2d(8, 8, 3, groups=8), 2)


def test_macs_of_known_layers():
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(8 * 4 * 4, 5))
    # conv: 8 out channels x 16 pixels x 3*3*3 ; linear: 128 x 5
    assert count_macs(model, (3, 4, 4)) == 8 * 16 * 27 + 128 * 5

    vit = build_model("vit", 10, 32)
    macs = count_macs(vit, (3, 32, 32))
    assert macs > 50_000_000  # attention is counted, not just the MLPs
    assert vit.training  # mode restored


def test_compress_cnn_cuts_macs_and_keeps_shapes():
    torch.manual_seed(0)
    model = build_model("cnn", 10, 64).eval()
    names = [n for n, _ in eligible_layers(model)]
    assert "features.0.0" not in names  # RGB stem stays dense
    assert "head" not in names  # 10-way head is too thin to factorize

    x = torch.randn(2, 3, 64, 64)
    for method in ("svd", "tucker"):
        small, choices = compress_model(model, method, rank_ratio=0.25)
        assert choices and all(c.params_after < c.params_before for c in choices)
        assert small(x).shape == (2, 10)
        assert count_macs(small, (3, 64, 64)) < 0.6 * count_macs(model, (3, 64, 64))
        assert count_params(small) < count_params(model)
    # the original is untouched
    assert isinstance(model.features[1][0], nn.Conv2d)


def test_energy_one_keeps_the_function():
    torch.manual_seed(0)
    model = build_model("cnn", 10, 32).eval()
    # at energy 1.0 no factorization is smaller than the dense layer, so nothing is replaced
    same, choices = compress_model(model, "svd", energy=1.0)
    assert choices == []
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(same(x), model(x))


def test_only_compresses_the_named_layer():
    model = build_model("cnn", 10, 32)
    target = eligible_layers(model)[2][0]
    _, choices = compress_model(model, "svd", rank_ratio=0.25, only={target})
    assert [c.name for c in choices] == [target]
    with pytest.raises(ValueError):
        compress_model(model, "cp")
