import json
import math

import pytest
import torch

from spectralens.analysis.spectra import (
    hill_alpha,
    layer_matrices,
    mp_edge,
    network_spectra,
    spectrum_of,
    summarize,
)
from spectralens.linalg import power_iteration
from spectralens.models import build_model
from spectralens.spectra import main as spectra_main


def gaussian(m, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=torch.float64)


def test_pure_noise_stays_inside_the_marchenko_pastur_bulk():
    s = spectrum_of("noise", gaussian(200, 800))
    assert s.outlier_fraction <= 0.01
    # for q = 1/4 the MP edge is (1 + 0.5)^2 = 2.25 times sigma^2
    assert mp_edge(1.0, 200, 800) == pytest.approx(2.25)


def test_a_planted_spike_is_an_outlier_and_lowers_the_stable_rank():
    noise = gaussian(200, 800)
    u, v = gaussian(200, 1, 1), gaussian(800, 1, 2)
    spiked = noise + 3.0 * u @ v.T / math.sqrt(800)  # strong rank-1 signal
    a, b = spectrum_of("noise", noise), spectrum_of("spiked", spiked)
    assert a.outlier_fraction == 0 and b.outlier_fraction == pytest.approx(1 / 200)
    # one dominant direction pulls ||W||_2 up, so the stable rank drops (~90 -> ~54 here)
    assert b.stable_rank < 0.75 * a.stable_rank


def test_spectral_norm_matches_our_power_iteration():
    w = gaussian(64, 96, 3)
    s = spectrum_of("w", w)
    top = power_iteration(w @ w.T, k=1, max_iters=2000, tol=1e-12).eigenvalues[0]
    assert s.spectral_norm == pytest.approx(float(top.sqrt()), rel=1e-6)


def test_ranks_on_extreme_matrices():
    eye = spectrum_of("eye", torch.eye(50, dtype=torch.float64))
    assert eye.stable_rank == pytest.approx(50) and eye.effective_rank == pytest.approx(50)
    rank1 = spectrum_of("r1", gaussian(50, 1) @ gaussian(1, 70, 1))
    assert rank1.stable_rank == pytest.approx(1, abs=1e-6)
    assert rank1.effective_rank == pytest.approx(1, abs=1e-3)


def test_tall_matrices_are_transposed():
    s = spectrum_of("tall", gaussian(300, 40))
    assert s.shape == (40, 300)


def test_hill_recovers_a_power_law_exponent():
    g = torch.Generator().manual_seed(0)
    alpha = 3.0
    # Pareto samples with density ~ x^-alpha: x = U^(-1/(alpha-1))
    x = torch.rand(200_000, generator=g, dtype=torch.float64) ** (-1 / (alpha - 1))
    assert hill_alpha(x, k=2_000) == pytest.approx(alpha, rel=0.1)
    assert math.isnan(hill_alpha(torch.ones(3)))


def test_layer_matrices_picks_the_right_weights():
    cnn = layer_matrices(build_model("cnn", 10, 32).state_dict())
    conv = cnn["features.0.0.weight"]
    assert conv.shape == (32, 27)  # 3 input channels x 3 x 3
    assert all(".bias" not in k and "bn" not in k for k in cnn)
    vit = layer_matrices(build_model("vit", 10, 32).state_dict())
    assert {"blocks.0.attn.in_proj_weight.q", "blocks.0.attn.in_proj_weight.k"} <= set(vit)
    assert vit["blocks.0.attn.in_proj_weight.q"].shape == (192, 192)
    assert not any("pos" in k or "norm" in k for k in vit)


def test_summary_of_a_random_init_has_no_outliers():
    torch.manual_seed(0)
    summary = summarize(network_spectra(build_model("resnet18", 10, 32).state_dict()))
    assert summary["layers"] == 21
    assert summary["outlier_fraction"] < 0.01
    assert 0 < summary["stable_rank_ratio"] < 1


def test_cli_writes_json_and_a_figure(tmp_path):
    run = tmp_path / "ckpt" / "cnn-eurosat-s0"
    run.mkdir(parents=True)
    model = build_model("cnn", 10, 32)
    for epoch in (0, 1):
        torch.save(
            {
                "epoch": epoch,
                "model": "cnn",
                "data": "eurosat",
                "state_dict": model.state_dict(),
                "val_accuracy": 0.1,
            },
            run / f"epoch_{epoch:03d}.pt",
        )
    spectra_main(["--checkpoints", str(tmp_path / "ckpt"), "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "spectra" / "cnn-eurosat-s0.json").read_text())
    assert [e["epoch"] for e in res["epochs"]] == [0, 1]
    assert len(res["final_layers"]) == res["epochs"][0]["layers"]
    assert (tmp_path / "out" / "figures" / "spectra_eurosat.png").stat().st_size > 1_000
