import pytest
import torch

from spectralens.models import MODELS, TinyViT, build_model, count_parameters


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("size", [32, 64])
def test_output_shape(name, size):
    model = build_model(name, num_classes=10, image_size=size).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 10)


def test_parameter_counts_stay_in_the_expected_range():
    # the analysis depends on these being small / medium / large-ish
    sizes = {name: count_parameters(build_model(name, 10, 32)) for name in MODELS}
    assert 250_000 < sizes["cnn"] < 350_000
    assert 1_500_000 < sizes["vit"] < 2_000_000
    assert 11_000_000 < sizes["resnet18"] < 11_300_000


def test_resnet_keeps_spatial_detail_on_small_images():
    model = build_model("resnet18", 10, 32)
    # 3x3 stride-1 stem and no max-pool: layer1 still sees 32x32
    feats = model.maxpool(model.relu(model.bn1(model.conv1(torch.randn(1, 3, 32, 32)))))
    assert feats.shape[-2:] == (32, 32)


def test_vit_validates_patch_size():
    with pytest.raises(ValueError, match="multiple"):
        TinyViT(10, image_size=30, patch=4)


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("mlp", 10, 32)
