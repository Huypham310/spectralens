"""Image classifiers studied in the experiments."""

from torch import nn

from spectralens.models.cnn import SmallCNN
from spectralens.models.resnet import resnet18_small
from spectralens.models.vit import TinyViT

MODELS = ("cnn", "resnet18", "vit")


def build_model(name: str, num_classes: int, image_size: int, in_channels: int = 3) -> nn.Module:
    if name == "cnn":
        return SmallCNN(num_classes, in_channels)
    if name == "resnet18":
        return resnet18_small(num_classes, in_channels)
    if name == "vit":
        # 8x8 patches on 64x64 EuroSAT, 4x4 on 32x32 CIFAR: 64 tokens either way
        return TinyViT(num_classes, image_size, patch=image_size // 8, in_channels=in_channels)
    raise ValueError(f"unknown model {name!r}, expected one of {MODELS}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["MODELS", "SmallCNN", "TinyViT", "build_model", "count_parameters", "resnet18_small"]
