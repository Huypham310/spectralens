from torch import nn
from torchvision.models import resnet18


def resnet18_small(num_classes: int, in_channels: int = 3) -> nn.Module:
    """ResNet-18 for 32-64 px images.

    The ImageNet stem (7x7 stride-2 conv + max-pool) shrinks a 32x32 CIFAR image to 8x8 before the
    first residual block, which throws away most of the detail; the usual fix is a 3x3 stride-1
    stem without the pool.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
