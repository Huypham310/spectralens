from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Plain VGG-style baseline: 3 conv blocks, global average pool, linear head (~0.3M params).

    Small on purpose: its Hessian is cheap enough to probe with many Lanczos steps.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width),
            _block(width, 2 * width),
            _block(2 * width, 4 * width),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))
