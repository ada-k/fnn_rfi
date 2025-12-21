"""Simple CNN baseline for image inputs.

Designed to be small and comparable to the FNN prototype for smoke tests
and baseline comparisons. Includes comments for clarity and reproducibility.
"""
from typing import Tuple

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small convolutional classifier.

    Architecture:
    (Conv -> BN -> SiLU) x3 -> AdaptiveAvgPool -> Linear
    """

    def __init__(self, in_ch: int = 3, num_classes: int = 10, base_channels: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SimpleCNN()
    print(m)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(y.shape)
    print("params:", count_params(m))
