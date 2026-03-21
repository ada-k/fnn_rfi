import math
from typing import List

import torch
import torch.nn as nn


def fib_sequence(n: int) -> List[int]:
    """Return first n Fibonacci numbers (1,1,2,3,...).

    This is used to drive channel-width choices in some FNN variants.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


class ConvBlock(nn.Module):
    """A small 2-layer convolutional block with normalization and activation.

    Used as the leaf block inside fractal recursion.
    """

    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        # Two conv layers to give some capacity in the leaf block
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        # Forward through the tiny conv stack
        return self.net(x)


class FractalBlock(nn.Module):
    """Recursive fractal block.

    Behaviour:
    - depth == 1: compute a small ConvBlock
    - depth > 1: create two child fractal blocks (left/right) and merge

    The merge is implemented by concatenation followed by a 1x1 conv to
    project the doubled channels back to the desired output width.
    """

    def __init__(self, in_ch, out_ch, depth=2, fib_scale: float = 1.0):
        super().__init__()
        self.depth = depth
        self.out_ch = out_ch
        if depth <= 1:
            # Leaf node: simple ConvBlock
            self.block = ConvBlock(in_ch, out_ch)
        else:
            # Determine child channel width. We use fib_scale to optionally
            # modulate child widths; here we conservatively split channels.
            child_ch = max(8, int(out_ch * fib_scale // 2))
            # Recursively create left and right child fractal blocks
            self.left = FractalBlock(in_ch, child_ch, depth - 1, fib_scale)
            self.right = FractalBlock(in_ch, child_ch, depth - 1, fib_scale)
            # Merge concatenated outputs and project to out_ch
            self.merge = nn.Sequential(
                nn.Conv2d(child_ch * 2, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            )

    def forward(self, x):
        if self.depth <= 1:
            return self.block(x)
        # Compute both child paths and merge
        l = self.left(x)
        r = self.right(x)
        out = torch.cat([l, r], dim=1)
        return self.merge(out)


class FNN(nn.Module):
    """High-level Fractal Neural Network.

    Parameters
    - in_ch: input channels (e.g., 3 for RGB)
    - num_classes: final number of classes
    - stages: number of downsampling stages
    - base_channels: base multiplier for channel widths
    - use_fib: whether to derive widths from Fibonacci numbers
    """

    def __init__(self, in_ch=3, num_classes=10, stages=3, base_channels=16, use_fib=True):
        super().__init__()
        self.stages = stages
        # Determine channel widths for each stage
        if use_fib:
            fibs = fib_sequence(stages + 1)
            widths = [max(8, base_channels * f) for f in fibs[:stages]]
        else:
            widths = [base_channels * (2 ** i) for i in range(stages)]

        layers = []
        ch = in_ch
        # Build a sequence of fractal blocks followed by pooling
        for i, w in enumerate(widths):
            depth = max(1, 2 + i)  # increase recursion depth with stage index
            layers.append(FractalBlock(ch, w, depth=depth, fib_scale=1.0))
            layers.append(nn.MaxPool2d(2))
            ch = w

        self.feature = nn.Sequential(*layers)
        # Simple classifier: global pool + linear
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        # Feature extraction path
        x = self.feature(x)
        # Classification head
        x = self.classifier(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = FNN()
    print(m)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(y.shape)
    print("params:", count_params(m))
