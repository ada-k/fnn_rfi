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


class DropPath(nn.Module):
    """Per-sample path dropping (stochastic depth) for fractal branches.

    During training, zeroes the tensor for a random fraction of samples in
    the batch; surviving samples are rescaled by 1/(1-drop_prob) to keep
    expected activations unchanged.  At eval time the full tensor passes.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep, dtype=x.dtype, device=x.device)
        )
        return x * mask / keep


class ConvBlock(nn.Module):
    """A small 2-layer convolutional block with normalization and activation.

    Used as the leaf block inside fractal recursion.
    """

    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlockSep(nn.Module):
    """Depthwise-separable convolution block: same interface as ConvBlock.

    Replaces each standard conv with (depthwise conv + pointwise 1x1 conv)
    to reduce parameter count and FLOPs, following MobileNet-style design.
    """

    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            # First depthwise-separable pair
            nn.Conv2d(in_ch, in_ch, kernel, padding=pad, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            # Second depthwise-separable pair
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class FractalBlock(nn.Module):
    """Recursive fractal block.

    Behaviour:
    - depth == 1: compute a leaf ConvBlock (standard or separable).
    - depth > 1: create two child fractal blocks (left/right), apply
      independent drop-path to each, concatenate, and merge via 1x1 conv.

    Parameters
    ----------
    in_ch : int
    out_ch : int
    depth : int
        Recursion depth (1 = leaf, >1 = recursive).
    fib_scale : float
        Multiplier on child channel width (typically 1.0).
    drop_path_rate : float
        Probability of dropping each child path during training.
    use_separable : bool
        If True, leaf blocks use depthwise-separable convolutions.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        depth=2,
        fib_scale: float = 1.0,
        drop_path_rate: float = 0.0,
        use_separable: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.out_ch = out_ch
        if depth <= 1:
            block_cls = ConvBlockSep if use_separable else ConvBlock
            self.block = block_cls(in_ch, out_ch)
        else:
            child_ch = max(8, int(out_ch * fib_scale // 2))
            self.left = FractalBlock(
                in_ch, child_ch, depth - 1, fib_scale, drop_path_rate, use_separable
            )
            self.right = FractalBlock(
                in_ch, child_ch, depth - 1, fib_scale, drop_path_rate, use_separable
            )
            self.left_drop = DropPath(drop_path_rate)
            self.right_drop = DropPath(drop_path_rate)
            self.merge = nn.Sequential(
                nn.Conv2d(child_ch * 2, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            )

    def forward(self, x):
        if self.depth <= 1:
            return self.block(x)
        l = self.left_drop(self.left(x))
        r = self.right_drop(self.right(x))
        return self.merge(torch.cat([l, r], dim=1))


class FNN(nn.Module):
    """High-level Fractal Neural Network classifier.

    Parameters
    ----------
    in_ch : int
        Input channels (e.g. 3 for RGB, 1 for spectrograms).
    num_classes : int
        Number of output classes.
    stages : int
        Number of downsampling stages.
    base_channels : int
        Base channel multiplier.
    use_fib : bool
        Derive channel widths from Fibonacci sequence if True.
    drop_path_rate : float
        Drop-path probability applied to FractalBlock branches.
    use_separable : bool
        Use depthwise-separable convolutions in leaf blocks.
    """

    def __init__(
        self,
        in_ch=3,
        num_classes=10,
        stages=3,
        base_channels=16,
        use_fib=True,
        drop_path_rate: float = 0.0,
        use_separable: bool = False,
    ):
        super().__init__()
        self.stages = stages
        if use_fib:
            fibs = fib_sequence(stages + 1)
            widths = [max(8, base_channels * f) for f in fibs[:stages]]
        else:
            widths = [base_channels * (2 ** i) for i in range(stages)]

        layers = []
        ch = in_ch
        for i, w in enumerate(widths):
            depth = max(1, 2 + i)
            layers.append(
                FractalBlock(ch, w, depth=depth, fib_scale=1.0,
                             drop_path_rate=drop_path_rate,
                             use_separable=use_separable)
            )
            layers.append(nn.MaxPool2d(2))
            ch = w

        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = FNN()
    print(m)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(y.shape)
    print("params:", count_params(m))
