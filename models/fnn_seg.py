"""Fractal/Fibonacci U-Net for 2D spectrogram RFI segmentation.

FNNSeg mirrors the standard U-Net encoder-decoder structure but replaces
plain convolutional blocks with FractalBlocks at every stage.  Channel
widths follow the Fibonacci sequence when ``use_fib=True``, giving a
principled, compact scaling law that is the main architectural novelty of
this project.

Architecture overview
---------------------
Encoder  : N stages of (FractalBlock → MaxPool2d).  Skip connections
           preserve the pre-pool feature maps.
Bottleneck: a single deeper FractalBlock at the coarsest resolution.
Decoder  : N stages of (ConvTranspose2d up-sample → concat skip → FractalBlock).
Head     : 1×1 conv → single-channel logit mask (use BCEWithLogitsLoss).

Channel widths (use_fib=True, base_ch=16, stages=3)
  Encoder: [16, 32, 48]   (base_ch × F_2, F_3, F_4)
  Bottleneck: 96          (last encoder width × 2)
  Decoder: mirrors encoder in reverse

Example
-------
>>> model = FNNSeg(in_ch=1, base_ch=16, stages=3)
>>> x = torch.randn(2, 1, 64, 64)
>>> model(x).shape
torch.Size([2, 1, 64, 64])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fnn import FractalBlock, fib_sequence


class _EncoderStage(nn.Module):
    """One encoder stage: FractalBlock then 2× MaxPool.

    Returns both the pooled output (passed to next stage) and the
    pre-pool skip connection (passed to the matching decoder stage).
    """

    def __init__(self, in_ch: int, out_ch: int, depth: int):
        super().__init__()
        self.block = FractalBlock(in_ch, out_ch, depth=depth)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.block(x)
        return self.pool(skip), skip


class _DecoderStage(nn.Module):
    """One decoder stage: 2× ConvTranspose → cat skip → FractalBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, depth: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = FractalBlock(out_ch + skip_ch, out_ch, depth=depth)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one differences in spatial size (odd inputs)
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class FNNSeg(nn.Module):
    """Fractal/Fibonacci U-Net for binary RFI mask prediction.

    Parameters
    ----------
    in_ch : int
        Number of input channels (1 for single-polarisation spectrograms).
    base_ch : int
        Base channel multiplier for the first encoder stage.
    stages : int
        Number of encoder / decoder stages (controls model depth).
    use_fib : bool
        If True, channel widths at each stage follow the Fibonacci sequence
        scaled by ``base_ch``.  If False, widths double each stage (standard
        U-Net convention).
    out_ch : int
        Number of output channels (1 for binary mask).
    max_depth : int
        Maximum FractalBlock recursion depth; caps the depth that grows with
        the stage index to prevent excessive compute.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        stages: int = 3,
        use_fib: bool = True,
        out_ch: int = 1,
        max_depth: int = 3,
    ):
        super().__init__()

        # --- channel widths -------------------------------------------------
        if use_fib:
            fibs = fib_sequence(stages + 2)   # e.g. [1,1,2,3,5] for stages=3
            widths = [max(8, base_ch * f) for f in fibs[1: stages + 1]]
        else:
            widths = [base_ch * (2 ** i) for i in range(stages)]

        btn_ch = widths[-1] * 2  # bottleneck is always 2× the last encoder width

        # --- encoder --------------------------------------------------------
        self.encoder_stages = nn.ModuleList()
        prev = in_ch
        for i, w in enumerate(widths):
            depth = min(max_depth, max(1, 2 + i))
            self.encoder_stages.append(_EncoderStage(prev, w, depth))
            prev = w

        # --- bottleneck -----------------------------------------------------
        btn_depth = min(max_depth, max(2, stages))
        self.bottleneck = FractalBlock(prev, btn_ch, depth=btn_depth)
        prev = btn_ch

        # --- decoder --------------------------------------------------------
        self.decoder_stages = nn.ModuleList()
        for i, w in enumerate(reversed(widths)):
            depth = min(max_depth, max(1, stages - i))
            self.decoder_stages.append(_DecoderStage(prev, w, w, depth))
            prev = w

        # --- output head ----------------------------------------------------
        self.head = nn.Conv2d(prev, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for stage in self.encoder_stages:
            x, skip = stage(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for stage, skip in zip(self.decoder_stages, reversed(skips)):
            x = stage(x, skip)

        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch

    for use_fib in (True, False):
        m = FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=use_fib)
        x = torch.randn(2, 1, 64, 64)
        y = m(x)
        tag = "fib" if use_fib else "std"
        print(f"FNNSeg({tag}): out={y.shape}  params={count_params(m):,}")
