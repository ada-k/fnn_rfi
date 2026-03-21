"""ResNet-derived baselines for RFI classification and segmentation.

Two models:

ResNetClassifier
    Small ResNet-18-inspired model for patch-level RFI classification.
    Stacks a stem conv + N downsampling residual stages + global pool + head.

ResUNet
    ResNet encoder + U-Net-style decoder for binary mask segmentation.
    Skip connections carry pre-downsampling features to decoder stages.
    Use this as a strong segmentation baseline against FNNSeg.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """Standard pre-activation residual block."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _ResStage(nn.Module):
    """Stride-2 downsampling + n residual blocks, with channel projection."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 2):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.blocks = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------

class ResNetClassifier(nn.Module):
    """Small ResNet for spectrogram patch classification.

    Parameters
    ----------
    in_ch : int
        Input channels (1 for single-pol spectrograms, 3 for RGB).
    num_classes : int
        Number of output classes.
    base_ch : int
        Number of channels after the stem convolution.
    stages : int
        Number of downsampling residual stages.
    n_blocks : int
        Residual blocks per stage.
    """

    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 2,
        base_ch: int = 16,
        stages: int = 3,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(),
        )
        ch = base_ch
        stage_list: List[nn.Module] = []
        for _ in range(stages):
            out = ch * 2
            stage_list.append(_ResStage(ch, out, n_blocks))
            ch = out
        self.stages = nn.Sequential(*stage_list)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Segmentation model
# ---------------------------------------------------------------------------

class ResUNet(nn.Module):
    """ResNet encoder + learnable U-Net decoder for binary mask segmentation.

    Architecture
    ------------
    Encoder  : stem(in_ch → base_ch) + N stages of (stride-2 _ResStage).
               Skip connections are taken from each input to _ResStage.
    Bottleneck: two residual blocks at the coarsest resolution.
    Decoder  : N stages of (ConvTranspose2d up-sample → cat skip → ResBlock + proj).
    Head     : 1×1 conv → single-channel logit mask.

    Parameters
    ----------
    in_ch : int
        Input channels.
    base_ch : int
        Channels after stem.
    stages : int
        Number of encoder/decoder stages.
    out_ch : int
        Output channels (1 for binary mask).
    n_blocks : int
        Residual blocks per encoder stage.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        stages: int = 3,
        out_ch: int = 1,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(),
        )

        # Encoder: build stages and track skip-connection widths.
        # skip_widths[i] = channels of the input to enc_stages[i]
        # (the feature map that will be used as skip connexion in the decoder).
        enc_widths: List[int] = []
        ch = base_ch
        self.enc_stages = nn.ModuleList()
        for _ in range(stages):
            enc_widths.append(ch)  # width before downsampling = skip width
            out = ch * 2
            self.enc_stages.append(_ResStage(ch, out, n_blocks))
            ch = out
        # ch = base_ch * 2^stages after the encoder

        self.bottleneck = nn.Sequential(_ResBlock(ch), _ResBlock(ch))

        # Decoder: iterate reversed skip widths
        self.dec_ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for s_ch in reversed(enc_widths):
            # Upsample from ch → s_ch, then cat skip (s_ch) → conv s_ch*2 → s_ch
            self.dec_ups.append(nn.ConvTranspose2d(ch, s_ch, kernel_size=2, stride=2))
            self.dec_convs.append(
                nn.Sequential(
                    nn.Conv2d(s_ch * 2, s_ch, 1, bias=False),
                    _ResBlock(s_ch),
                )
            )
            ch = s_ch

        self.head = nn.Conv2d(ch, out_ch, kernel_size=1)
        # Store enc_widths for use in forward
        self._enc_widths = enc_widths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Encoder: collect skip connections before each downsampling stage
        skips: List[torch.Tensor] = []
        for stage in self.enc_stages:
            skips.append(x)  # x is the skip for this stage's up-counterpart
            x = stage(x)

        x = self.bottleneck(x)

        # Decoder: reverse through skips
        for up, conv, skip in zip(self.dec_ups, self.dec_convs, reversed(skips)):
            x = up(x)
            # Pad if spatial dims differ (odd-sized inputs)
            if x.shape[2:] != skip.shape[2:]:
                x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                               0, skip.shape[2] - x.shape[2]])
            x = conv(torch.cat([x, skip], dim=1))

        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x_cls = torch.randn(2, 1, 32, 32)
    cls = ResNetClassifier(in_ch=1, num_classes=2, base_ch=16, stages=3)
    print(f"ResNetClassifier: params={count_params(cls):,}  out={cls(x_cls).shape}")

    x_seg = torch.randn(2, 1, 64, 64)
    seg = ResUNet(in_ch=1, base_ch=16, stages=3)
    print(f"ResUNet:          params={count_params(seg):,}  out={seg(x_seg).shape}")
