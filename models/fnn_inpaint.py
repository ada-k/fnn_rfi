"""FNN-based joint segmentation and inpainting model.

FNNInpaint shares the fractal encoder/bottleneck with FNNSeg but attaches
two independent decoder heads:

- seg_head   : produces a binary mask logit  (1, H, W) — like FNNSeg.
- inpaint_head : produces a restored spectrogram (1, H, W).

Training uses a combined loss:

    L = bce_weight  * BCE(mask_logit, mask_target)
      + mse_weight  * MSE(restored * mask_target, clean * mask_target)

This confines the MSE loss to masked (RFI-affected) pixels where
reconstruction is meaningful.

The model requires SyntheticSpectrogramDataset(return_clean=True) so that
(contaminated, mask, clean) triples are available for training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fnn import FractalBlock, fib_sequence


# ---------------------------------------------------------------------------
# Re-use encoder / decoder primitives from fnn_seg, but import only the
# low-level FractalBlock so FNNInpaint does not couple to FNNSeg directly.
# ---------------------------------------------------------------------------

class _InpaintEncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int,
                 drop_path_rate: float = 0.0, use_separable: bool = False):
        super().__init__()
        self.block = FractalBlock(in_ch, out_ch, depth=depth,
                                  drop_path_rate=drop_path_rate,
                                  use_separable=use_separable)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.block(x)
        return self.pool(skip), skip


class _InpaintDecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, depth: int,
                 drop_path_rate: float = 0.0, use_separable: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = FractalBlock(out_ch + skip_ch, out_ch, depth=depth,
                                  drop_path_rate=drop_path_rate,
                                  use_separable=use_separable)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        return self.block(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# FNNInpaint
# ---------------------------------------------------------------------------

class FNNInpaint(nn.Module):
    """Fractal/Fibonacci U-Net with dual decoder heads for joint
    segmentation (mask prediction) and inpainting (signal restoration).

    Parameters
    ----------
    in_ch : int
        Input channels (1 for single-pol spectrograms).
    base_ch : int
        Base channel multiplier for the first encoder stage.
    stages : int
        Number of encoder/decoder stages.
    use_fib : bool
        Use Fibonacci channel widths if True, standard doubling if False.
    max_depth : int
        Maximum FractalBlock recursion depth.
    drop_path_rate : float
        Drop-path probability for FractalBlock regularisation.
    use_separable : bool
        Use depthwise-separable convolutions in leaf FractalBlocks.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        stages: int = 3,
        use_fib: bool = True,
        max_depth: int = 3,
        drop_path_rate: float = 0.0,
        use_separable: bool = False,
    ):
        super().__init__()

        # Channel widths per stage
        if use_fib:
            fibs = fib_sequence(stages + 2)
            widths = [max(8, base_ch * f) for f in fibs[1: stages + 1]]
        else:
            widths = [base_ch * (2 ** i) for i in range(stages)]

        btn_ch = widths[-1] * 2

        # Shared encoder
        self.encoder_stages = nn.ModuleList()
        prev = in_ch
        for i, w in enumerate(widths):
            depth = min(max_depth, max(1, 2 + i))
            self.encoder_stages.append(
                _InpaintEncoderStage(prev, w, depth, drop_path_rate, use_separable)
            )
            prev = w

        # Shared bottleneck
        btn_depth = min(max_depth, max(2, stages))
        self.bottleneck = FractalBlock(prev, btn_ch, depth=btn_depth,
                                       drop_path_rate=drop_path_rate,
                                       use_separable=use_separable)
        prev = btn_ch

        # Segmentation decoder
        self.seg_decoder = nn.ModuleList()
        seg_prev = prev
        for i, w in enumerate(reversed(widths)):
            depth = min(max_depth, max(1, stages - i))
            self.seg_decoder.append(
                _InpaintDecoderStage(seg_prev, w, w, depth, drop_path_rate, use_separable)
            )
            seg_prev = w
        self.seg_head = nn.Conv2d(seg_prev, 1, kernel_size=1)

        # Inpainting decoder (independent weights)
        self.inpaint_decoder = nn.ModuleList()
        inp_prev = prev
        for i, w in enumerate(reversed(widths)):
            depth = min(max_depth, max(1, stages - i))
            self.inpaint_decoder.append(
                _InpaintDecoderStage(inp_prev, w, w, depth, drop_path_rate, use_separable)
            )
            inp_prev = w
        self.inpaint_head = nn.Conv2d(inp_prev, 1, kernel_size=1)

        # Store for reference
        self.widths = widths

    def forward(self, x: torch.Tensor):
        """Return (mask_logit, restored_spectrogram) both of shape (B, 1, H, W)."""
        skips = []
        for stage in self.encoder_stages:
            x, skip = stage(x)
            skips.append(skip)

        bottleneck_out = self.bottleneck(x)

        # Segmentation head
        seg_x = bottleneck_out
        for stage, skip in zip(self.seg_decoder, reversed(skips)):
            seg_x = stage(seg_x, skip)
        mask_logit = self.seg_head(seg_x)

        # Inpainting head
        inp_x = bottleneck_out
        for stage, skip in zip(self.inpaint_decoder, reversed(skips)):
            inp_x = stage(inp_x, skip)
        restored = self.inpaint_head(inp_x)

        return mask_logit, restored


def inpaint_loss(
    mask_logit: torch.Tensor,
    restored: torch.Tensor,
    mask_target: torch.Tensor,
    clean: torch.Tensor,
    bce_weight: float = 1.0,
    mse_weight: float = 1.0,
) -> torch.Tensor:
    """Combined segmentation + masked restoration loss.

    Parameters
    ----------
    mask_logit   : (B, 1, H, W) raw logits for mask prediction.
    restored     : (B, 1, H, W) inpainted spectrogram values.
    mask_target  : (B, 1, H, W) ground-truth binary mask.
    clean        : (B, 1, H, W) clean spectrogram target.
    bce_weight   : weight on the segmentation BCE term.
    mse_weight   : weight on the masked MSE restoration term.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(mask_logit, mask_target)
    # MSE only on RFI-contaminated pixels
    diff = (restored - clean) * mask_target
    mse = (diff ** 2).sum() / (mask_target.sum() + 1e-6)
    return bce_weight * bce + mse_weight * mse


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = FNNInpaint(in_ch=1, base_ch=16, stages=3)
    x = torch.randn(2, 1, 64, 64)
    mask_logit, restored = m(x)
    print(f"FNNInpaint: params={count_params(m):,}")
    print(f"  mask_logit: {mask_logit.shape}  restored: {restored.shape}")
    clean = torch.rand_like(x)
    mask = (torch.rand_like(x) > 0.7).float()
    loss = inpaint_loss(mask_logit, restored, mask, clean)
    print(f"  loss: {loss.item():.4f}")
