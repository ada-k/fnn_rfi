"""Evaluation metrics for segmentation and restoration tasks.

All functions accept PyTorch tensors on any device (CPU or GPU) and return
plain Python floats so results are easy to log and compare.

Segmentation metrics
--------------------
- iou_score        : Intersection-over-Union (Jaccard index)
- dice_score       : Dice coefficient (= F1 for binary masks)
- precision_recall_f1 : precision, recall, F1

Restoration metrics
-------------------
- psnr             : Peak Signal-to-Noise Ratio (dB)
- ssim             : Structural Similarity Index (requires scikit-image;
                     falls back to a simplified tensor implementation)

Evaluation helpers
------------------
- evaluate_segmentation  : run a full segmentation eval loop over a DataLoader
- evaluate_restoration   : run a full restoration eval loop
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detach_binary(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Detach, apply sigmoid (if raw logits) and threshold to binary."""
    p = pred.detach()
    p = torch.sigmoid(p)
    return (p >= threshold).float()


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Binary Intersection-over-Union.

    Parameters
    ----------
    pred   : raw logits or probabilities, shape (B, 1, H, W) or (B, H, W).
    target : binary ground-truth mask, same shape as pred.
    """
    p = _detach_binary(pred, threshold)
    t = target.detach().float()
    intersection = (p * t).sum()
    union = (p + t).clamp(0, 1).sum()
    return float((intersection + eps) / (union + eps))


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Dice coefficient (equivalent to F1 for binary masks)."""
    p = _detach_binary(pred, threshold)
    t = target.detach().float()
    intersection = (p * t).sum()
    return float((2.0 * intersection + eps) / (p.sum() + t.sum() + eps))


def precision_recall_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    """Pixel-level precision, recall and F1 for binary masks.

    Returns
    -------
    (precision, recall, f1) as plain floats.
    """
    p = _detach_binary(pred, threshold)
    t = target.detach().float()
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    fn = ((1.0 - p) * t).sum()
    precision = float((tp + eps) / (tp + fp + eps))
    recall    = float((tp + eps) / (tp + fn + eps))
    f1        = float(2.0 * precision * recall / (precision + recall + eps))
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Restoration metrics
# ---------------------------------------------------------------------------

def psnr(
    original: torch.Tensor,
    restored: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio (dB).

    A higher value indicates a better reconstruction.  Returns ``inf`` when
    the two tensors are identical.
    """
    mse = F.mse_loss(restored.detach(), original.detach()).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim(
    original: torch.Tensor,
    restored: torch.Tensor,
) -> float:
    """Structural Similarity Index (SSIM).

    Uses scikit-image when available (recommended); falls back to a
    simplified global SSIM formula otherwise.
    """
    try:
        from skimage.metrics import structural_similarity  # type: ignore
        import numpy as np

        orig_np = original.detach().cpu().numpy()
        rest_np = restored.detach().cpu().numpy()
        scores = []
        for b in range(orig_np.shape[0]):
            for c in range(orig_np.shape[1]):
                data_range = float(orig_np[b, c].max() - orig_np[b, c].min()) or 1.0
                s = structural_similarity(
                    orig_np[b, c],
                    rest_np[b, c],
                    data_range=data_range,
                )
                scores.append(s)
        return float(sum(scores) / len(scores)) if scores else 0.0

    except ImportError:
        # Simplified global SSIM — less accurate but dependency-free
        o, r = original.detach().float(), restored.detach().float()
        mu_o, mu_r = o.mean(), r.mean()
        cov   = ((o - mu_o) * (r - mu_r)).mean()
        var_o = ((o - mu_o) ** 2).mean()
        var_r = ((r - mu_r) ** 2).mean()
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        num = (2 * mu_o * mu_r + c1) * (2 * cov + c2)
        den = (mu_o ** 2 + mu_r ** 2 + c1) * (var_o + var_r + c2)
        return float(num / den)


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    loader,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict:
    """Evaluate a segmentation model over an entire DataLoader.

    Returns a dict with averaged IoU, Dice, precision, recall and F1.

    Parameters
    ----------
    model     : a PyTorch segmentation model (returns per-pixel logits).
    loader    : DataLoader yielding (spectrogram, mask) pairs.
    device    : device string, e.g. ``"cpu"`` or ``"cuda"``.
    threshold : probability threshold for binarising predictions.
    """
    model.eval()
    ious, dices, precs, recs, f1s = [], [], [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        ious.append(iou_score(pred, y, threshold))
        dices.append(dice_score(pred, y, threshold))
        p, r, f = precision_recall_f1(pred, y, threshold)
        precs.append(p)
        recs.append(r)
        f1s.append(f)

    n = max(len(ious), 1)
    return {
        "iou":       sum(ious)   / n,
        "dice":      sum(dices)  / n,
        "precision": sum(precs)  / n,
        "recall":    sum(recs)   / n,
        "f1":        sum(f1s)    / n,
    }


@torch.no_grad()
def evaluate_restoration(
    model: torch.nn.Module,
    loader,
    device: str = "cpu",
) -> dict:
    """Evaluate a restoration model over an entire DataLoader.

    The model is expected to accept a masked spectrogram and return a
    restored spectrogram of the same shape.  Reports PSNR and SSIM.

    Parameters
    ----------
    model  : PyTorch restoration model.
    loader : DataLoader yielding (masked_spectrogram, clean_spectrogram) pairs.
    """
    model.eval()
    psnrs, ssims = [], []

    for masked, clean in loader:
        masked, clean = masked.to(device), clean.to(device)
        restored = model(masked)
        psnrs.append(psnr(clean, restored))
        ssims.append(ssim(clean, restored))

    n = max(len(psnrs), 1)
    return {
        "psnr": sum(psnrs) / n,
        "ssim": sum(ssims) / n,
    }
