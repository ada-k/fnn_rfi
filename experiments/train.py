"""Unified training script for quick experiments.

Usage examples (run from repo root):

    python -m experiments.train --model fnn     --dataset cifar     --epochs 3
    python -m experiments.train --model unet    --dataset synthetic --epochs 5
    python -m experiments.train --model fnn_seg --dataset synthetic --epochs 5

Results (loss curve + final metrics) are written to ``runs/<run_name>/``.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .evaluate import evaluate_segmentation
from .utils import get_dataloader, get_device, get_model, save_checkpoint, save_config, set_seed

# Models that output per-pixel logit masks (segmentation + baseline segmenters)
_SEG_MODELS = {"unet", "fnn_seg", "resunet"}


def train_epoch(model, loader, optimiser, criterion, device):
    """One training epoch; returns average loss."""
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        out = model(xb)
        loss = criterion(out, yb) if out.shape == yb.shape else criterion(out, yb.long())
        loss.backward()
        optimiser.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, is_seg: bool):
    """One validation epoch; returns a metrics dict."""
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb) if out.shape == yb.shape else criterion(out, yb.long())
        total += float(loss.item())
        n += 1

    metrics = {"val_loss": total / max(n, 1)}

    if is_seg:
        seg = evaluate_segmentation(model, loader, device=str(device))
        metrics.update({f"val_{k}": v for k, v in seg.items()})

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR or synthetic spectrograms.")
    parser.add_argument("--model",      required=True,
                        help="Model: fnn | cnn | rnn | transformer | unet | fnn_seg | "
                             "resnet_cls | resunet")
    parser.add_argument("--dataset",    required=True, help="Dataset: cifar | synthetic")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--out",        default="runs")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--subset",     type=int,   default=None,
                        help="Limit training set to this many samples.")
    parser.add_argument("--val-size",   type=int,   default=256,
                        help="Validation set size for synthetic dataset.")
    parser.add_argument("--drop-path",  type=float, default=0.0,
                        help="Drop-path rate for FractalBlock branches (0.0 = disabled).")
    parser.add_argument("--separable",  action="store_true", default=False,
                        help="Use depthwise-separable convolutions in FractalBlocks.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    is_seg = args.model.lower() in _SEG_MODELS

    # ---- data ----------------------------------------------------------------
    train_loader, num_classes = get_dataloader(
        args.dataset, batch_size=args.batch_size, subset=args.subset, split="train"
    )
    val_loader, _ = get_dataloader(
        args.dataset, batch_size=args.batch_size,
        subset=args.val_size if args.dataset == "synthetic" else None,
        split="val",
    )

    # ---- model ---------------------------------------------------------------
    model = get_model(
        args.model,
        num_classes=num_classes,
        drop_path_rate=args.drop_path,
        use_separable=args.separable,
    ).to(device)

    # ---- loss ----------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss() if is_seg else nn.CrossEntropyLoss()

    # ---- optimiser -----------------------------------------------------------
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    # ---- run directory -------------------------------------------------------
    run_name = f"{args.model}_{args.dataset}_{int(time.time())}"
    outdir = os.path.join(args.out, run_name)
    os.makedirs(outdir, exist_ok=True)
    save_config(vars(args), outdir)

    # ---- training loop -------------------------------------------------------
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        v_metrics = val_epoch(model, val_loader, criterion, device, is_seg)

        row = {"epoch": epoch, "train_loss": round(tr_loss, 5)}
        row.update({k: round(v, 5) for k, v in v_metrics.items()})
        history.append(row)

        parts = [f"epoch {epoch}/{args.epochs}",
                 f"train_loss={tr_loss:.4f}",
                 f"val_loss={v_metrics['val_loss']:.4f}"]
        if is_seg:
            parts.append(f"val_iou={v_metrics.get('val_iou', 0):.4f}")
            parts.append(f"val_f1={v_metrics.get('val_f1', 0):.4f}")
        print("  ".join(parts))

        save_checkpoint(
            {"model_state": model.state_dict(),
             "optim_state": optimiser.state_dict(),
             "epoch": epoch,
             "args": vars(args)},
            outdir,
            name=f"ckpt_epoch{epoch}.pth",
        )

    # ---- save results --------------------------------------------------------
    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"config": vars(args), "history": history}, f, indent=2)

    print(f"\nDone. Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
