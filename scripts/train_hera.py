#!/usr/bin/env python3
"""Train a segmentation model on real HERA or LOFAR data.

Compares against the du Toit et al. benchmark (Zenodo 8275061) by:
  - Using the same HERA_28-03-2023_all.pkl data
  - Reporting test accuracy, F1, and AUC to stdout and a results JSON

Usage examples:
    # FNNSeg (best config from synthetic ablation)
    python3 fnn_rfi/scripts/train_hera.py \
        --model fnn_seg --base_ch 16 --use_fib --patch 256 --epochs 30

    # UNet baseline (for comparison)
    python3 fnn_rfi/scripts/train_hera.py \
        --model unet --base_ch 16 --depth 3 --patch 256 --epochs 30

    # Quick smoke test (5 epochs, small subset)
    python3 fnn_rfi/scripts/train_hera.py \
        --model fnn_seg --epochs 5 --subset 256 --batch 4

    # Use AOFlagger-labelled data
    python3 fnn_rfi/scripts/train_hera.py \
        --pkl data/zenodo/HERA_AOF_20-07-2023_all.pkl --model fnn_seg

Full 3-seed experiment:
    for seed in 0 1 2; do
        python3 fnn_rfi/scripts/train_hera.py --model fnn_seg --seed $seed
    done
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score

from fnn_rfi.data.hera_dataset import build_hera_loaders
from fnn_rfi.experiments.utils import get_model, get_device, set_seed


def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── loss ────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """Combined binary cross-entropy + Dice loss (common for medical/RFI segmentation)."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        inter = (probs * targets).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + smooth) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


# ─── metrics ─────────────────────────────────────────────────────────────────

def pixel_accuracy(logits: torch.Tensor, masks: torch.Tensor) -> float:
    preds = (logits.sigmoid() > 0.5).float()
    return (preds == masks).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn) -> Dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = total_acc = n_batches = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            total_acc += pixel_accuracy(logits, masks)
            n_batches += 1
            all_probs.append(logits.sigmoid().cpu().numpy().ravel())
            all_labels.append(masks.cpu().numpy().ravel().astype(int))

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    auc   = float(roc_auc_score(labels, probs)) if labels.sum() > 0 else 0.0
    f1    = float(f1_score(labels, (probs > 0.5).astype(int), zero_division=0))

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "f1": f1,
        "auc": auc,
    }


# ─── training loop ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, loss_fn, device) -> Dict:
    model.train()
    total_loss = total_acc = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimiser.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        total_acc += pixel_accuracy(logits, masks)
    n = len(loader)
    return {"loss": total_loss / n, "accuracy": total_acc / n}


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train segmentation model on HERA/LOFAR RFI data")

    # Data
    parser.add_argument("--pkl", default="data/zenodo/HERA_28-03-2023_all.pkl")
    parser.add_argument("--patch", type=int, default=256, help="Patch size (default 256×256)")
    parser.add_argument("--stride_train", type=int, default=None,
                        help="Stride for training patches (default = patch//2 for 50% overlap)")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=None, help="Limit training patches to N")
    parser.add_argument("--eval_subset", type=int, default=None,
                        help="Limit val+test batches to N for quick smoke tests")

    # Model
    parser.add_argument("--model", default="fnn_seg",
                        choices=["fnn_seg", "unet", "resunet", "cnn"])
    parser.add_argument("--in_ch", type=int, default=2, help="Input channels (1=amp only, 2=amp+phase)")
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3, help="Depth/stages parameter")
    parser.add_argument("--use_fib", action="store_true", default=True)
    parser.add_argument("--no_fib", dest="use_fib", action="store_false")
    parser.add_argument("--use_sep", action="store_true", default=False)
    parser.add_argument("--drop_path", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience based on val_f1 (0 = disabled)")

    # Output
    parser.add_argument("--outdir", default=None,
                        help="Directory for results JSON (auto-generated if not set)")

    args = parser.parse_args()

    # --- setup ---
    set_seed(args.seed)
    device = get_device()
    pkl_name = Path(args.pkl).stem
    stride_train = args.stride_train or (args.patch // 2)

    _log(f"=== train_hera start ===")
    _log(f"Model:   {args.model}  base_ch={args.base_ch}  depth={args.depth}  "
         f"fib={args.use_fib}  sep={args.use_sep}  drop_path={args.drop_path}")
    _log(f"Dataset: {args.pkl}")
    _log(f"Patch:   {args.patch}px  stride_train={stride_train}px  batch={args.batch}")
    _log(f"Epochs:  {args.epochs}  lr={args.lr}  seed={args.seed}")
    _log(f"Device:  {device}")

    # --- data ---
    train_loader, val_loader, test_loader, stats = build_hera_loaders(
        pkl_path=args.pkl,
        batch_size=args.batch,
        patch_size=args.patch,
        stride=stride_train,
        log_amplitude=True,
        normalize=True,
        val_fraction=args.val_frac,
        seed=args.seed,
    )
    if args.subset:
        from torch.utils.data import Subset as _Subset
        train_ds = _Subset(train_loader.dataset, list(range(min(args.subset, len(train_loader.dataset)))))
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    if args.eval_subset:
        from torch.utils.data import Subset as _Subset
        n_val = min(args.eval_subset, len(val_loader.dataset))
        n_test = min(args.eval_subset, len(test_loader.dataset))
        val_loader  = DataLoader(_Subset(val_loader.dataset,  list(range(n_val))),
                                 batch_size=args.batch, shuffle=False, num_workers=0)
        test_loader = DataLoader(_Subset(test_loader.dataset, list(range(n_test))),
                                 batch_size=args.batch, shuffle=False, num_workers=0)

    _log(f"Train batches: {len(train_loader)}  "
         f"Val batches: {len(val_loader)}  "
         f"Test batches: {len(test_loader)}")

    # --- model ---
    model = get_model(
        args.model,
        in_ch=args.in_ch,
        base_ch=args.base_ch,
        stages=args.depth,
        depth=args.depth,
        use_fib=args.use_fib,
        use_separable=args.use_sep,
        drop_path_rate=args.drop_path,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model built: {args.model}  trainable params={n_params:,}")

    # --- output directory (created before training so checkpoints can be saved mid-run) ---
    model_tag = (
        f"{args.model}"
        f"_ch{args.base_ch}"
        f"_d{args.depth}"
        f"{'_fib' if args.use_fib else '_std'}"
        f"{'_sep' if args.use_sep else ''}"
        f"_seed{args.seed}"
    )
    outdir = args.outdir or f"fnn_rfi/results/hera/{pkl_name}/{model_tag}"
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, "best_model.pt")

    # --- training ---
    loss_fn = DiceBCELoss(bce_weight=0.5)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_val_acc = 0.0
    best_val_f1  = 0.0
    no_improve   = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimiser, loss_fn, device)
        vl = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        history.append({"epoch": epoch, "train": tr, "val": vl})
        _log(f"Epoch {epoch:3d}/{args.epochs} | "
             f"train_loss={tr['loss']:.4f}  train_acc={tr['accuracy']:.4f} | "
             f"val_loss={vl['loss']:.4f}  val_acc={vl['accuracy']:.4f}  "
             f"val_f1={vl['f1']:.4f}  val_auc={vl['auc']:.4f}")

        if vl["accuracy"] > best_val_acc:
            best_val_acc = vl["accuracy"]
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ckpt_path)

        if vl["f1"] > best_val_f1:
            best_val_f1 = vl["f1"]
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            _log(f"Early stopping: no val_f1 improvement for {args.patience} epochs (epoch {epoch})")
            break

    # --- test evaluation with best checkpoint ---
    _log("Training complete — evaluating best checkpoint on test set …")
    _log(f"Checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    test_metrics = evaluate(model, test_loader, device, loss_fn)
    _log(f"=== Test results (best val checkpoint) ===")
    _log(f"  test_accuracy : {test_metrics['accuracy']:.6f}")
    _log(f"  test_f1       : {test_metrics['f1']:.6f}")
    _log(f"  test_auc      : {test_metrics['auc']:.6f}")
    _log(f"  best_val_acc  : {best_val_acc:.6f}")

    # --- save results ---
    results = {
        "dataset": pkl_name,
        "model": args.model,
        "model_tag": model_tag,
        "params": n_params,
        "seed": args.seed,
        "epochs": args.epochs,
        "patch_size": args.patch,
        "in_ch": args.in_ch,
        "base_ch": args.base_ch,
        "depth": args.depth,
        "use_fib": args.use_fib,
        "use_sep": args.use_sep,
        "drop_path": args.drop_path,
        "best_val_acc": best_val_acc,
        "checkpoint": ckpt_path,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "history": history,
        "normalisation": stats,
    }

    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _log(f"Results saved to {results_path}")
    _log(f"=== train_hera done ===")


if __name__ == "__main__":
    main()
