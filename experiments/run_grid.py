"""Run a grid of segmentation/classification experiments with multi-seed.

Usage (from repo root):

    python3 -m fnn_rfi.experiments.run_grid --config experiments/grid_seg.json
    python3 -m fnn_rfi.experiments.run_grid --mode seg --epochs 30 --seeds 3

Produces runs/<model>_<params>_s<seed>/ directories and a combined
results JSON at runs/grid_results_<timestamp>.json.
"""
import argparse
import itertools
import json
import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from .evaluate import evaluate_segmentation
from .utils import (
    get_dataloader,
    get_device,
    get_model,
    save_checkpoint,
    save_config,
    set_seed,
)


# ------------------------------------------------------------------
# Default experiment grids
# ------------------------------------------------------------------

SEG_GRID: Dict[str, List[Any]] = {
    "model": ["fnn_seg"],
    "use_fib": [True, False],
    "use_separable": [True, False],
    "drop_path_rate": [0.0, 0.05, 0.1, 0.2],
    "stages": [2, 3, 4],
    "base_ch": [8, 16, 32],
    "max_depth": [2, 3],
}

BASELINE_SEG_GRID: Dict[str, List[Any]] = {
    "model": ["unet", "resunet"],
    "base_ch": [16],
    "stages": [3],
}

CLS_GRID: Dict[str, List[Any]] = {
    "model": ["fnn"],
    "use_fib": [True, False],
    "use_separable": [True, False],
    "drop_path_rate": [0.0, 0.1],
    "stages": [2, 3, 4],
    "base_channels": [8, 16],
}

BASELINE_CLS_GRID: Dict[str, List[Any]] = {
    "model": ["cnn", "rnn", "transformer", "resnet_cls"],
}

INPAINT_GRID: Dict[str, List[Any]] = {
    "model": ["fnn_inpaint"],
    "use_fib": [True],
    "use_separable": [True, False],
    "drop_path_rate": [0.0, 0.1],
    "stages": [3],
    "base_ch": [16, 32],
}

# Segmentation model names (produce pixel masks)
_SEG_MODELS = {"unet", "fnn_seg", "resunet"}


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand a parameter grid dict into a list of individual configs."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def _run_name(cfg: Dict[str, Any], seed: int) -> str:
    name = cfg["model"]
    if cfg.get("use_fib"):
        name += "_fib"
    if cfg.get("use_separable"):
        name += "_sep"
    dp = cfg.get("drop_path_rate", 0.0)
    if dp > 0:
        name += f"_dp{dp}"
    stages = cfg.get("stages", 3)
    name += f"_st{stages}"
    bch = cfg.get("base_ch", cfg.get("base_channels", 16))
    name += f"_ch{bch}"
    name += f"_s{seed}"
    return name


# ------------------------------------------------------------------
# Training loop (self-contained per experiment)
# ------------------------------------------------------------------

def run_single(
    cfg: Dict[str, Any],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    dataset: str,
    out_root: str,
    device: torch.device,
    subset: int = None,
    val_size: int = 512,
) -> Dict[str, Any]:
    """Train one model configuration with one seed. Returns result dict."""
    set_seed(seed)
    model_name = cfg["model"]
    is_seg = model_name in _SEG_MODELS
    is_inpaint = model_name == "fnn_inpaint"

    # Dataset
    ds_name = "synthetic_inpaint" if is_inpaint else dataset
    train_loader, num_classes = get_dataloader(
        ds_name, batch_size=batch_size, subset=subset, split="train"
    )
    val_loader, _ = get_dataloader(
        ds_name, batch_size=batch_size, subset=val_size, split="val"
    )

    # Model
    model_kwargs = dict(cfg)
    model_kwargs.pop("model", None)
    model_kwargs["num_classes"] = num_classes
    model = get_model(model_name, **model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loss and optimizer
    if is_inpaint:
        from ..models.fnn_inpaint import inpaint_loss
        criterion = None  # handled inline
    elif is_seg:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimiser = optim.Adam(model.parameters(), lr=lr)

    # Run directory
    rname = _run_name(cfg, seed)
    outdir = os.path.join(out_root, rname)
    os.makedirs(outdir, exist_ok=True)
    full_cfg = {**cfg, "seed": seed, "epochs": epochs, "lr": lr,
                "batch_size": batch_size, "dataset": dataset, "params": param_count}
    save_config(full_cfg, outdir)

    history = []
    best_metric = -float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        if is_inpaint:
            for contaminated, mask, clean in train_loader:
                contaminated = contaminated.to(device)
                mask = mask.to(device)
                clean = clean.to(device)
                optimiser.zero_grad()
                mask_logit, restored = model(contaminated)
                loss = inpaint_loss(mask_logit, restored, mask, clean)
                loss.backward()
                optimiser.step()
                train_loss += loss.item()
                n_batches += 1
        else:
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                out = model(xb)
                loss = criterion(out, yb) if out.shape == yb.shape else criterion(out, yb.long())
                loss.backward()
                optimiser.step()
                train_loss += loss.item()
                n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_metrics = {}
        with torch.no_grad():
            if is_inpaint:
                from .evaluate import iou_score, psnr, ssim, dice_score
                vl, ious, psnrs, ssims = 0.0, [], [], []
                for contaminated, mask, clean in val_loader:
                    contaminated = contaminated.to(device)
                    mask = mask.to(device)
                    clean = clean.to(device)
                    mask_logit, restored = model(contaminated)
                    vl += inpaint_loss(mask_logit, restored, mask, clean).item()
                    ious.append(iou_score(mask_logit, mask))
                    psnrs.append(psnr(clean, restored))
                    ssims.append(ssim(clean, restored))
                n = max(len(ious), 1)
                val_metrics = {
                    "val_loss": vl / n,
                    "val_iou": sum(ious) / n,
                    "val_psnr": sum(psnrs) / n,
                    "val_ssim": sum(ssims) / n,
                }
            elif is_seg:
                vl, n = 0.0, 0
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    vl += criterion(out, yb).item()
                    n += 1
                val_metrics["val_loss"] = vl / max(n, 1)
                seg_m = evaluate_segmentation(model, val_loader, device=str(device))
                val_metrics.update({f"val_{k}": v for k, v in seg_m.items()})
            else:
                vl, correct, total_samples, n = 0.0, 0, 0, 0
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    vl += criterion(out, yb.long()).item()
                    n += 1
                    preds = out.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total_samples += yb.numel()
                val_metrics["val_loss"] = vl / max(n, 1)
                val_metrics["val_acc"] = correct / max(total_samples, 1)

        row = {"epoch": epoch, "train_loss": round(train_loss, 5)}
        row.update({k: round(v, 5) for k, v in val_metrics.items()})
        history.append(row)

        # Track best
        primary = val_metrics.get("val_iou", val_metrics.get("val_acc", -val_metrics.get("val_loss", 0)))
        if primary > best_metric:
            best_metric = primary
            best_epoch = epoch
            save_checkpoint(
                {"model_state": model.state_dict(), "epoch": epoch, "config": full_cfg},
                outdir, name="best.pth",
            )

    # Save last checkpoint and history
    save_checkpoint(
        {"model_state": model.state_dict(), "epoch": epochs, "config": full_cfg},
        outdir, name="last.pth",
    )
    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"config": full_cfg, "history": history}, f, indent=2)

    # Summary
    final = history[-1] if history else {}
    summary = {
        "run_name": rname,
        "model": model_name,
        "config": full_cfg,
        "params": param_count,
        "best_epoch": best_epoch,
        "final_metrics": final,
        "outdir": outdir,
    }
    return summary


# ------------------------------------------------------------------
# Grid runner
# ------------------------------------------------------------------

def run_grid(
    grids: List[Dict[str, List[Any]]],
    seeds: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    dataset: str,
    out_root: str,
    subset: int = None,
    val_size: int = 512,
) -> List[Dict[str, Any]]:
    """Run all configs x seeds. Returns list of result summaries."""
    device = get_device()
    print(f"Device: {device}")

    all_configs = []
    for grid in grids:
        all_configs.extend(_expand_grid(grid))

    total = len(all_configs) * len(seeds)
    print(f"Total experiments: {total} ({len(all_configs)} configs x {len(seeds)} seeds)")

    results = []
    for i, cfg in enumerate(all_configs):
        for seed in seeds:
            rname = _run_name(cfg, seed)
            outdir = os.path.join(out_root, rname)
            # Skip if already completed
            if os.path.exists(os.path.join(outdir, "results.json")):
                print(f"  [{i+1}/{len(all_configs)}] SKIP {rname} (exists)")
                with open(os.path.join(outdir, "results.json")) as f:
                    prev = json.load(f)
                results.append({
                    "run_name": rname,
                    "model": cfg["model"],
                    "config": prev.get("config", cfg),
                    "params": prev.get("config", {}).get("params", 0),
                    "final_metrics": prev["history"][-1] if prev.get("history") else {},
                    "outdir": outdir,
                })
                continue

            idx = len(results) + 1
            print(f"  [{idx}/{total}] {rname}  ...", end=" ", flush=True)
            t0 = time.time()
            try:
                summary = run_single(
                    cfg, seed, epochs, batch_size, lr, dataset,
                    out_root, device, subset, val_size,
                )
                dt = time.time() - t0
                iou = summary["final_metrics"].get("val_iou", "")
                acc = summary["final_metrics"].get("val_acc", "")
                metric_str = f"IoU={iou:.4f}" if iou else (f"Acc={acc:.4f}" if acc else "")
                print(f"done in {dt:.1f}s  {metric_str}")
                results.append(summary)
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({"run_name": rname, "error": str(e)})

    # Save combined results
    combined_path = os.path.join(out_root, f"grid_results_{int(time.time())}.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGrid results saved to {combined_path}")
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run experiment grid for FNN-RFI.")
    parser.add_argument("--mode", default="seg",
                        choices=["seg", "cls", "inpaint", "all"],
                        help="Which grid to run.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (42, 123, 7).")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--config", default=None,
                        help="JSON file with custom grid(s).")
    args = parser.parse_args()

    seed_list = [42, 123, 7][:args.seeds]

    if args.config:
        with open(args.config) as f:
            grids = json.load(f)
        if isinstance(grids, dict):
            grids = [grids]
    else:
        grids = []
        if args.mode in ("seg", "all"):
            grids.extend([SEG_GRID, BASELINE_SEG_GRID])
        if args.mode in ("cls", "all"):
            grids.extend([CLS_GRID, BASELINE_CLS_GRID])
        if args.mode in ("inpaint", "all"):
            grids.append(INPAINT_GRID)

    run_grid(
        grids, seed_list, args.epochs, args.batch_size, args.lr,
        args.dataset, args.out, args.subset, args.val_size,
    )


if __name__ == "__main__":
    main()
