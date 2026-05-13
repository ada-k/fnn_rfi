#!/usr/bin/env python3
"""Post-training evaluation: learning curves, test-set visualization,
and comparison against the du Toit et al. (2023) published benchmark.

Modes
-----
  curves   – plot training/validation curves for one run directory
  predict  – side-by-side test predictions: amplitude | GT | pred | error map
             (requires best_model.pt saved by the updated train_hera.py)
  compare  – bar chart + efficiency scatter: our runs vs du Toit published
  all      – curves + compare  (default; works without checkpoints)

Examples
--------
  # Learning curves for a single completed run:
  python3 fnn_rfi/scripts/eval_hera.py --mode curves \\
      --results_dir fnn_rfi/results/hera/HERA_28-03-2023_all/fnn_seg_ch16_d3_fib_seed0

  # Full comparison report (all HERA runs vs du Toit):
  python3 fnn_rfi/scripts/eval_hera.py --mode compare \\
      --results_root fnn_rfi/results/hera/HERA_28-03-2023_all \\
      --dutoit_json  fnn_rfi/figs/dutoit_benchmark.json \\
      --outdir       fnn_rfi/figs/eval

  # Test predictions (needs best_model.pt from a new run):
  python3 fnn_rfi/scripts/eval_hera.py --mode predict \\
      --results_dir fnn_rfi/results/hera/HERA_28-03-2023_all/fnn_seg_ch16_d3_fib_seed0 \\
      --pkl data/zenodo/HERA_28-03-2023_all.pkl \\
      --n_samples 6

  # Everything at once (curves + compare + predict) for one run:
  python3 fnn_rfi/scripts/eval_hera.py --mode all \\
      --results_dir fnn_rfi/results/hera/HERA_28-03-2023_all/fnn_seg_ch16_d3_fib_seed0 \\
      --pkl data/zenodo/HERA_28-03-2023_all.pkl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ──────────────────────────── helpers ────────────────────────────────────────

def _log(msg: str):
    print(f"  {msg}", flush=True)


def load_results(results_dir: str) -> dict:
    p = os.path.join(results_dir, "results.json")
    with open(p) as f:
        d = json.load(f)
    d["_results_dir"] = results_dir
    return d


def _savefig(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved: {path}")


def _patch_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    return (2 * tp) / (2 * tp + fp + fn + 1e-8)


# ──────────────────────────── mode: curves ───────────────────────────────────

def plot_curves(results: dict, outdir: str):
    """Plot train/val loss, F1, and AUC over epochs."""
    history = results.get("history", [])
    if not history:
        _log("No history found in results.json — skipping curves.")
        return

    epochs     = [h["epoch"]                for h in history]
    train_loss = [h["train"]["loss"]         for h in history]
    val_loss   = [h["val"]["loss"]           for h in history]
    val_f1     = [h["val"]["f1"]             for h in history]
    val_auc    = [h["val"]["auc"]            for h in history]
    train_acc  = [h["train"]["accuracy"]     for h in history]
    val_acc    = [h["val"]["accuracy"]       for h in history]

    test_f1  = results.get("test_f1",  None)
    test_auc = results.get("test_auc", None)
    tag      = results.get("model_tag", Path(results.get("_results_dir", "run")).name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, train_loss, label="train", color="#1f77b4")
    axes[0].plot(epochs, val_loss,   label="val",   color="#ff7f0e")
    axes[0].set_title("DiceBCE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, label="train", color="#1f77b4")
    axes[1].plot(epochs, val_acc,   label="val",   color="#ff7f0e")
    axes[1].set_title("Pixel Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.99, 1.001)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 + AUC
    ax3 = axes[2]
    ax3.plot(epochs, val_f1,  label="val F1",  color="#2ca02c")
    ax3.plot(epochs, val_auc, label="val AUC", color="#9467bd", ls="--")
    if test_f1 is not None:
        ax3.axhline(test_f1,  color="#2ca02c", ls=":", lw=1.5,
                    label=f"test F1={test_f1:.4f}")
    if test_auc is not None:
        ax3.axhline(test_auc, color="#9467bd", ls=":", lw=1.5,
                    label=f"test AUC={test_auc:.4f}")
    ax3.set_title("F1 & AUC")
    ax3.set_xlabel("Epoch")
    ax3.set_ylim(0.9, 1.005)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(tag, fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, f"curves_{tag}.png"))


# ──────────────────────────── mode: predict ──────────────────────────────────

def plot_predictions(results: dict, pkl_path: str, n_samples: int, outdir: str):
    """Load best_model.pt, run inference on n_samples test patches,
    plot: amplitude | phase | ground truth | prediction | FP/FN map."""
    import torch
    from fnn_rfi.data.hera_dataset import load_pkl, HERADataset
    from fnn_rfi.experiments.utils import get_model

    results_dir = results.get("_results_dir", "")
    ckpt_path   = os.path.join(results_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        _log(f"[SKIP predict] No checkpoint at {ckpt_path}")
        _log("  Re-run training with the updated train_hera.py to generate best_model.pt.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Loading checkpoint from {ckpt_path}  (device={device})")

    # Rebuild model from saved config
    model = get_model(
        results["model"],
        in_ch=results.get("in_ch", 2),
        base_ch=results.get("base_ch", 16),
        stages=results.get("depth", 3),
        depth=results.get("depth", 3),
        use_fib=results.get("use_fib", True),
        use_separable=results.get("use_sep", False),
        drop_path_rate=results.get("drop_path", 0.0),
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Normalisation stats saved during training
    norm      = results.get("normalisation", {})
    amp_stats = norm.get("amplitude_stats")
    ph_stats  = norm.get("phase_stats")

    _log(f"Loading test split from {pkl_path} …")
    arrays = load_pkl(pkl_path)
    test_ds = HERADataset(
        arrays=arrays,
        split="test",
        patch_size=results.get("patch_size", 256),
        stride=results.get("patch_size", 256),   # non-overlapping for test
        log_amplitude=True,
        normalize=True,
        val_fraction=results.get("val_frac", 0.1),
        seed=results.get("seed", 0),
        amplitude_stats=tuple(amp_stats) if amp_stats else None,
        phase_stats=tuple(ph_stats)      if ph_stats  else None,
    )

    # Pick evenly-spaced patches that have RFI (more informative)
    rfi_idx = [i for i in range(len(test_ds)) if test_ds[i][1].sum() > 0]
    if len(rfi_idx) < n_samples:
        rfi_idx = list(range(len(test_ds)))
    step = max(1, len(rfi_idx) // n_samples)
    indices = rfi_idx[::step][:n_samples]

    n_cols = 5
    col_titles = ["Amplitude (norm.)", "Phase (norm.)", "Ground Truth", "Prediction (p>0.5)", "FP (red) / FN (blue)"]

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 3.2, n_samples * 3.2))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight="bold")

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask = test_ds[idx]           # (2,H,W), (1,H,W)
            inp  = img.unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            gt   = mask.squeeze().numpy().astype(np.uint8)
            amp  = img[0].numpy()
            ph   = img[1].numpy()

            # FP/FN colour map: TP=white, FP=red, FN=blue, TN=black
            diff = np.zeros((*gt.shape, 3), dtype=np.float32)
            diff[..., 0] += pred * (1 - gt)   # FP → red channel
            diff[..., 2] += (1 - pred) * gt    # FN → blue channel
            tp_mask = (pred * gt).astype(bool)
            diff[tp_mask, :] = 1.0              # TP → white

            row_ax = axes[row]
            row_ax[0].imshow(amp,  cmap="inferno", aspect="auto")
            row_ax[1].imshow(ph,   cmap="RdBu",    aspect="auto")
            row_ax[2].imshow(gt,   cmap="Greys",   vmin=0, vmax=1, aspect="auto")
            row_ax[3].imshow(pred, cmap="Greys",   vmin=0, vmax=1, aspect="auto")
            row_ax[4].imshow(diff, aspect="auto")

            row_ax[0].set_ylabel(f"patch {idx}", fontsize=8)
            rfi_frac = gt.mean()
            row_ax[2].set_xlabel(f"RFI={rfi_frac:.2%}", fontsize=7)
            row_ax[3].set_xlabel(f"F1={_patch_f1(pred, gt):.3f}", fontsize=7)

            for ax in row_ax:
                ax.set_xticks([])
                ax.set_yticks([])

    tag = results.get("model_tag", "run")
    fig.suptitle(f"Test predictions — {tag}", fontsize=12)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, f"predictions_{tag}.png"), dpi=120)


# ──────────────────────────── mode: compare ──────────────────────────────────

# Key du Toit models to include in the comparison figure
DUTOIT_MODELS = {
    "UNET":        "UNet (du Toit)",
    "DSC_RESUNET": "ResUNet (du Toit)",
    "RNET":        "RNET (du Toit)",
    "RFI_NET":     "RFI-NET (du Toit)",
}

OUR_DISPLAY = {
    "fnn_seg": "FNNSeg (ours)",
    "unet":    "UNet (ours)",
    "resunet": "ResUNet (ours)",
}


def load_our_results(results_root: str) -> dict:
    """Walk results_root, load all results.json, group by model name."""
    groups = defaultdict(list)
    root = Path(results_root)
    if not root.exists():
        return groups
    for rdir in sorted(root.iterdir()):
        rjson = rdir / "results.json"
        if not rjson.exists():
            continue
        with open(rjson) as f:
            r = json.load(f)
        model_name = r.get("model", rdir.name)
        groups[model_name].append(r)
    return groups


def plot_comparison(groups: dict, dutoit_json: str, outdir: str):
    """F1 bar chart + efficiency scatter (params vs F1) comparing our models
    against the du Toit et al. (2023) published best results."""

    with open(dutoit_json) as f:
        dut = json.load(f)
    per_model = dut.get("per_model", {})

    # ── aggregate our results (mean ± std across seeds) ──────────────────────
    our_labels   = []
    our_f1_mean  = []; our_f1_std  = []
    our_auc_mean = []; our_auc_std = []
    our_params   = []

    for model_name in ["fnn_seg", "unet", "resunet"]:
        runs = groups.get(model_name, [])
        if not runs:
            continue
        f1s  = [r["test_f1"]  for r in runs]
        aucs = [r["test_auc"] for r in runs]
        our_labels.append(OUR_DISPLAY.get(model_name, model_name))
        our_f1_mean.append(float(np.mean(f1s)))
        our_f1_std.append(float(np.std(f1s)) if len(f1s) > 1 else 0.0)
        our_auc_mean.append(float(np.mean(aucs)))
        our_auc_std.append(float(np.std(aucs)) if len(aucs) > 1 else 0.0)
        our_params.append(float(np.mean([r.get("params", 0) for r in runs])))

    # ── du Toit reference (best published value per model) ───────────────────
    dut_labels = []; dut_f1 = []; dut_auc = []; dut_params = []
    for key, disp in DUTOIT_MODELS.items():
        if key not in per_model:
            continue
        m = per_model[key]
        dut_labels.append(disp)
        dut_f1.append(m["test_f1"]["max"])
        dut_auc.append(m["test_auroc"]["max"])
        dut_params.append(m["n_weights"]["mean"])

    # ── Figure 1: F1 bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    bar_kw = dict(width=0.6, edgecolor="black", linewidth=0.5)
    x_our = np.arange(len(our_labels))
    x_dut = np.arange(len(dut_labels)) + len(our_labels) + 0.8

    bars_our = ax.bar(x_our, our_f1_mean, color="#2196F3", label="This work", **bar_kw)
    ax.errorbar(x_our, our_f1_mean, yerr=our_f1_std,
                fmt="none", color="black", capsize=4, lw=1.2)
    bars_dut = ax.bar(x_dut, dut_f1, color="#FF9800",
                      label="du Toit et al. (2023) — best published", **bar_kw)

    all_x      = list(x_our) + list(x_dut)
    all_labels = our_labels + dut_labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=9)
    ax.axvline(len(our_labels) - 0.2, color="grey", ls="--", lw=0.8)

    ax.set_ylabel("Test F1 Score")
    ax.set_title("RFI Detection Performance: This Work vs. du Toit et al. (2023)")
    ymin = max(0.94, min(our_f1_mean + dut_f1) - 0.01)
    ax.set_ylim(ymin, 1.006)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, val, std in zip(bars_our, our_f1_mean, our_f1_std):
        label = f"{val:.4f}" + (f"\n±{std:.4f}" if std > 0 else "")
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0003, label,
                ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars_dut, dut_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0003, f"{val:.4f}",
                ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, "comparison_f1.png"))

    # ── Figure 2: AUC bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    bars_our = ax.bar(x_our, our_auc_mean, color="#2196F3", label="This work", **bar_kw)
    ax.errorbar(x_our, our_auc_mean, yerr=our_auc_std,
                fmt="none", color="black", capsize=4, lw=1.2)
    bars_dut = ax.bar(x_dut, dut_auc, color="#FF9800",
                      label="du Toit et al. (2023) — best published", **bar_kw)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=9)
    ax.axvline(len(our_labels) - 0.2, color="grey", ls="--", lw=0.8)
    ax.set_ylabel("Test AUROC")
    ax.set_title("RFI Detection AUC: This Work vs. du Toit et al. (2023)")
    ymin_auc = max(0.94, min(our_auc_mean + dut_auc) - 0.01)
    ax.set_ylim(ymin_auc, 1.006)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, "comparison_auc.png"))

    # ── Figure 3: Efficiency scatter (params vs F1) ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_our = ["#2196F3", "#03A9F4", "#00BCD4"]
    for i, (lbl, pm, f1, std) in enumerate(zip(our_labels, our_params, our_f1_mean, our_f1_std)):
        c = colors_our[i % len(colors_our)]
        ax.scatter(pm / 1e3, f1, s=140, marker="*", color=c, zorder=5, label=lbl)
        if std > 0:
            ax.errorbar(pm / 1e3, f1, yerr=std, fmt="none", color=c, capsize=3)

    colors_dut = ["#FF9800", "#F44336", "#4CAF50", "#9C27B0"]
    for i, (lbl, pm, f1) in enumerate(zip(dut_labels, dut_params, dut_f1)):
        c = colors_dut[i % len(colors_dut)]
        ax.scatter(pm / 1e3, f1, s=80, marker="o", color=c, alpha=0.85, label=lbl)

    ax.set_xlabel("Trainable Parameters (K)")
    ax.set_ylabel("Test F1 Score")
    ax.set_title("Model Efficiency: Parameters vs. F1 Score")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, "efficiency_scatter.png"))

    # ── Summary table (stdout) ────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print(f"{'Model':<30} {'F1 (mean±std)':<22} {'AUC (mean±std)':<22} {'Params':>8}")
    print("-" * 76)
    for lbl, f1m, f1s, aucm, aucs, pm in zip(
            our_labels, our_f1_mean, our_f1_std,
            our_auc_mean, our_auc_std, our_params):
        n_seeds = next(
            (len(runs) for mn, runs in groups.items()
             if OUR_DISPLAY.get(mn) == lbl), 1)
        f1_str  = f"{f1m:.4f}±{f1s:.4f}" if f1s > 0 else f"{f1m:.4f} (n=1)"
        auc_str = f"{aucm:.4f}±{aucs:.4f}" if aucs > 0 else f"{aucm:.4f}"
        print(f"{lbl:<30} {f1_str:<22} {auc_str:<22} {pm/1e3:>6.1f}K")
    print("-" * 76)
    for lbl, f1, auc, pm in zip(dut_labels, dut_f1, dut_auc, dut_params):
        print(f"{lbl:<30} {f1:.4f} (best pub.){'':8} {auc:.4f} (best pub.){'':4} {pm/1e3:>6.1f}K")
    print("=" * 76)


# ──────────────────────────── main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-training evaluation and comparison for HERA/LOFAR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", default="all",
                        choices=["curves", "predict", "compare", "all"],
                        help="Evaluation mode (default: all = curves + compare)")

    # Single-run options (curves / predict)
    parser.add_argument("--results_dir", default=None,
                        help="Path to a single run directory containing results.json")
    parser.add_argument("--pkl", default="data/zenodo/HERA_28-03-2023_all.pkl",
                        help="Pickle path (required for --mode predict)")
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Number of test patches to visualize in predict mode")

    # Multi-run / comparison options
    parser.add_argument("--results_root",
                        default="fnn_rfi/results/hera/HERA_28-03-2023_all",
                        help="Root dir containing all per-run subdirectories")
    parser.add_argument("--dutoit_json",
                        default="fnn_rfi/figs/dutoit_benchmark.json",
                        help="du Toit benchmark JSON from extract_benchmarks.py")

    # Output
    parser.add_argument("--outdir", default="fnn_rfi/figs/eval",
                        help="Directory to write figures (default: fnn_rfi/figs/eval)")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── curves ──
    if args.mode in ("curves", "all"):
        if args.results_dir:
            print(f"\n[curves] {args.results_dir}")
            r = load_results(args.results_dir)
            plot_curves(r, args.outdir)
        elif args.mode == "curves":
            parser.error("--mode curves requires --results_dir")

    # ── predict ──
    if args.mode == "predict":
        if not args.results_dir:
            parser.error("--mode predict requires --results_dir")
        print(f"\n[predict] {args.results_dir}")
        r = load_results(args.results_dir)
        plot_predictions(r, args.pkl, args.n_samples, args.outdir)

    # ── also run predict when mode=all and results_dir given ──
    if args.mode == "all" and args.results_dir:
        print(f"\n[predict] {args.results_dir}")
        r = load_results(args.results_dir)
        plot_predictions(r, args.pkl, args.n_samples, args.outdir)

    # ── compare ──
    if args.mode in ("compare", "all"):
        print(f"\n[compare] scanning {args.results_root}")
        groups = load_our_results(args.results_root)
        if not groups:
            print("  No results.json files found — run experiments first.")
        else:
            if not os.path.exists(args.dutoit_json):
                print(f"  [WARN] du Toit JSON not found: {args.dutoit_json}")
                print("  Run: python3 fnn_rfi/scripts/extract_benchmarks.py first.")
            else:
                plot_comparison(groups, args.dutoit_json, args.outdir)

    print(f"\nDone — figures in: {args.outdir}")


if __name__ == "__main__":
    main()
