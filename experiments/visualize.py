"""Visualization utilities for FNN-RFI experiments.

Generates publication-quality figures for the thesis:
- Spectrogram examples (original, contaminated, mask, restored)
- Learning curves
- Ablation heatmaps
- Pareto efficiency plots
- Architecture diagrams (text-based)
- SNR sweep curves
- Metric comparison bar charts

All functions save to a specified output directory and return the figure path.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.figsize": (10, 6),
})


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if "/" in path else ".", exist_ok=True)


# ------------------------------------------------------------------
# 1. Spectrogram panel visualizations
# ------------------------------------------------------------------

def plot_spectrogram_samples(
    dataset,
    n_samples: int = 4,
    out_path: str = "figs/spectrogram_samples.png",
) -> str:
    """Plot original + mask side by side for n samples from dataset."""
    _ensure_dir(out_path)
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        sample = dataset[i]
        if len(sample) == 3:
            spec, mask, clean = sample
        else:
            spec, mask = sample
            clean = None

        spec_np = spec.squeeze().numpy()
        mask_np = mask.squeeze().numpy()

        axes[i, 0].imshow(spec_np, aspect="auto", cmap="viridis", origin="lower")
        axes[i, 0].set_title(f"Spectrogram #{i}")
        axes[i, 0].set_ylabel("Time")

        axes[i, 1].imshow(mask_np, aspect="auto", cmap="Reds", origin="lower",
                          vmin=0, vmax=1)
        axes[i, 1].set_title(f"RFI Mask #{i} (frac={mask_np.mean():.3f})")

    axes[-1, 0].set_xlabel("Frequency")
    axes[-1, 1].set_xlabel("Frequency")
    plt.suptitle("Synthetic Spectrogram Samples", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_inpaint_examples(
    model: torch.nn.Module,
    dataset,
    n_samples: int = 4,
    device: str = "cpu",
    out_path: str = "figs/inpaint_examples.png",
) -> str:
    """Show contaminated / predicted mask / restored / clean for inpaint model."""
    _ensure_dir(out_path)
    model.eval()
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    titles = ["Contaminated", "Predicted Mask", "Restored", "Clean (GT)"]

    with torch.no_grad():
        for i in range(n_samples):
            contaminated, mask_gt, clean = dataset[i]
            inp = contaminated.unsqueeze(0).to(device)
            mask_logit, restored = model(inp)
            pred_mask = torch.sigmoid(mask_logit).squeeze().cpu().numpy()
            restored_np = restored.squeeze().cpu().numpy()

            axes[i, 0].imshow(contaminated.squeeze().numpy(), aspect="auto",
                              cmap="viridis", origin="lower")
            axes[i, 1].imshow(pred_mask, aspect="auto", cmap="Reds",
                              origin="lower", vmin=0, vmax=1)
            axes[i, 2].imshow(restored_np, aspect="auto", cmap="viridis",
                              origin="lower")
            axes[i, 3].imshow(clean.squeeze().numpy(), aspect="auto",
                              cmap="viridis", origin="lower")

            for j in range(4):
                if i == 0:
                    axes[i, j].set_title(titles[j])
                axes[i, j].set_ylabel(f"Sample {i}" if j == 0 else "")
                if i == n_samples - 1:
                    axes[i, j].set_xlabel("Frequency")

    plt.suptitle("FNNInpaint: Segmentation + Restoration", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_segmentation_examples(
    model: torch.nn.Module,
    dataset,
    n_samples: int = 4,
    device: str = "cpu",
    out_path: str = "figs/seg_examples.png",
) -> str:
    """Show spectrogram / GT mask / predicted mask for segmentation model."""
    _ensure_dir(out_path)
    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(n_samples):
            spec, mask_gt = dataset[i]
            inp = spec.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            spec_np = spec.squeeze().numpy()
            mask_np = mask_gt.squeeze().numpy()

            axes[i, 0].imshow(spec_np, aspect="auto", cmap="viridis", origin="lower")
            axes[i, 0].set_title("Input" if i == 0 else "")
            axes[i, 1].imshow(mask_np, aspect="auto", cmap="Reds", origin="lower",
                              vmin=0, vmax=1)
            axes[i, 1].set_title("GT Mask" if i == 0 else "")
            axes[i, 2].imshow(pred, aspect="auto", cmap="Reds", origin="lower",
                              vmin=0, vmax=1)
            axes[i, 2].set_title("Predicted Mask" if i == 0 else "")

    plt.suptitle("Segmentation Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------------------------------------------------
# 2. Learning curves
# ------------------------------------------------------------------

def plot_learning_curves(
    results_paths: List[str],
    labels: List[str] = None,
    metric: str = "val_iou",
    out_path: str = "figs/learning_curves.png",
) -> str:
    """Plot a metric over epochs for multiple runs."""
    _ensure_dir(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, rpath in enumerate(results_paths):
        with open(rpath) as f:
            data = json.load(f)
        history = data["history"]
        label = labels[i] if labels else os.path.basename(os.path.dirname(rpath))
        epochs = [h["epoch"] for h in history]

        ax1.plot(epochs, [h["train_loss"] for h in history], label=label)
        if metric in history[0]:
            ax2.plot(epochs, [h[metric] for h in history], label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric)
    ax2.set_title(f"Validation {metric}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_multi_metric_curves(
    results_path: str,
    metrics: List[str] = None,
    out_path: str = "figs/multi_metric.png",
) -> str:
    """Plot multiple metrics from a single run."""
    _ensure_dir(out_path)
    with open(results_path) as f:
        data = json.load(f)
    history = data["history"]
    epochs = [h["epoch"] for h in history]

    if metrics is None:
        metrics = [k for k in history[0].keys() if k not in ("epoch", "train_loss")]

    n = len(metrics)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, m in enumerate(metrics):
        vals = [h.get(m, 0) for h in history]
        axes[i].plot(epochs, vals, "o-", markersize=3)
        axes[i].set_title(m)
        axes[i].set_xlabel("Epoch")
        axes[i].grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Metrics: {os.path.basename(results_path)}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------------------------------------------------
# 3. Ablation heatmaps
# ------------------------------------------------------------------

def plot_ablation_heatmap(
    grid_results: List[Dict],
    row_key: str,
    col_key: str,
    metric: str = "val_iou",
    out_path: str = "figs/ablation_heatmap.png",
    aggregate: str = "mean",
) -> str:
    """Create a heatmap of metric values over two grid dimensions."""
    _ensure_dir(out_path)

    # Collect values
    data = {}
    for r in grid_results:
        if "error" in r:
            continue
        cfg = r.get("config", {})
        rv = cfg.get(row_key, r.get(row_key))
        cv = cfg.get(col_key, r.get(col_key))
        val = r.get("final_metrics", {}).get(metric)
        if rv is not None and cv is not None and val is not None:
            data.setdefault((rv, cv), []).append(val)

    if not data:
        return ""

    rows = sorted(set(k[0] for k in data.keys()), key=str)
    cols = sorted(set(k[1] for k in data.keys()), key=str)

    matrix = np.full((len(rows), len(cols)), np.nan)
    for (rv, cv), vals in data.items():
        ri = rows.index(rv)
        ci = cols.index(cv)
        matrix[ri, ci] = np.mean(vals) if aggregate == "mean" else np.std(vals)

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.2), max(4, len(rows) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(r) for r in rows])
    ax.set_xlabel(col_key)
    ax.set_ylabel(row_key)

    # Annotate cells
    for ri in range(len(rows)):
        for ci in range(len(cols)):
            if not np.isnan(matrix[ri, ci]):
                ax.text(ci, ri, f"{matrix[ri, ci]:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if matrix[ri, ci] > np.nanmean(matrix) else "black")

    plt.colorbar(im, label=metric)
    ax.set_title(f"{metric} by {row_key} x {col_key}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------------------------------------------------
# 4. Pareto efficiency plots
# ------------------------------------------------------------------

def plot_pareto(
    grid_results: List[Dict],
    x_key: str = "params",
    y_metric: str = "val_iou",
    out_path: str = "figs/pareto.png",
) -> str:
    """Scatter plot of metric vs efficiency dimension with Pareto front."""
    _ensure_dir(out_path)

    points = []
    for r in grid_results:
        if "error" in r:
            continue
        x = r.get(x_key, r.get("config", {}).get(x_key, 0))
        y = r.get("final_metrics", {}).get(y_metric, 0)
        name = r.get("run_name", "")
        model = r.get("model", "")
        if x and y:
            points.append((x, y, name, model))

    if not points:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by model type
    models = sorted(set(p[3] for p in points))
    cmap = plt.cm.get_cmap("tab10", len(models))
    model_colors = {m: cmap(i) for i, m in enumerate(models)}

    for x, y, name, model in points:
        ax.scatter(x, y, c=[model_colors[model]], s=40, alpha=0.7,
                   edgecolors="black", linewidth=0.5)

    # Pareto front
    sorted_pts = sorted(points, key=lambda p: p[0])
    pareto = []
    best_y = -float("inf")
    for x, y, name, model in sorted_pts:
        if y > best_y:
            pareto.append((x, y))
            best_y = y
    if pareto:
        px, py = zip(*pareto)
        ax.step(list(px) + [max(p[0] for p in points)],
                list(py) + [py[-1]],
                where="post", color="red", linewidth=1.5, linestyle="--",
                label="Pareto front")

    # Legend
    for m in models:
        ax.scatter([], [], c=[model_colors[m]], label=m, s=40,
                   edgecolors="black", linewidth=0.5)
    ax.legend(fontsize=8)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_metric)
    ax.set_title(f"Efficiency: {y_metric} vs {x_key}")
    ax.grid(True, alpha=0.3)
    if max(p[0] for p in points) > 10000:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------------------------------------------------
# 5. Metric comparison bar charts
# ------------------------------------------------------------------

def plot_metric_bars(
    grid_results: List[Dict],
    metric: str = "val_iou",
    group_by: str = "model",
    out_path: str = "figs/metric_bars.png",
    show_std: bool = True,
) -> str:
    """Group results by a key and plot mean +/- std for a metric."""
    _ensure_dir(out_path)

    groups = {}
    for r in grid_results:
        if "error" in r:
            continue
        key = r.get(group_by, r.get("config", {}).get(group_by, "unknown"))
        val = r.get("final_metrics", {}).get(metric)
        if val is not None:
            groups.setdefault(str(key), []).append(val)

    if not groups:
        return ""

    names = sorted(groups.keys())
    means = [np.mean(groups[n]) for n in names]
    stds = [np.std(groups[n]) for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    bars = ax.bar(range(len(names)), means,
                  yerr=stds if show_std else None,
                  capsize=4, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by {group_by}")
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------------------------------------------------
# 6. SNR sweep
# ------------------------------------------------------------------

def run_snr_sweep(
    model: torch.nn.Module,
    snr_values: List[float],
    device: str = "cpu",
    n_samples: int = 256,
    out_path: str = "figs/snr_sweep.png",
) -> Tuple[str, Dict]:
    """Evaluate segmentation model at different SNR levels."""
    from ..data.synthetic_spectrogram import SyntheticSpectrogramDataset
    from .evaluate import evaluate_segmentation
    from torch.utils.data import DataLoader

    _ensure_dir(out_path)
    model.eval()
    results = {}

    for snr in snr_values:
        ds = SyntheticSpectrogramDataset(
            length=n_samples, h=64, w=64, snr_range=(snr, snr), seed=999
        )
        loader = DataLoader(ds, batch_size=32, num_workers=0)
        metrics = evaluate_segmentation(model, loader, device=device)
        results[snr] = metrics

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    snrs = sorted(results.keys())
    for ax, m, label in zip(axes, ["iou", "precision", "recall"],
                            ["IoU", "Precision", "Recall"]):
        vals = [results[s][m] for s in snrs]
        ax.plot(snrs, vals, "o-", linewidth=2)
        ax.set_xlabel("SNR")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs SNR")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness: Segmentation Metrics vs RFI SNR", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path, results


# ------------------------------------------------------------------
# 7. Architecture diagram (text-based for thesis)
# ------------------------------------------------------------------

def plot_fnn_architecture(
    stages: int = 3,
    use_fib: bool = True,
    base_ch: int = 16,
    out_path: str = "figs/fnn_architecture.png",
) -> str:
    """Visualize FNN channel schedule and fractal block structure."""
    from ..models.fnn import fib_sequence

    _ensure_dir(out_path)

    if use_fib:
        fibs = fib_sequence(stages + 1)
        widths = [max(8, base_ch * f) for f in fibs[:stages]]
        schedule_label = "Fibonacci"
    else:
        widths = [base_ch * (2 ** i) for i in range(stages)]
        schedule_label = "Standard (power-of-2)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Channel schedule comparison
    fib_widths_all = [max(8, base_ch * f) for f in fib_sequence(stages + 1)[:stages]]
    std_widths_all = [base_ch * (2 ** i) for i in range(stages)]

    x = range(1, stages + 1)
    ax1.bar([xi - 0.15 for xi in x], fib_widths_all, width=0.3, label="Fibonacci",
            color="steelblue", edgecolor="black")
    ax1.bar([xi + 0.15 for xi in x], std_widths_all, width=0.3, label="Standard",
            color="coral", edgecolor="black")
    for xi, fw, sw in zip(x, fib_widths_all, std_widths_all):
        ax1.text(xi - 0.15, fw + 1, str(fw), ha="center", fontsize=9)
        ax1.text(xi + 0.15, sw + 1, str(sw), ha="center", fontsize=9)
    ax1.set_xlabel("Stage")
    ax1.set_ylabel("Channels")
    ax1.set_title(f"Channel Schedule (base={base_ch})")
    ax1.legend()
    ax1.set_xticks(list(x))
    ax1.grid(True, alpha=0.3, axis="y")

    # Fractal block structure (depth diagram)
    def draw_fractal(ax, x, y, depth, width, height):
        if depth <= 1:
            ax.add_patch(plt.Rectangle((x, y), width, height,
                                       facecolor="lightblue",
                                       edgecolor="black", linewidth=1))
            ax.text(x + width / 2, y + height / 2, f"Conv\n(leaf)",
                    ha="center", va="center", fontsize=7)
            return
        half_w = width * 0.45
        gap = width * 0.1
        # Left child
        draw_fractal(ax, x, y, depth - 1, half_w, height * 0.8)
        # Right child
        draw_fractal(ax, x + half_w + gap, y, depth - 1, half_w, height * 0.8)
        # Merge
        merge_y = y + height * 0.85
        ax.add_patch(plt.Rectangle((x, merge_y), width, height * 0.15,
                                   facecolor="lightyellow",
                                   edgecolor="black", linewidth=1))
        ax.text(x + width / 2, merge_y + height * 0.075, "Merge (1x1)",
                ha="center", va="center", fontsize=7)

    draw_fractal(ax2, 0.1, 0.05, 3, 0.8, 0.85)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("FractalBlock (depth=3)")
    ax2.axis("off")

    plt.suptitle("FNN Architecture Components", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# ------------------------------------------------------------------
# 8. Comprehensive results table
# ------------------------------------------------------------------

def make_results_table(
    grid_results: List[Dict],
    metrics: List[str] = None,
    out_path: str = "figs/results_table.txt",
) -> str:
    """Create a formatted text table from grid results."""
    _ensure_dir(out_path)

    if metrics is None:
        metrics = ["val_iou", "val_dice", "val_f1", "val_precision", "val_recall"]

    rows = []
    for r in grid_results:
        if "error" in r:
            continue
        fm = r.get("final_metrics", {})
        row = {
            "model": r.get("run_name", ""),
            "params": r.get("params", 0),
        }
        for m in metrics:
            row[m] = fm.get(m, "")
        rows.append(row)

    if not rows:
        return ""

    # Format
    headers = ["model", "params"] + metrics
    lines = []
    header_line = "  ".join(f"{h:>14}" for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in sorted(rows, key=lambda r: -float(r.get("val_iou", 0) or 0)):
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:14.4f}")
            elif isinstance(v, int):
                vals.append(f"{v:14,}")
            else:
                vals.append(f"{str(v):>14}")
        lines.append("  ".join(vals))

    table_str = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(table_str)
    return out_path
