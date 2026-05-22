"""
Generate raster (prediction) plots from trained checkpoints + real dataset.

Run this ON THE GCP VM where the dataset .pkl files are stored:
    cd /path/to/project_root
    python3 fnn_rfi/scripts/generate_raster_plots.py \
        --pkl data/zenodo/HERA_28-03-2023_all.pkl \
        --ckpt fnn_rfi/results/hera/HERA_28-03-2023_all/fnn_seg_ch16_d3_fib_seed1/best_model.pt \
        --dataset hera \
        --out fnn_rfi/figs/thesis/fig21_raster_hera.png

    python3 fnn_rfi/scripts/generate_raster_plots.py \
        --pkl data/zenodo/LOFAR_Full_RFI_dataset.pkl \
        --ckpt fnn_rfi/results/hera/LOFAR_Full_RFI_dataset/fnn_seg_ch16_d3_fib_seed1/best_model.pt \
        --dataset lofar \
        --out fnn_rfi/figs/thesis/fig22_raster_lofar.png

Then copy the resulting PNGs to your Mac:
    gcloud compute scp fnn-rfi-gpu:~/your_repo/fnn_rfi/figs/thesis/fig21_raster_hera.png \
        /Users/ada/uel/fnn_rfi/figs/thesis/ --zone us-central1-a
    gcloud compute scp fnn-rfi-gpu:~/your_repo/fnn_rfi/figs/thesis/fig22_raster_lofar.png \
        /Users/ada/uel/fnn_rfi/figs/thesis/ --zone us-central1-a
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fnn_rfi.models.fnn_seg import FNNSeg


def load_checkpoint(ckpt_path, in_ch=2, device="cpu"):
    model = FNNSeg(in_ch=in_ch, base_ch=16, stages=3)
    state = torch.load(ckpt_path, map_location=device)
    # handle both raw state_dict and wrapped dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def load_samples(pkl_path, n_samples=4, seed=42):
    """Load a few random test-split samples from the Zenodo pkl."""
    rng = np.random.default_rng(seed)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Zenodo pkl structure: dict with "x" (N, H, W, C) and "y" (N, H, W, 1) arrays
    # or list of dicts, etc. Handle both.
    if isinstance(data, dict):
        x_all = np.array(data.get("x", data.get("X", data.get("data"))))
        y_all = np.array(data.get("y", data.get("Y", data.get("mask", data.get("labels")))))
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        x_all, y_all = np.array(data[0]), np.array(data[1])
    else:
        raise ValueError(f"Unexpected pkl structure: {type(data)}")

    # Use last 20% as test split (matches train_hera.py logic)
    n = len(x_all)
    test_start = int(n * 0.8)
    x_test = x_all[test_start:]
    y_test = y_all[test_start:]

    idxs = rng.choice(len(x_test), size=n_samples, replace=False)
    return x_test[idxs], y_test[idxs]


def predict_patch(model, x_patch, device="cpu"):
    """Run inference on a single patch. x_patch: (H, W, C) numpy array."""
    if x_patch.ndim == 2:
        x_patch = x_patch[..., np.newaxis]
    # normalise per-channel
    xn = x_patch.astype(np.float32)
    for c in range(xn.shape[-1]):
        mn, mx = xn[..., c].min(), xn[..., c].max()
        if mx > mn:
            xn[..., c] = (xn[..., c] - mn) / (mx - mn)
    t = torch.from_numpy(xn.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(t)
        prob = torch.sigmoid(logit).squeeze().cpu().numpy()
    return prob


def make_raster_figure(x_samples, y_samples, model, device, dataset_type, out_path):
    """
    Create the raster plot: N rows x 4 columns
      Col 0: Input amplitude (channel 0)
      Col 1: Input phase (channel 1, HERA only) or blank for LOFAR
      Col 2: Ground-truth mask
      Col 3: Predicted mask (probability)
      Col 4: Error map (|GT - pred|)
    """
    n = len(x_samples)
    is_hera = (dataset_type == "hera")
    ncols = 5 if is_hera else 4
    col_titles = (["Amplitude", "Phase", "GT Mask", "Predicted", "Error Map"]
                  if is_hera else
                  ["Amplitude", "GT Mask", "Predicted", "Error Map"])

    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.2, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    rfi_cmap = ListedColormap(["#F0F9FF", "#DC2626"])   # clean / RFI

    for row, (x, y) in enumerate(zip(x_samples, y_samples)):
        # x: (H, W, C), y: (H, W) or (H, W, 1)
        if y.ndim == 3:
            y = y.squeeze(-1)
        gt = (y > 0.5).astype(np.float32)

        # run inference
        pred_prob = predict_patch(model, x, device=device)
        pred_bin  = (pred_prob > 0.5).astype(np.float32)
        error     = np.abs(gt - pred_bin)

        col = 0
        # Amplitude
        amp = x[..., 0] if x.ndim == 3 else x
        axes[row, col].imshow(amp, cmap="viridis", aspect="auto", origin="lower")
        axes[row, col].set_title(col_titles[0] if row == 0 else "", fontsize=9, fontweight="bold")
        axes[row, col].axis("off")
        col += 1

        if is_hera and x.shape[-1] > 1:
            phase = x[..., 1]
            axes[row, col].imshow(phase, cmap="RdBu_r", aspect="auto", origin="lower")
            axes[row, col].set_title(col_titles[1] if row == 0 else "", fontsize=9, fontweight="bold")
            axes[row, col].axis("off")
            col += 1

        # GT mask
        axes[row, col].imshow(gt, cmap=rfi_cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
        axes[row, col].set_title(col_titles[col] if row == 0 else "", fontsize=9, fontweight="bold")
        axes[row, col].axis("off")
        col += 1

        # Predicted probability (heatmap)
        im = axes[row, col].imshow(pred_prob, cmap="hot_r", vmin=0, vmax=1,
                                    aspect="auto", origin="lower")
        axes[row, col].set_title(col_titles[col] if row == 0 else "", fontsize=9, fontweight="bold")
        axes[row, col].axis("off")
        col += 1

        # Error map
        axes[row, col].imshow(error, cmap="Reds", vmin=0, vmax=1,
                               aspect="auto", origin="lower")
        # Show F1 for this sample
        tp = ((pred_bin == 1) & (gt == 1)).sum()
        fp = ((pred_bin == 1) & (gt == 0)).sum()
        fn = ((pred_bin == 0) & (gt == 1)).sum()
        f1 = 2*tp / (2*tp + fp + fn + 1e-8)
        axes[row, col].set_title(col_titles[col] if row == 0 else "", fontsize=9, fontweight="bold")
        axes[row, col].set_xlabel(f"F1={f1:.4f}", fontsize=8)
        axes[row, col].axis("off")

        # Row label
        axes[row, 0].set_ylabel(f"Sample {row+1}", fontsize=9, rotation=90, labelpad=4)

    ds_name = "HERA (pixel-perfect)" if is_hera else "LOFAR"
    fig.suptitle(f"FNNSeg Predictions on {ds_name} Test Set\n"
                 f"Red = RFI  |  Hot = prediction confidence  |  Error = |GT - pred|",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",     required=True, help="Path to Zenodo .pkl file")
    parser.add_argument("--ckpt",    required=True, help="Path to best_model.pt")
    parser.add_argument("--dataset", default="hera", choices=["hera", "lofar"])
    parser.add_argument("--out",     default="raster_predictions.png")
    parser.add_argument("--n",       type=int, default=4, help="Number of samples to show")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    in_ch = 2 if args.dataset == "hera" else 1
    print(f"Loading model from {args.ckpt} ...")
    model = load_checkpoint(args.ckpt, in_ch=in_ch, device=args.device)

    print(f"Loading {args.n} test samples from {args.pkl} ...")
    x_samples, y_samples = load_samples(args.pkl, n_samples=args.n)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    make_raster_figure(x_samples, y_samples, model, args.device, args.dataset, args.out)


if __name__ == "__main__":
    main()
