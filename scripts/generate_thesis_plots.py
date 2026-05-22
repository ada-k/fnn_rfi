"""
generate_thesis_plots.py
Generates ALL figures needed for the thesis from results JSON files.
Run from the project root: python3 fnn_rfi/scripts/generate_thesis_plots.py
Outputs → fnn_rfi/figs/thesis/
"""
import json, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ──────────────────────────────────────────────────────────────
OUTDIR = "fnn_rfi/figs/thesis"
os.makedirs(OUTDIR, exist_ok=True)

PALETTE = {
    "fnn_seg": "#2563EB",   # blue
    "unet":    "#16A34A",   # green
    "resunet": "#DC2626",   # red
}
MODEL_LABELS = {
    "fnn_seg": "FNNSeg",
    "unet":    "U-Net",
    "resunet": "ResU-Net",
}
DS_LABELS = {
    "HERA_28-03-2023_all":  "HERA (pixel-perfect)",
    "HERA_AOF_20-07-2023_all": "HERA (AOFlagger)",
    "LOFAR_Full_RFI_dataset":  "LOFAR",
}
DS_SHORT = {
    "HERA_28-03-2023_all":  "HERA-PP",
    "HERA_AOF_20-07-2023_all": "HERA-AOF",
    "LOFAR_Full_RFI_dataset":  "LOFAR",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ──────────────────────────────────────────────────────────────
# LOAD ALL RESULTS
# ──────────────────────────────────────────────────────────────
def load_all():
    results = []
    for f in sorted(glob.glob("fnn_rfi/results/hera/**/**/results.json", recursive=True)):
        with open(f) as fp:
            d = json.load(fp)
        parts = f.replace("\\","/").split("/")
        dataset = parts[3]
        d["_dataset"] = dataset
        d["_path"] = f
        results.append(d)
    return results

ALL = load_all()
DATASETS = ["HERA_28-03-2023_all", "HERA_AOF_20-07-2023_all", "LOFAR_Full_RFI_dataset"]
MODELS   = ["fnn_seg", "unet", "resunet"]

def get(dataset, model, seed):
    for r in ALL:
        if r["_dataset"]==dataset and r["model"]==model and r["seed"]==seed:
            return r
    return None

def acc_key(d):
    return d.get("test_accuracy", d.get("test_acc", np.nan))

def mean_std(vals):
    v = [x for x in vals if x == x]  # drop NaN
    return np.mean(v), np.std(v)

# ──────────────────────────────────────────────────────────────
# FIG 1 — F1 BAR CHART (3 datasets, 3 models, mean±std over seeds)
# Full 0–1 scale with 0.1 ticks
# ──────────────────────────────────────────────────────────────
def fig_f1_bars():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    seeds = [0, 1, 2]
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(MODELS))
        width = 0.5
        for i, m in enumerate(MODELS):
            f1s = [get(ds, m, s)["test_f1"] for s in seeds if get(ds, m, s)]
            mu, sd = mean_std(f1s)
            ax.bar(i, mu, width, color=PALETTE[m], alpha=0.85,
                   label=MODEL_LABELS[m], zorder=3)
            ax.errorbar(i, mu, yerr=sd, fmt="none", color="black",
                        capsize=4, linewidth=1.5, zorder=4)
            ax.text(i, mu + sd + 0.012, f"{mu:.4f}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=12, ha="right")
        ax.set_title(DS_LABELS[ds], fontweight="bold")
        ax.set_ylabel("Test F1 Score")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.1, 0.1)])
    handles = [mpatches.Patch(color=PALETTE[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Test F1 Score (mean ± std over 3 seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTDIR}/fig1_f1_bars.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 2 — AUC BAR CHART (0–1 scale, 0.1 ticks)
# ──────────────────────────────────────────────────────────────
def fig_auc_bars():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    seeds = [0, 1, 2]
    for ax, ds in zip(axes, DATASETS):
        for i, m in enumerate(MODELS):
            vals = [get(ds, m, s)["test_auc"] for s in seeds if get(ds, m, s)]
            mu, sd = mean_std(vals)
            ax.bar(i, mu, 0.5, color=PALETTE[m], alpha=0.85, zorder=3)
            ax.errorbar(i, mu, yerr=sd, fmt="none", color="black",
                        capsize=4, linewidth=1.5, zorder=4)
            ax.text(i, mu + sd + 0.012, f"{mu:.4f}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=12, ha="right")
        ax.set_title(DS_LABELS[ds], fontweight="bold")
        ax.set_ylabel("Test AUC (ROC)")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
    handles = [mpatches.Patch(color=PALETTE[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Test AUC-ROC (mean ± std over 3 seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTDIR}/fig2_auc_bars.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 3 — ACCURACY BAR CHART (0–1 scale)
# ──────────────────────────────────────────────────────────────
def fig_acc_bars():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    seeds = [0, 1, 2]
    for ax, ds in zip(axes, DATASETS):
        for i, m in enumerate(MODELS):
            vals = [acc_key(get(ds, m, s)) for s in seeds if get(ds, m, s)]
            mu, sd = mean_std(vals)
            ax.bar(i, mu, 0.5, color=PALETTE[m], alpha=0.85, zorder=3)
            ax.errorbar(i, mu, yerr=sd, fmt="none", color="black",
                        capsize=4, linewidth=1.5, zorder=4)
            ax.text(i, mu + sd + 0.012, f"{mu:.5f}", ha="center",
                    va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=12, ha="right")
        ax.set_title(DS_LABELS[ds], fontweight="bold")
        ax.set_ylabel("Test Pixel Accuracy")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
    handles = [mpatches.Patch(color=PALETTE[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Test Pixel Accuracy (mean ± std over 3 seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTDIR}/fig3_acc_bars.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 4 — LEARNING CURVES HERA-PP
# Standardised axes across all subplots for easy comparison
# ──────────────────────────────────────────────────────────────
def fig_learning_curves():
    ds = "HERA_28-03-2023_all"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    seeds = [0, 1, 2]
    # Pre-compute global y-ranges for standardised axes
    all_train_loss, all_val_loss, all_val_f1 = [], [], []
    for m in MODELS:
        for s in seeds:
            r = get(ds, m, s)
            if r and "history" in r:
                all_train_loss += [h["train"]["loss"] for h in r["history"]]
                all_val_loss   += [h["val"]["loss"] for h in r["history"]]
                all_val_f1     += [h["val"]["f1"] for h in r["history"]]
    loss_max = max(max(all_train_loss), max(all_val_loss)) * 1.05
    f1_min = min(all_val_f1) - 0.01

    for col, seed in enumerate(seeds):
        for row, m in enumerate(MODELS):
            ax = axes[row, col]
            r = get(ds, m, seed)
            if r is None or "history" not in r:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            hist = r["history"]
            epochs   = [h["epoch"] for h in hist]
            t_loss   = [h["train"]["loss"] for h in hist]
            v_loss   = [h["val"]["loss"] for h in hist]
            v_f1     = [h["val"]["f1"] for h in hist]
            ax2 = ax.twinx()
            ax.plot(epochs, t_loss, color="#F97316", linewidth=1.8, label="Train loss")
            ax.plot(epochs, v_loss, color="#6366F1", linewidth=1.8, linestyle="--", label="Val loss")
            ax2.plot(epochs, v_f1,  color="#16A34A", linewidth=1.8, linestyle=":", label="Val F1")
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Loss", color="#374151", fontsize=8)
            ax2.set_ylabel("Val F1", color="#16A34A", fontsize=8)
            ax2.tick_params(axis="y", colors="#16A34A", labelsize=7)
            ax2.set_ylim(f1_min, 1.005)
            ax.set_ylim(0, loss_max)
            ax.set_title(f"{MODEL_LABELS[m]}  |  seed={seed}", fontsize=9, fontweight="bold")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                l1, lb1 = ax.get_legend_handles_labels()
                l2, lb2 = ax2.get_legend_handles_labels()
                ax.legend(l1+l2, lb1+lb2, fontsize=7, loc="upper right")
            # Annotate notable val loss spikes
            spike_ep = max(range(len(v_loss)), key=lambda i: v_loss[i])
            if v_loss[spike_ep] > np.mean(v_loss) * 1.5 and spike_ep > 0:
                ax.annotate(f"spike\nep{epochs[spike_ep]}", xy=(epochs[spike_ep], v_loss[spike_ep]),
                            xytext=(epochs[spike_ep]+0.5, v_loss[spike_ep]*0.85),
                            fontsize=6, color="#6366F1",
                            arrowprops=dict(arrowstyle="->", color="#6366F1", lw=0.8))
    fig.suptitle(f"Training and Validation Curves — {DS_LABELS[ds]}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = f"{OUTDIR}/fig4_learning_curves_hera.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 5 — LEARNING CURVES HERA_AOF (standardised axes)
# ──────────────────────────────────────────────────────────────
def fig_learning_curves_aof():
    ds = "HERA_AOF_20-07-2023_all"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    seeds = [0, 1, 2]
    all_tl, all_vl, all_f1 = [], [], []
    for m in MODELS:
        for s in seeds:
            r = get(ds, m, s)
            if r and "history" in r:
                all_tl += [h["train"]["loss"] for h in r["history"]]
                all_vl += [h["val"]["loss"] for h in r["history"]]
                all_f1 += [h["val"]["f1"] for h in r["history"]]
    loss_max = max(max(all_tl), max(all_vl)) * 1.05
    f1_min = min(all_f1) - 0.01

    for col, seed in enumerate(seeds):
        for row, m in enumerate(MODELS):
            ax = axes[row, col]
            r = get(ds, m, seed)
            if r is None or "history" not in r:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            hist = r["history"]
            epochs = [h["epoch"] for h in hist]
            ax2 = ax.twinx()
            ax.plot(epochs, [h["train"]["loss"] for h in hist], color="#F97316", linewidth=1.8, label="Train loss")
            ax.plot(epochs, [h["val"]["loss"]   for h in hist], color="#6366F1", linewidth=1.8, linestyle="--", label="Val loss")
            ax2.plot(epochs,[h["val"]["f1"]     for h in hist], color="#16A34A", linewidth=1.8, linestyle=":", label="Val F1")
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Loss", fontsize=8)
            ax2.set_ylabel("Val F1", color="#16A34A", fontsize=8)
            ax2.tick_params(axis="y", colors="#16A34A", labelsize=7)
            ax.set_ylim(0, loss_max)
            ax2.set_ylim(f1_min, 1.005)
            ax.set_title(f"{MODEL_LABELS[m]}  |  seed={seed}", fontsize=9, fontweight="bold")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Training and Validation Curves — {DS_LABELS[ds]}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = f"{OUTDIR}/fig5_learning_curves_aof.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 6 — LEARNING CURVES LOFAR (standardised axes)
# ──────────────────────────────────────────────────────────────
def fig_learning_curves_lofar():
    ds = "LOFAR_Full_RFI_dataset"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    seeds = [0, 1, 2]
    all_tl, all_vl, all_f1 = [], [], []
    for m in MODELS:
        for s in seeds:
            r = get(ds, m, s)
            if r and "history" in r:
                all_tl += [h["train"]["loss"] for h in r["history"]]
                all_vl += [h["val"]["loss"] for h in r["history"]]
                all_f1 += [h["val"]["f1"] for h in r["history"]]
    loss_max = max(max(all_tl), max(all_vl)) * 1.05
    f1_min = max(0, min(all_f1) - 0.02)

    for col, seed in enumerate(seeds):
        for row, m in enumerate(MODELS):
            ax = axes[row, col]
            r = get(ds, m, seed)
            if r is None or "history" not in r:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            hist = r["history"]
            epochs = [h["epoch"] for h in hist]
            ax2 = ax.twinx()
            ax.plot(epochs, [h["train"]["loss"] for h in hist], color="#F97316", linewidth=1.8, label="Train loss")
            ax.plot(epochs, [h["val"]["loss"]   for h in hist], color="#6366F1", linewidth=1.8, linestyle="--", label="Val loss")
            ax2.plot(epochs,[h["val"]["f1"]     for h in hist], color="#16A34A", linewidth=1.8, linestyle=":", label="Val F1")
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Loss", fontsize=8)
            ax2.set_ylabel("Val F1", color="#16A34A", fontsize=8)
            ax2.tick_params(axis="y", colors="#16A34A", labelsize=7)
            ax.set_ylim(0, loss_max)
            ax2.set_ylim(f1_min, 1.005)
            ax.set_title(f"{MODEL_LABELS[m]}  |  seed={seed}", fontsize=9, fontweight="bold")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.tick_params(labelsize=7)
    # Shared legend at bottom
    solid = plt.Line2D([0],[0], color="#F97316", linewidth=1.8, label="Train loss")
    dashed = plt.Line2D([0],[0], color="#6366F1", linewidth=1.8, linestyle="--", label="Val loss")
    dotted = plt.Line2D([0],[0], color="#16A34A", linewidth=1.8, linestyle=":", label="Val F1 (right axis)")
    fig.legend(handles=[solid, dashed, dotted], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle(f"Training and Validation Curves — {DS_LABELS[ds]}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = f"{OUTDIR}/fig6_learning_curves_lofar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 7 — PARETO EFFICIENCY: Params vs F1 (including du Toit)
# ──────────────────────────────────────────────────────────────
def fig_pareto():
    ds = "HERA_28-03-2023_all"
    seeds = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(9, 6))
    # Our models
    for m in MODELS:
        f1s = [get(ds, m, s)["test_f1"] for s in seeds if get(ds, m, s)]
        params = get(ds, m, 0)["params"]
        mu, sd = mean_std(f1s)
        ax.errorbar(params, mu, yerr=sd, fmt="o", color=PALETTE[m],
                    markersize=10, capsize=5, label=MODEL_LABELS[m],
                    zorder=5, linewidth=2)
        ax.annotate(MODEL_LABELS[m], (params, mu),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    # du Toit models
    dutoit = {
        "UNET":          (291633,  0.9869),
        "ASPP_UNET":     (291633,  0.9863),
        "DSC_DUAL_RESUNET": (370494, 0.9859),
        "DSC_MONO_RESUNET": (185226, 0.9826),
        "RFI_NET":       (805473,  0.9936),
        "RNET":          (32705,   0.9895),
        "RNET5":         (19809,   0.9851),
    }
    for name, (params, f1) in dutoit.items():
        ax.scatter(params, f1, marker="^", s=80, color="#6B7280",
                   zorder=3, alpha=0.75)
        ax.annotate(name, (params, f1),
                    textcoords="offset points", xytext=(5, -10),
                    fontsize=7.5, color="#374151")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Test F1 Score")
    ax.set_xscale("log")
    ax.set_title("Parameter Efficiency: Test F1 vs Model Size\n(● = this work, ▲ = du Toit et al., 2023)",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0.96, 1.002)
    # Shading for "better" quadrant
    ax.axvline(x=218609, color=PALETTE["fnn_seg"], linestyle=":", alpha=0.4)
    fig.tight_layout()
    path = f"{OUTDIR}/fig7_pareto_efficiency.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 8 — COMPARISON VS du Toit (F1 grouped bar chart)
# ──────────────────────────────────────────────────────────────
def fig_comparison_dutoit():
    fig, ax = plt.subplots(figsize=(13, 6))
    ds = "HERA_28-03-2023_all"
    seeds = [0, 1, 2]
    our_models = {
        "FNNSeg\n(Ours)": ("fnn_seg", "#2563EB"),
        "U-Net\n(Ours)":  ("unet",    "#16A34A"),
        "ResU-Net\n(Ours)": ("resunet","#DC2626"),
    }
    dutoit_models = {
        "RNET5":            (0.9851, "#9CA3AF"),
        "DSC_MONO\nRESUNET": (0.9826, "#9CA3AF"),
        "RNET":             (0.9895, "#9CA3AF"),
        "ASPP\nUNET":       (0.9863, "#9CA3AF"),
        "UNET\n(du Toit)":  (0.9869, "#9CA3AF"),
        "DSC_DUAL\nRESUNET":(0.9859, "#9CA3AF"),
        "RFI_NET":          (0.9936, "#9CA3AF"),
    }
    labels, f1s, colors, errs = [], [], [], []
    for label, (mkey, col) in our_models.items():
        vals = [get(ds, mkey, s)["test_f1"] for s in seeds if get(ds, mkey, s)]
        mu, sd = mean_std(vals)
        labels.append(label); f1s.append(mu); colors.append(col); errs.append(sd)
    for label, (f1, col) in dutoit_models.items():
        labels.append(label); f1s.append(f1); colors.append(col); errs.append(0)
    x = np.arange(len(labels))
    bars = ax.bar(x, f1s, 0.6, color=colors, alpha=0.85, zorder=3)
    ax.errorbar(x[:3], f1s[:3], yerr=errs[:3], fmt="none", color="black",
                capsize=4, linewidth=1.5, zorder=4)
    for xi, f1 in zip(x, f1s):
        ax.text(xi, f1 + 0.0005, f"{f1:.4f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Test F1 Score")
    ax.set_title("F1 Comparison: This Work vs du Toit et al. (2023) — HERA Pixel-Perfect Labels",
                 fontweight="bold", fontsize=11)
    ax.set_ylim(0.975, 1.003)
    ours_patch = mpatches.Patch(color="#2563EB", label="This work (mean ± std, n=3 seeds)")
    ref_patch  = mpatches.Patch(color="#9CA3AF", label="du Toit et al. (2023) — best run")
    ax.legend(handles=[ours_patch, ref_patch], loc="lower left")
    fig.tight_layout()
    path = f"{OUTDIR}/fig8_comparison_dutoit.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 9 — CROSS-DATASET F1 HEATMAP (model × dataset)
# ──────────────────────────────────────────────────────────────
def fig_cross_dataset_heatmap():
    seeds = [0, 1, 2]
    data = np.zeros((3, 3))
    for i, m in enumerate(MODELS):
        for j, ds in enumerate(DATASETS):
            vals = [get(ds, m, s)["test_f1"] for s in seeds if get(ds, m, s)]
            data[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([DS_SHORT[d] for d in DATASETS], fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.4f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if data[i,j] > 0.85 else "black")
    plt.colorbar(im, ax=ax, label="Mean Test F1 (n=3 seeds)")
    ax.set_title("Cross-Dataset Performance: Mean Test F1 Score", fontweight="bold")
    fig.tight_layout()
    path = f"{OUTDIR}/fig9_cross_dataset_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 10 — SEED VARIANCE (box plots over seeds per model per dataset)
# ──────────────────────────────────────────────────────────────
def fig_seed_variance():
    seeds = [0, 1, 2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, ds in zip(axes, DATASETS):
        data_f1  = [[get(ds, m, s)["test_f1"] for s in seeds if get(ds, m, s)] for m in MODELS]
        bp = ax.boxplot(data_f1, patch_artist=True, widths=0.4,
                        medianprops=dict(color="black", linewidth=2))
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(PALETTE[m])
            patch.set_alpha(0.75)
        for i, (vals, m) in enumerate(zip(data_f1, MODELS)):
            ax.scatter([i+1]*len(vals), vals, color="black", s=30, zorder=5)
        ax.set_xticks(range(1, len(MODELS)+1))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=12, ha="right")
        ax.set_title(DS_LABELS[ds], fontweight="bold")
        ax.set_ylabel("Test F1 Score")
    handles = [mpatches.Patch(color=PALETTE[m], alpha=0.75, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Seed Variance: Distribution of Test F1 Across 3 Random Seeds",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTDIR}/fig10_seed_variance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 11 — RADAR CHART (F1, AUC, Acc normalised) per dataset
# ──────────────────────────────────────────────────────────────
def fig_radar():
    from matplotlib.patches import FancyArrowPatch
    categories = ["F1", "AUC", "Accuracy"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    seeds = [0,1,2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                              subplot_kw=dict(polar=True))
    for ax, ds in zip(axes, DATASETS):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        for m in MODELS:
            vals_f1  = np.mean([get(ds, m, s)["test_f1"]  for s in seeds if get(ds,m,s)])
            vals_auc = np.mean([get(ds, m, s)["test_auc"] for s in seeds if get(ds,m,s)])
            vals_acc = np.mean([acc_key(get(ds, m, s))    for s in seeds if get(ds,m,s)])
            # Normalise to 0–1 range within plottable [0.6, 1.0]
            lo, hi = 0.6, 1.0
            norm = lambda v: (v - lo) / (hi - lo)
            values = [norm(vals_f1), norm(vals_auc), norm(vals_acc)]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, color=PALETTE[m], label=MODEL_LABELS[m])
            ax.fill(angles, values, alpha=0.1, color=PALETTE[m])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.7", "0.8", "0.9", "1.0"], fontsize=7)
        ax.set_title(DS_SHORT[ds], fontsize=10, fontweight="bold", pad=12)
    handles = [plt.Line2D([0],[0], color=PALETTE[m], linewidth=2,
                          label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Multi-Metric Radar Chart (F1 / AUC / Accuracy)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    path = f"{OUTDIR}/fig11_radar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 12 — CONVERGENCE SPEED (val F1 vs epoch, best seed per model per dataset)
# ──────────────────────────────────────────────────────────────
def fig_convergence():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ds in zip(axes, DATASETS):
        for m in MODELS:
            # use seed 0 (or best available)
            for s in [0, 1, 2]:
                r = get(ds, m, s)
                if r and "history" in r:
                    break
            if not r or "history" not in r:
                continue
            epochs = [h["epoch"] for h in r["history"]]
            val_f1 = [h["val"]["f1"] for h in r["history"]]
            ax.plot(epochs, val_f1, color=PALETTE[m], linewidth=2,
                    label=MODEL_LABELS[m], marker="o", markersize=4, alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation F1")
        ax.set_title(DS_LABELS[ds], fontweight="bold")
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    fig.suptitle("Convergence: Validation F1 vs Training Epoch (seed=0)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = f"{OUTDIR}/fig12_convergence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 13 — TRAIN LOSS CONVERGENCE (all seeds overlaid)
# ──────────────────────────────────────────────────────────────
def fig_loss_all_seeds():
    ds = "HERA_28-03-2023_all"
    seeds = [0,1,2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, MODELS):
        for s in seeds:
            r = get(ds, m, s)
            if not r or "history" not in r: continue
            eps = [h["epoch"] for h in r["history"]]
            tl  = [h["train"]["loss"] for h in r["history"]]
            vl  = [h["val"]["loss"] for h in r["history"]]
            ax.plot(eps, tl, color=PALETTE[m], linewidth=1.5,
                    alpha=0.8, label=f"Train s{s}")
            ax.plot(eps, vl, color=PALETTE[m], linewidth=1.5,
                    alpha=0.4, linestyle="--", label=f"Val s{s}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("DiceBCE Loss")
        ax.set_title(MODEL_LABELS[m], fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    fig.suptitle(f"Train/Val Loss Over All Seeds — {DS_LABELS[ds]}",
                 fontsize=12, fontweight="bold")
    # shared legend
    solid = plt.Line2D([0],[0], color="gray", linewidth=1.5, label="Train loss")
    dashed = plt.Line2D([0],[0], color="gray", linewidth=1.5,
                        linestyle="--", label="Val loss", alpha=0.4)
    fig.legend(handles=[solid, dashed], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTDIR}/fig13_loss_all_seeds.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 14 — DATASET DIFFICULTY COMPARISON (F1 across 3 datasets for each model)
# ──────────────────────────────────────────────────────────────
def fig_dataset_difficulty():
    seeds = [0,1,2]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DATASETS))
    width = 0.22
    offsets = [-width, 0, width]
    for i, m in enumerate(MODELS):
        mus, sds = [], []
        for ds in DATASETS:
            vals = [get(ds, m, s)["test_f1"] for s in seeds if get(ds, m, s)]
            mu, sd = mean_std(vals)
            mus.append(mu); sds.append(sd)
        ax.bar(x + offsets[i], mus, width, color=PALETTE[m],
               alpha=0.85, label=MODEL_LABELS[m], zorder=3)
        ax.errorbar(x + offsets[i], mus, yerr=sds, fmt="none",
                    color="black", capsize=3, linewidth=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=9)
    ax.set_ylabel("Mean Test F1 Score (n=3 seeds)")
    ax.set_title("Performance Across Datasets: Effect of Label Type and Instrument",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0.5, 1.005)
    # annotate difficulty arrows
    ax.annotate("", xy=(2, 0.65), xytext=(0, 0.99),
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.5))
    ax.text(1.05, 0.80, "Increasing\ndifficulty →", fontsize=8,
            color="#6B7280", ha="center", style="italic")
    fig.tight_layout()
    path = f"{OUTDIR}/fig14_dataset_difficulty.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 15 — SUMMARY TABLE FIGURE (text-based results table)
# ──────────────────────────────────────────────────────────────
def fig_summary_table():
    seeds = [0,1,2]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")
    col_labels = ["Dataset", "Model", "Params", "Acc (mean±std)", "F1 (mean±std)", "AUC (mean±std)"]
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            vals_acc = [acc_key(get(ds,m,s)) for s in seeds if get(ds,m,s)]
            vals_f1  = [get(ds,m,s)["test_f1"] for s in seeds if get(ds,m,s)]
            vals_auc = [get(ds,m,s)["test_auc"] for s in seeds if get(ds,m,s)]
            params   = get(ds, m, 0)["params"]
            mu_acc, sd_acc = mean_std(vals_acc)
            mu_f1,  sd_f1  = mean_std(vals_f1)
            mu_auc, sd_auc = mean_std(vals_auc)
            rows.append([
                DS_SHORT[ds], MODEL_LABELS[m], f"{params:,}",
                f"{mu_acc:.6f} ± {sd_acc:.6f}",
                f"{mu_f1:.4f} ± {sd_f1:.4f}",
                f"{mu_auc:.4f} ± {sd_auc:.4f}",
            ])
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    # Colour header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1E3A5F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row colours
    for i, row in enumerate(rows):
        bg = "#EFF6FF" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(bg)
        # Highlight FNNSeg
        if row[1] == "FNNSeg (Ours)":
            for j in range(len(col_labels)):
                table[i+1, j].set_facecolor("#DBEAFE")
    ax.set_title("Summary Results Table (mean ± std, n=3 seeds)",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    path = f"{OUTDIR}/fig15_summary_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 16 — BEST EPOCH DISTRIBUTION (early stopping effect)
# ──────────────────────────────────────────────────────────────
def fig_best_epoch():
    seeds = [0,1,2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, ds in zip(axes, DATASETS):
        for m in MODELS:
            best_epochs = []
            for s in seeds:
                r = get(ds, m, s)
                if r and "history" in r:
                    # best epoch = epoch with max val F1
                    best_ep = max(r["history"], key=lambda h: h["val"]["f1"])["epoch"]
                    best_epochs.append(best_ep)
            if best_epochs:
                x_pos = MODELS.index(m)
                ax.bar(x_pos, np.mean(best_epochs), 0.5, color=PALETTE[m], alpha=0.8, zorder=3)
                ax.errorbar(x_pos, np.mean(best_epochs), yerr=np.std(best_epochs),
                            fmt="none", color="black", capsize=4, zorder=4)
                ax.text(x_pos, np.mean(best_epochs) + np.std(best_epochs) + 0.3,
                        f"{np.mean(best_epochs):.1f}", ha="center", fontsize=9,
                        fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=12, ha="right")
        ax.set_ylabel("Best Epoch (val F1 peak)")
        ax.set_title(DS_LABELS[ds], fontweight="bold")
        ax.set_ylim(0, 33)
        ax.axhline(y=30, color="red", linestyle=":", alpha=0.5, label="Max epochs")
    handles = [mpatches.Patch(color=PALETTE[m], label=MODEL_LABELS[m]) for m in MODELS]
    handles.append(plt.Line2D([0],[0], color="red", linestyle=":", label="Max epochs (30)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Best Validation Epoch Distribution (with Early Stopping, patience=7)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{OUTDIR}/fig16_best_epoch.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# FIG 17 — ARCHITECTURE COMPARISON (params vs F1 scatter annotated)
# ──────────────────────────────────────────────────────────────
def fig_arch_comparison():
    ds = "HERA_28-03-2023_all"
    seeds = [0,1,2]
    fig, ax = plt.subplots(figsize=(10, 6))
    # Our models over 3 datasets
    markers = {"HERA_28-03-2023_all": "o", "HERA_AOF_20-07-2023_all": "s",
               "LOFAR_Full_RFI_dataset": "D"}
    for ds2 in DATASETS:
        for m in MODELS:
            vals = [get(ds2, m, s)["test_f1"] for s in seeds if get(ds2, m, s)]
            params = get(ds2, m, 0)["params"]
            mu, sd = mean_std(vals)
            ax.scatter(params, mu, marker=markers[ds2], s=100,
                       color=PALETTE[m], zorder=5, alpha=0.9,
                       label=f"{MODEL_LABELS[m]} ({DS_SHORT[ds2]})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Mean Test F1 Score (n=3)")
    ax.set_title("Parameter Efficiency Across All Datasets\n(○=HERA-PP, □=HERA-AOF, ◇=LOFAR)",
                 fontweight="bold")
    from matplotlib.lines import Line2D
    handles = []
    for m in MODELS:
        handles.append(Line2D([0],[0], marker="o", color=PALETTE[m],
                               linewidth=0, markersize=8, label=MODEL_LABELS[m]))
    for ds2, mk in markers.items():
        handles.append(Line2D([0],[0], marker=mk, color="gray",
                               linewidth=0, markersize=8, label=DS_SHORT[ds2]))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    path = f"{OUTDIR}/fig17_arch_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ──────────────────────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating thesis figures → {OUTDIR}/\n")
    fig_f1_bars()
    fig_auc_bars()
    fig_acc_bars()
    fig_learning_curves()
    fig_learning_curves_aof()
    fig_learning_curves_lofar()
    fig_pareto()
    fig_comparison_dutoit()
    fig_cross_dataset_heatmap()
    fig_seed_variance()
    fig_radar()
    fig_convergence()
    fig_loss_all_seeds()
    fig_dataset_difficulty()
    fig_summary_table()
    fig_best_epoch()
    fig_arch_comparison()
    print(f"\nDone! {len(os.listdir(OUTDIR))} files in {OUTDIR}/")
