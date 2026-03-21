"""Generate Pareto efficiency plot from available results."""
import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = os.path.join(ROOT, "fnn_rfi", "figs")
RUNS = os.path.join(ROOT, "runs")


def main():
    # Collect best-seed results per config
    from fnn_rfi.scripts.aggregate_results import load_all_results, group_by_config
    results = load_all_results()
    summary = group_by_config(results)

    # Filter to segmentation models with IoU
    seg = [s for s in summary if s.get("val_iou_mean")]
    if len(seg) < 2:
        print("Need at least 2 configs for Pareto plot.")
        return

    # Also add efficiency data
    efficiency = {
        "fnn_seg fib s=3": {"params": 218465, "macs": 92.83},
        "fnn_seg fib+sep s=3": {"params": 81843, "macs": 23.56},
        "fnn_seg std s=3": {"params": 344817, "macs": 109.58},
        "fnn_seg sep s=3": {"params": 121985, "macs": 23.56},  # approx
        "unet d=3": {"params": 121033, "macs": 35.01},
        "resunet s=3": {"params": 1614833, "macs": 229.83},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Params vs IoU
    for s in seg:
        params = s["params"] if s["params"] > 0 else 200000  # estimate
        iou = s["val_iou_mean"]
        label = s["config_key"].split(" | ")[0]
        if "fib=True" in s["config_key"] and "sep=True" in s["config_key"]:
            label = "FNNSeg (fib+sep)"
            color = "green"
        elif "fib=True" in s["config_key"]:
            label = "FNNSeg (fib)"
            color = "blue"
        elif "sep=True" in s["config_key"]:
            label = "FNNSeg (sep)"
            color = "orange"
        elif s["config_key"].startswith("resunet"):
            label = "ResUNet"
            color = "red"
        elif s["config_key"].startswith("fnn_seg") and s["params"] == 0:
            label = "FNNSeg (smoke)"
            color = "gray"
        else:
            label = s["config_key"][:20]
            color = "purple"

        std = s.get("val_iou_std", 0)
        ax1.errorbar(params / 1000, iou, yerr=std, fmt="o", markersize=10,
                     color=color, label=label, capsize=4)

    ax1.set_xlabel("Parameters (K)")
    ax1.set_ylabel("Val IoU")
    ax1.set_title("Params vs IoU (Pareto)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: MACs vs IoU (from efficiency data)
    mac_data = []
    for s in seg:
        key = s["config_key"]
        iou = s["val_iou_mean"]
        if "fib=True" in key and "sep=True" in key:
            mac_data.append(("FNNSeg (fib+sep)", 23.56, iou, "green"))
        elif "fib=True" in key:
            mac_data.append(("FNNSeg (fib)", 92.83, iou, "blue"))
        elif key.startswith("resunet"):
            mac_data.append(("ResUNet", 229.83, iou, "red"))

    for label, macs, iou, color in mac_data:
        ax2.plot(macs, iou, "o", markersize=10, color=color, label=label)

    ax2.set_xlabel("MACs (M)")
    ax2.set_ylabel("Val IoU")
    ax2.set_title("MACs vs IoU (Pareto)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Efficiency vs Accuracy Trade-off", fontsize=13)
    plt.tight_layout()

    out = os.path.join(FIGS, "pareto_interim.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
