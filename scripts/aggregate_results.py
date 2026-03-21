"""Aggregate experiment results across seeds, generate summary tables and
update the research report. Run at any point to get a snapshot of current
progress.

Usage:
    python3 -m fnn_rfi.scripts.aggregate_results
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RUNS_DIR = os.path.join(ROOT, "runs")
FIGS_DIR = os.path.join(ROOT, "fnn_rfi", "figs")


def load_all_results():
    """Load all results.json and group by config (model + hyperparams)."""
    results = []
    for d in sorted(os.listdir(RUNS_DIR)):
        rpath = os.path.join(RUNS_DIR, d, "results.json")
        if not os.path.isfile(rpath):
            continue
        with open(rpath) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        hist = data.get("history", [])
        if not hist:
            continue
        final = hist[-1]
        # Build a config key (exclude seed, epochs, etc.)
        key_parts = []
        model = cfg.get("model", "unknown")
        key_parts.append(model)
        for k in ["use_fib", "use_separable", "drop_path_rate",
                   "stages", "base_ch", "base_channels", "depth",
                   "max_depth"]:
            if k in cfg:
                key_parts.append(f"{k}={cfg[k]}")
        config_key = " | ".join(key_parts)

        results.append({
            "run_name": d,
            "config_key": config_key,
            "model": model,
            "seed": cfg.get("seed", "?"),
            "params": cfg.get("params", 0),
            "final": final,
            "config": cfg,
        })
    return results


def group_by_config(results):
    """Group results by config_key, compute mean/std across seeds."""
    groups = defaultdict(list)
    for r in results:
        groups[r["config_key"]].append(r)

    summary = []
    for key, runs in groups.items():
        import numpy as np
        metrics = defaultdict(list)
        for r in runs:
            for k, v in r["final"].items():
                if k != "epoch" and isinstance(v, (int, float)):
                    metrics[k].append(v)

        row = {
            "config_key": key,
            "model": runs[0]["model"],
            "params": runs[0]["params"],
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
        }
        for k, vals in metrics.items():
            arr = [float(v) for v in vals]
            row[f"{k}_mean"] = sum(arr) / len(arr)
            if len(arr) > 1:
                mean = row[f"{k}_mean"]
                row[f"{k}_std"] = (sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)) ** 0.5
            else:
                row[f"{k}_std"] = 0.0

        summary.append(row)

    return summary


def print_seg_table(summary):
    """Print segmentation results table."""
    seg = [s for s in summary if s["model"] in ("fnn_seg", "unet", "resunet")]
    if not seg:
        return
    seg.sort(key=lambda s: s.get("val_iou_mean", 0), reverse=True)

    print("\n=== SEGMENTATION RESULTS (sorted by IoU) ===")
    print(f"{'Config':<55} {'Params':>8} {'Seeds':>5} "
          f"{'IoU':>12} {'F1':>12} {'Precision':>12} {'Recall':>12}")
    print("-" * 120)
    for s in seg:
        iou = f"{s.get('val_iou_mean', 0):.4f}"
        if s.get("val_iou_std", 0) > 0:
            iou += f"+/-{s['val_iou_std']:.4f}"
        f1 = f"{s.get('val_f1_mean', 0):.4f}"
        prec = f"{s.get('val_precision_mean', 0):.4f}"
        rec = f"{s.get('val_recall_mean', 0):.4f}"
        print(f"{s['config_key']:<55} {s['params']:>8,} {s['n_seeds']:>5} "
              f"{iou:>12} {f1:>12} {prec:>12} {rec:>12}")


def print_cls_table(summary):
    """Print classification results table."""
    cls = [s for s in summary
           if s["model"] in ("fnn", "cnn", "rnn", "transformer", "resnet_cls")]
    if not cls:
        return
    cls.sort(key=lambda s: s.get("val_acc_mean", 0), reverse=True)

    print("\n=== CLASSIFICATION RESULTS (sorted by Accuracy) ===")
    print(f"{'Config':<55} {'Params':>8} {'Seeds':>5} {'Accuracy':>12} {'Loss':>12}")
    print("-" * 95)
    for s in cls:
        acc = f"{s.get('val_acc_mean', 0):.4f}"
        if s.get("val_acc_std", 0) > 0:
            acc += f"+/-{s['val_acc_std']:.4f}"
        loss = f"{s.get('train_loss_mean', 0):.4f}"
        print(f"{s['config_key']:<55} {s['params']:>8,} {s['n_seeds']:>5} "
              f"{acc:>12} {loss:>12}")


def print_inpaint_table(summary):
    """Print inpainting results table."""
    inp = [s for s in summary if s["model"] == "fnn_inpaint"]
    if not inp:
        return
    inp.sort(key=lambda s: s.get("val_iou_mean", 0), reverse=True)

    print("\n=== INPAINTING RESULTS (sorted by IoU) ===")
    print(f"{'Config':<55} {'Params':>8} {'Seeds':>5} "
          f"{'IoU':>12} {'PSNR':>12} {'SSIM':>12}")
    print("-" * 100)
    for s in inp:
        iou = f"{s.get('val_iou_mean', 0):.4f}"
        psnr = f"{s.get('val_psnr_mean', 0):.2f}"
        ssim = f"{s.get('val_ssim_mean', 0):.4f}"
        print(f"{s['config_key']:<55} {s['params']:>8,} {s['n_seeds']:>5} "
              f"{iou:>12} {psnr:>12} {ssim:>12}")


def main():
    results = load_all_results()
    print(f"Total completed runs: {len(results)}")
    print(f"Run directory: {RUNS_DIR}")

    summary = group_by_config(results)
    print(f"Unique configs: {len(summary)}")

    print_seg_table(summary)
    print_cls_table(summary)
    print_inpaint_table(summary)

    # Save summary as JSON
    out_path = os.path.join(FIGS_DIR, "aggregated_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
