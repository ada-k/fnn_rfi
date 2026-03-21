"""Generate interim visualizations from completed experiment runs."""
import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from fnn_rfi.experiments.visualize import (
    plot_segmentation_examples,
    plot_learning_curves,
    plot_multi_metric_curves,
)
from fnn_rfi.data.synthetic_spectrogram import SyntheticSpectrogramDataset
from fnn_rfi.models.fnn_seg import FNNSeg

FIGS = os.path.join(ROOT, "fnn_rfi", "figs")
RUNS = os.path.join(ROOT, "runs")
os.makedirs(FIGS, exist_ok=True)


def main():
    # Best completed seg run
    best_run = os.path.join(RUNS, "fnn_seg_fib_sep_st3_ch16_s42")
    ckpt_path = os.path.join(best_run, "best.pth")
    results_path = os.path.join(best_run, "results.json")

    if not os.path.exists(results_path):
        print("No completed runs yet.")
        return

    # Load model
    model = FNNSeg(in_ch=1, base_ch=16, stages=3,
                   use_fib=True, use_separable=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Prediction examples
    ds = SyntheticSpectrogramDataset(length=8, h=64, w=64, seed=999)
    plot_segmentation_examples(
        model, ds, n_samples=4, device="cpu",
        out_path=os.path.join(FIGS, "seg_predictions_interim.png"),
    )
    print("seg_predictions_interim.png")

    # Learning curves for all completed seg runs
    completed = []
    labels = []
    for d in sorted(os.listdir(RUNS)):
        rfile = os.path.join(RUNS, d, "results.json")
        if os.path.exists(rfile) and "fnn_seg" in d:
            completed.append(rfile)
            labels.append(d)

    if completed:
        plot_learning_curves(
            completed, labels, metric="val_iou",
            out_path=os.path.join(FIGS, "learning_curves_interim.png"),
        )
        print("learning_curves_interim.png")

    # Multi-metric for best run
    plot_multi_metric_curves(
        results_path, metrics=["val_iou", "val_f1", "val_precision", "val_recall"],
        out_path=os.path.join(FIGS, "multi_metric_best.png"),
    )
    print("multi_metric_best.png")


if __name__ == "__main__":
    main()
