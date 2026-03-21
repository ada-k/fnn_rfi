"""Run SNR robustness sweep on the best segmentation model."""
import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from fnn_rfi.experiments.visualize import run_snr_sweep
from fnn_rfi.models.fnn_seg import FNNSeg

FIGS = os.path.join(ROOT, "fnn_rfi", "figs")
RUNS = os.path.join(ROOT, "runs")


def main():
    # Use the best-performing run (fib, no sep, seed=42)
    best_dir = os.path.join(RUNS, "fnn_seg_fib_st3_ch16_s42")
    ckpt = os.path.join(best_dir, "best.pth")

    if not os.path.exists(ckpt):
        # Fall back to fib+sep if fib-only not done
        best_dir = os.path.join(RUNS, "fnn_seg_fib_sep_st3_ch16_s42")
        ckpt = os.path.join(best_dir, "best.pth")
        model = FNNSeg(in_ch=1, base_ch=16, stages=3,
                       use_fib=True, use_separable=True)
        print("Using FNNSeg (fib+sep)")
    else:
        model = FNNSeg(in_ch=1, base_ch=16, stages=3,
                       use_fib=True, use_separable=False)
        print("Using FNNSeg (fib)")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    snr_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    out_path = os.path.join(FIGS, "snr_sweep.png")
    _, snr_data = run_snr_sweep(
        model, snr_values, device="cpu", n_samples=256,
        out_path=out_path,
    )

    # Save data
    data_path = os.path.join(FIGS, "snr_sweep_data.json")
    with open(data_path, "w") as f:
        json.dump({str(k): v for k, v in snr_data.items()}, f, indent=2)

    print(f"SNR sweep saved to {out_path}")
    print("Results:")
    for snr, metrics in sorted(snr_data.items()):
        print(f"  SNR={snr:5.1f}: IoU={metrics['iou']:.4f}  "
              f"F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
