"""Model benchmark: compares parameter counts and CPU inference latency.

Run from the repo root (fnn_rfi/ parent directory):

    python -m fnn_rfi.scripts.benchmark

Or from inside fnn_rfi/:

    python scripts/benchmark.py

Outputs a formatted table and saves results to benchmark_results.json.
"""
import json
import os
import sys
import time
from typing import Dict, List

import torch

# Allow running as a plain script from within fnn_rfi/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.fnn import FNN
from models.fnn_seg import FNNSeg, count_params
from models.cnn import SimpleCNN
from models.rnn import LSTMClassifier
from models.transformer import TinyVisionTransformer
from models.unet import UNet


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def measure_latency(model: torch.nn.Module, x: torch.Tensor, n_runs: int = 50) -> float:
    """Return average inference time in milliseconds over ``n_runs`` forward passes."""
    model.eval()
    # warm-up
    with torch.no_grad():
        for _ in range(5):
            model(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_runs
    return elapsed_ms


def benchmark_models() -> List[Dict]:
    device = torch.device("cpu")
    results = []

    # ---- classification models (CIFAR-style 32×32 RGB) ---------------------
    cls_input = torch.randn(1, 3, 32, 32).to(device)

    cls_models = {
        "FNN (fib, stages=3)":      FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True),
        "FNN (std, stages=3)":      FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=False),
        "SimpleCNN":                SimpleCNN(in_ch=3, num_classes=10, base_channels=8),
        "LSTMClassifier":           LSTMClassifier(in_ch=3, hidden_size=128, num_layers=1, num_classes=10),
        "TinyViT (d=2, e=64)":      TinyVisionTransformer(
                                        in_ch=3, img_size=32, patch_size=4,
                                        emb_dim=64, depth=2, num_heads=4, num_classes=10
                                    ),
    }

    for name, model in cls_models.items():
        model.eval()
        # trigger lazy layers if any (LSTM uses LazyLinear)
        with torch.no_grad():
            model(cls_input)
        params = count_params(model)
        latency = measure_latency(model, cls_input)
        results.append({
            "model": name,
            "task": "classification",
            "input": "1×3×32×32",
            "params": params,
            "latency_ms": round(latency, 3),
        })

    # ---- segmentation models (synthetic 64×64 single-channel) --------------
    seg_input = torch.randn(1, 1, 64, 64).to(device)

    seg_models = {
        "FNNSeg (fib, s=3)":    FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=True),
        "FNNSeg (std, s=3)":    FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=False),
        "UNet (base=8, d=3)":   UNet(in_ch=1, base_ch=8, depth=3),
        "UNet (base=16, d=4)":  UNet(in_ch=1, base_ch=16, depth=4),
    }

    for name, model in seg_models.items():
        model.eval()
        with torch.no_grad():
            model(seg_input)
        params = count_params(model)
        latency = measure_latency(model, seg_input)
        results.append({
            "model": name,
            "task": "segmentation",
            "input": "1×1×64×64",
            "params": params,
            "latency_ms": round(latency, 3),
        })

    return results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_table(results: List[Dict]):
    header = f"{'Model':<30} {'Task':<16} {'Params':>10} {'Latency (ms)':>14}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    last_task = None
    for r in results:
        if r["task"] != last_task and last_task is not None:
            print()
        last_task = r["task"]
        print(
            f"{r['model']:<30} {r['task']:<16} "
            f"{r['params']:>10,} {r['latency_ms']:>14.3f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nRunning benchmark on CPU …\n")
    results = benchmark_models()
    print_table(results)

    # Save next to the script's package root (fnn_rfi/)
    out_path = os.path.join(ROOT, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}\n")
