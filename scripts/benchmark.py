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
from models.fnn_inpaint import FNNInpaint
from models.fnn_seg import FNNSeg, count_params
from models.cnn import SimpleCNN
from models.resnet_baseline import ResNetClassifier, ResUNet
from models.rnn import LSTMClassifier
from models.transformer import TinyVisionTransformer
from models.unet import UNet


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def measure_latency(model: torch.nn.Module, x: torch.Tensor, n_runs: int = 50) -> float:
    """Return average inference time in milliseconds over ``n_runs`` forward passes."""
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_runs
    return elapsed_ms


def measure_throughput(model: torch.nn.Module, x: torch.Tensor, n_runs: int = 50) -> float:
    """Return throughput in samples per second."""
    batch_size = x.shape[0]
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    elapsed_s = time.perf_counter() - start
    return (batch_size * n_runs) / elapsed_s


def benchmark_models() -> List[Dict]:
    device = torch.device("cpu")
    results = []
    BATCH = 4  # use batch>1 for throughput measurement

    # ---- classification models (CIFAR-style 32×32 RGB) ---------------------
    cls_input = torch.randn(BATCH, 3, 32, 32).to(device)

    cls_models = {
        "FNN (fib, stages=3)":      FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True),
        "FNN (std, stages=3)":      FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=False),
        "FNN (sep, stages=3)":      FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True, use_separable=True),
        "SimpleCNN":                SimpleCNN(in_ch=3, num_classes=10, base_channels=8),
        "LSTMClassifier":           LSTMClassifier(in_ch=3, hidden_size=128, num_layers=1, num_classes=10),
        "TinyViT (d=2, e=64)":      TinyVisionTransformer(
                                        in_ch=3, img_size=32, patch_size=4,
                                        emb_dim=64, depth=2, num_heads=4, num_classes=10),
        "ResNetCls (base=16, s=3)": ResNetClassifier(in_ch=3, num_classes=10, base_ch=16, stages=3),
    }

    for name, model in cls_models.items():
        model.eval()
        with torch.no_grad():
            model(cls_input)
        params = count_params(model)
        latency = measure_latency(model, cls_input)
        throughput = measure_throughput(model, cls_input)
        results.append({
            "model": name, "task": "classification",
            "input": f"{BATCH}×3×32×32", "params": params,
            "latency_ms": round(latency, 3),
            "throughput_sps": round(throughput, 1),
        })

    # ---- segmentation models (synthetic 64×64 single-channel) --------------
    seg_input = torch.randn(BATCH, 1, 64, 64).to(device)

    seg_models = {
        "FNNSeg (fib, s=3)":        FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=True),
        "FNNSeg (std, s=3)":        FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=False),
        "FNNSeg (sep, s=3)":        FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=True, use_separable=True),
        "UNet (base=8, d=3)":       UNet(in_ch=1, base_ch=8, depth=3),
        "UNet (base=16, d=4)":      UNet(in_ch=1, base_ch=16, depth=4),
        "ResUNet (base=16, s=3)":   ResUNet(in_ch=1, base_ch=16, stages=3),
        "FNNInpaint (fib, s=3)":    None,  # dual output; handled below
    }

    for name, model in seg_models.items():
        if model is None:
            continue
        model.eval()
        with torch.no_grad():
            model(seg_input)
        params = count_params(model)
        latency = measure_latency(model, seg_input)
        throughput = measure_throughput(model, seg_input)
        results.append({
            "model": name, "task": "segmentation",
            "input": f"{BATCH}×1×64×64", "params": params,
            "latency_ms": round(latency, 3),
            "throughput_sps": round(throughput, 1),
        })

    # FNNInpaint separately (returns tuple)
    inpaint_m = FNNInpaint(in_ch=1, base_ch=16, stages=3)
    inpaint_m.eval()
    with torch.no_grad():
        inpaint_m(seg_input)
    params = count_params(inpaint_m)
    latency = measure_latency(inpaint_m, seg_input)
    throughput = measure_throughput(inpaint_m, seg_input)
    results.append({
        "model": "FNNInpaint (fib, s=3)", "task": "seg+inpaint",
        "input": f"{BATCH}×1×64×64", "params": params,
        "latency_ms": round(latency, 3),
        "throughput_sps": round(throughput, 1),
    })

    return results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_table(results: List[Dict]):
    header = f"{'Model':<32} {'Task':<14} {'Params':>10} {'Latency (ms)':>14} {'Throughput (S/s)':>18}"
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
            f"{r['model']:<32} {r['task']:<14} "
            f"{r['params']:>10,} {r['latency_ms']:>14.3f} "
            f"{r['throughput_sps']:>18.1f}"
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
