"""Re-run failed or missing experiment configurations.

Usage:
    python3 -m fnn_rfi.scripts.rerun_failed

Scans the runs/ directory for configs that were attempted but failed (have
config.json but no results.json), and re-runs them.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fnn_rfi.experiments.run_grid import run_single
from fnn_rfi.experiments.utils import get_device

RUNS_DIR = os.path.join(ROOT, "runs")
DEVICE = get_device()


def find_failed():
    """Find run directories with config.json but no results.json."""
    failed = []
    for d in sorted(os.listdir(RUNS_DIR)):
        dpath = os.path.join(RUNS_DIR, d)
        if not os.path.isdir(dpath):
            continue
        cfg_path = os.path.join(dpath, "config.json")
        res_path = os.path.join(dpath, "results.json")
        if os.path.exists(cfg_path) and not os.path.exists(res_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            failed.append((d, cfg))
    return failed


def main():
    failed = find_failed()
    if not failed:
        print("No failed runs found.")
        return

    print(f"Found {len(failed)} failed runs:")
    for name, cfg in failed:
        print(f"  {name}: {cfg.get('model', '?')}")

    print(f"\nDevice: {DEVICE}")
    print("Re-running...\n")

    for name, cfg in failed:
        seed = cfg.pop("seed", 42)
        epochs = cfg.pop("epochs", 15)
        lr = cfg.pop("lr", 1e-3)
        batch_size = cfg.pop("batch_size", 16)
        dataset = cfg.pop("dataset", "synthetic")
        params = cfg.pop("params", None)
        subset = cfg.pop("subset", 1024)
        val_size = cfg.pop("val_size", 256)

        print(f"  {name} ...", end=" ", flush=True)
        try:
            summary = run_single(
                cfg, seed, epochs, batch_size, lr, dataset,
                RUNS_DIR, DEVICE, subset, val_size,
            )
            iou = summary["final_metrics"].get("val_iou", "")
            acc = summary["final_metrics"].get("val_acc", "")
            info = (f"IoU={iou:.4f}" if isinstance(iou, float)
                    else f"Acc={acc:.4f}" if isinstance(acc, float)
                    else "done")
            print(info)
        except Exception as e:
            print(f"FAILED again: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
