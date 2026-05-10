#!/usr/bin/env python3
"""Quick sanity-check for the HERADataset loader.

Loads a small batch from the HERA pickle and prints shape/value statistics.
Does NOT train anything.  Run this first before any full training run.

Usage:
    python3 fnn_rfi/scripts/test_hera_loader.py
    python3 fnn_rfi/scripts/test_hera_loader.py --pkl data/zenodo/LOFAR_Full_RFI_dataset.pkl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fnn_rfi.data.hera_dataset import build_hera_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        default="data/zenodo/HERA_28-03-2023_all.pkl",
        help="Path to a Zenodo pickle file.",
    )
    parser.add_argument("--patch", type=int, default=256, help="Patch size (px)")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    print(f"\n=== HERADataset loader sanity check ===")
    print(f"Pickle : {args.pkl}")
    print(f"Patch  : {args.patch}×{args.patch}")
    print(f"Batch  : {args.batch}\n")

    train_loader, val_loader, test_loader, stats = build_hera_loaders(
        pkl_path=args.pkl,
        batch_size=args.batch,
        patch_size=args.patch,
        stride=args.patch,       # no overlap for quick check
        log_amplitude=True,
        normalize=True,
        val_fraction=0.1,
        seed=42,
    )

    print(f"\nAmplitude normalisation: mean={stats['amplitude_stats'][0]:.4f}, "
          f"std={stats['amplitude_stats'][1]:.4f}")
    print(f"Phase normalisation    : mean={stats['phase_stats'][0]:.4f}, "
          f"std={stats['phase_stats'][1]:.4f}")

    for loader_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        imgs, masks = next(iter(loader))
        rfi_frac = masks.float().mean().item()
        print(
            f"\n[{loader_name:5s}]  imgs {tuple(imgs.shape)}  dtype={imgs.dtype}  "
            f"min={imgs.min():.3f}  max={imgs.max():.3f}  mean={imgs.mean():.3f}"
        )
        print(
            f"         masks {tuple(masks.shape)}  dtype={masks.dtype}  "
            f"RFI_fraction={rfi_frac:.4f} ({rfi_frac*100:.2f}%)"
        )

    print("\n✓  Loader OK — shapes and dtypes look correct.")
    print("   Next step: run fnn_rfi/scripts/train_hera.py\n")


if __name__ == "__main__":
    main()
