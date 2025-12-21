#!/usr/bin/env python3
"""Tiny training script to sanity-check the FNN implementation on CIFAR10.

This script is intentionally minimal: it trains for a single epoch on a small
subset of CIFAR10 to verify forward/backward passes, loss computation and
optimizer updates. It is not intended as a benchmark.

Usage:
  python3 fnn_rfi/scripts/train_toy.py --batch 64 --subset 512
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Make the `fnn_rfi` package directory importable when running the script
# This inserts the parent folder of `scripts/` (the project package folder)
# onto sys.path so `import models` works when executing the script from repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.fnn import FNN


def get_loader(batch_size=64, subset_size=1024):
    """Return a DataLoader for a small CIFAR10 subset.

    - Normalizes data with mean/std ~0.5 for quick convergence in smoke tests.
    - Uses a contiguous slice of the dataset for deterministic runs.
    """
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    if subset_size and subset_size < len(train):
        train = Subset(train, list(range(subset_size)))
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


def train_one_epoch(model, loader, device):
    """Run a single training epoch over `loader`.

    Reports occasional batch losses to confirm training is working.
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    total = 0
    running_loss = 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(x)
        # Some models might output a larger embedding; slice conservatively
        if out.shape[1] > 10:
            out = out[:, :10]
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        total += 1
        if i % 10 == 0:
            print(f"batch {i} loss {loss.item():.4f}")
    print(f"avg loss: {running_loss/total:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--subset", type=int, default=1024)
    args = parser.parse_args()

    # Use CPU on your laptop (no GPU available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(batch_size=args.batch, subset_size=args.subset)
    # Small model configuration for smoke tests
    model = FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True)
    model = model.to(device)
    print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_one_epoch(model, loader, device)


if __name__ == "__main__":
    main()
