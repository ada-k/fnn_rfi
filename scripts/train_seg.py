#!/usr/bin/env python3
"""Train a UNet segmentation model on synthetic spectrograms (smoke test).

This script is configurable but defaults to a tiny dataset and one epoch so it
can run on CPU for quick validation. Outputs basic loss numbers and saves a
single checkpoint.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.unet import UNet
from data.synthetic_spectrogram import SyntheticSpectrogramDataset


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, running = 0, 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(x)
        out = out.squeeze(1)  # (B, H, W)
        # use BCEWithLogits; target is 0/1 mask
        loss = loss_fn(out, y.squeeze(1))
        loss.backward()
        opt.step()
        running += loss.item()
        total += 1
        if i % 10 == 0:
            print(f"batch {i} loss {loss.item():.4f}")
    print(f"avg loss: {running/total:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SyntheticSpectrogramDataset(length=args.dataset_size, h=64, w=64)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)

    model = UNet(in_ch=1, base_ch=8, depth=3).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, loader, opt, loss_fn, device)

    # save a small checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state": model.state_dict()}, "checkpoints/unet_smoke.pth")
    print("Saved checkpoint to checkpoints/unet_smoke.pth")


if __name__ == "__main__":
    main()
