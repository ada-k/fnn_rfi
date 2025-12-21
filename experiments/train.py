"""Unified training script for quick experiments.

Usage examples (run from repo root):

python -m experiments.train --model fnn --dataset cifar --epochs 2 --batch-size 32
python -m experiments.train --model unet --dataset synthetic --epochs 1 --batch-size 8
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import set_seed, get_model, get_dataloader, save_checkpoint, save_config


def train_epoch(model, loader, optimz, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimz.zero_grad()
        out = model(xb)
        # adapt to classification vs segmentation
        if out.shape == yb.shape:
            loss = criterion(out, yb)
        else:
            # assume classification logits
            loss = criterion(out, yb.long())
        loss.backward()
        optimz.step()
        total_loss += float(loss.item())
        count += 1
    return total_loss / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name: fnn|cnn|rnn|transformer|unet")
    parser.add_argument("--dataset", required=True, help="Dataset: cifar|synthetic")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu")

    loader, num_classes = get_dataloader(args.dataset, batch_size=args.batch_size, subset=args.subset)
    model = get_model(args.model, num_classes=num_classes)
    model.to(device)

    # Choose loss
    if args.model == "unet":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimz = optim.Adam(model.parameters(), lr=args.lr)

    # run directory
    run_name = f"{args.model}_{args.dataset}_{int(time.time())}"
    outdir = os.path.join(args.out, run_name)
    os.makedirs(outdir, exist_ok=True)
    save_config(vars(args), outdir)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimz, criterion, device)
        print(f"epoch {epoch} loss {loss:.4f}")
        # save checkpoint each epoch
        save_checkpoint({
            "model_state": model.state_dict(),
            "optim_state": optimz.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, outdir, name=f"ckpt_epoch{epoch}.pth")

    print("Training finished. Outputs saved to:", outdir)


if __name__ == "__main__":
    main()
