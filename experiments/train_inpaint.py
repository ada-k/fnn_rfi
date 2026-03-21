"""Training script for FNNInpaint: joint segmentation and spectrogram restoration.

Uses SyntheticSpectrogramDataset(return_clean=True) so that each batch
provides (contaminated, mask, clean) triples.

Loss = bce_weight * BCE(mask_logit, mask) + mse_weight * masked_MSE(restored, clean)

Usage (run from repo root):

    python -m fnn_rfi.experiments.train_inpaint \\
        --epochs 20 --batch-size 16 --seed 42

Results are written to runs/fnn_inpaint_<timestamp>/.
"""
import argparse
import json
import os
import time

import torch
import torch.optim as optim

from .evaluate import dice_score, iou_score, precision_recall_f1, psnr, ssim
from .utils import get_dataloader, get_device, save_checkpoint, save_config, set_seed
from ..models.fnn_inpaint import FNNInpaint, inpaint_loss


def train_epoch(model, loader, optimiser, bce_w, mse_w, device):
    model.train()
    total, n = 0.0, 0
    for contaminated, mask, clean in loader:
        contaminated = contaminated.to(device)
        mask = mask.to(device)
        clean = clean.to(device)
        optimiser.zero_grad()
        mask_logit, restored = model(contaminated)
        loss = inpaint_loss(mask_logit, restored, mask, clean, bce_w, mse_w)
        loss.backward()
        optimiser.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, bce_w, mse_w, device):
    model.eval()
    total_loss = 0.0
    ious, dices, f1s, psnrs, ssims = [], [], [], [], []
    for contaminated, mask, clean in loader:
        contaminated = contaminated.to(device)
        mask = mask.to(device)
        clean = clean.to(device)
        mask_logit, restored = model(contaminated)
        loss = inpaint_loss(mask_logit, restored, mask, clean, bce_w, mse_w)
        total_loss += float(loss.item())
        ious.append(iou_score(mask_logit, mask))
        dices.append(dice_score(mask_logit, mask))
        _, _, f1 = precision_recall_f1(mask_logit, mask)
        f1s.append(f1)
        psnrs.append(psnr(clean, restored))
        ssims.append(ssim(clean, restored))

    n = max(len(ious), 1)
    return {
        "val_loss": total_loss / n,
        "val_iou": sum(ious) / n,
        "val_dice": sum(dices) / n,
        "val_f1": sum(f1s) / n,
        "val_psnr": sum(psnrs) / n,
        "val_ssim": sum(ssims) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train FNNInpaint on synthetic spectrograms.")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--base-ch",    type=int,   default=16)
    parser.add_argument("--stages",     type=int,   default=3)
    parser.add_argument("--drop-path",  type=float, default=0.0)
    parser.add_argument("--out",        default="runs")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--subset",     type=int,   default=None)
    parser.add_argument("--val-size",   type=int,   default=256)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_loader, _ = get_dataloader(
        "synthetic_inpaint", batch_size=args.batch_size,
        subset=args.subset, split="train"
    )
    val_loader, _ = get_dataloader(
        "synthetic_inpaint", batch_size=args.batch_size,
        subset=args.val_size, split="val"
    )

    model = FNNInpaint(
        in_ch=1,
        base_ch=args.base_ch,
        stages=args.stages,
        drop_path_rate=args.drop_path,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"fnn_inpaint_{int(time.time())}"
    outdir = os.path.join(args.out, run_name)
    os.makedirs(outdir, exist_ok=True)
    save_config(vars(args), outdir)

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimiser,
                              args.bce_weight, args.mse_weight, device)
        v = val_epoch(model, val_loader, args.bce_weight, args.mse_weight, device)

        row = {"epoch": epoch, "train_loss": round(tr_loss, 5)}
        row.update({k: round(float(vv), 5) for k, vv in v.items()})
        history.append(row)

        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  "
            f"val_iou={v['val_iou']:.4f}  "
            f"val_psnr={v['val_psnr']:.2f}  "
            f"val_ssim={v['val_ssim']:.4f}"
        )
        save_checkpoint(
            {"model_state": model.state_dict(),
             "optim_state": optimiser.state_dict(),
             "epoch": epoch, "args": vars(args)},
            outdir, name=f"ckpt_epoch{epoch}.pth",
        )

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "history": history}, f, indent=2)

    print(f"\nDone. Results saved to: {outdir}")


if __name__ == "__main__":
    main()
