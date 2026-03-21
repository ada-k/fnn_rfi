"""Smoke-test script to verify all models run a forward pass on CPU.

Runs a single forward pass for each model with random data and prints
parameter counts and output shapes. Also verifies the synthetic dataset
generates correctly.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from models.fnn import FNN
from models.fnn_seg import FNNSeg
from models.fnn_inpaint import FNNInpaint
from models.cnn import SimpleCNN
from models.resnet_baseline import ResNetClassifier, ResUNet
from models.rnn import LSTMClassifier
from models.transformer import TinyVisionTransformer
from models.unet import UNet
from data.synthetic_spectrogram import SyntheticSpectrogramDataset


def stats(name, model, x):
    with torch.no_grad():
        out = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if isinstance(out, tuple):
        shape_str = " + ".join(str(o.shape) for o in out)
    else:
        shape_str = str(out.shape)
    print(f"  {name:<36} params={params:>10,}  out={shape_str}")


def main():
    device = torch.device("cpu")

    print("\n-- Classification models (B=2, C=3, 32x32) --")
    x_cls = torch.randn(2, 3, 32, 32).to(device)
    stats("FNN (fib)",        FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True),  x_cls)
    stats("FNN (std)",        FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=False), x_cls)
    stats("FNN (sep)",        FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True, use_separable=True), x_cls)
    stats("FNN (drop-path)",  FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True, drop_path_rate=0.1), x_cls)
    stats("SimpleCNN",        SimpleCNN(in_ch=3, num_classes=10, base_channels=8), x_cls)
    stats("LSTMClassifier",   LSTMClassifier(in_ch=3, hidden_size=128, num_layers=1, num_classes=10), x_cls)
    stats("TinyViT",          TinyVisionTransformer(in_ch=3, img_size=32, patch_size=4, emb_dim=64, depth=2, num_heads=4, num_classes=10), x_cls)
    stats("ResNetClassifier", ResNetClassifier(in_ch=3, num_classes=10, base_ch=16, stages=3), x_cls)

    print("\n-- Segmentation models (B=2, C=1, 64x64) --")
    x_seg = torch.randn(2, 1, 64, 64).to(device)
    stats("FNNSeg (fib)",     FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=True),  x_seg)
    stats("FNNSeg (std)",     FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=False), x_seg)
    stats("FNNSeg (sep)",     FNNSeg(in_ch=1, base_ch=16, stages=3, use_fib=True, use_separable=True), x_seg)
    stats("UNet (base=8)",    UNet(in_ch=1, base_ch=8, depth=3), x_seg)
    stats("UNet (base=16)",   UNet(in_ch=1, base_ch=16, depth=4), x_seg)
    stats("ResUNet (base=16)",ResUNet(in_ch=1, base_ch=16, stages=3), x_seg)

    print("\n-- FNNInpaint (B=2, C=1, 64x64) — dual output --")
    stats("FNNInpaint (fib)", FNNInpaint(in_ch=1, base_ch=16, stages=3), x_seg)

    print("\n-- Synthetic dataset --")
    from torch.utils.data import DataLoader
    ds = SyntheticSpectrogramDataset(length=8, h=64, w=64, seed=0)
    spec, mask = ds[0]
    print(f"  spec shape: {spec.shape}  range=[{spec.min():.3f}, {spec.max():.3f}]")
    print(f"  mask shape: {mask.shape}  rfi_frac={mask.mean():.3f}")
    ds_clean = SyntheticSpectrogramDataset(length=4, return_clean=True)
    spec, mask, clean = ds_clean[0]
    print(f"  (return_clean) contaminated={spec.shape} mask={mask.shape} clean={clean.shape}")

    loader = DataLoader(ds, batch_size=4, num_workers=0)
    xb, yb = next(iter(loader))
    print(f"  loader batch: x={xb.shape} y={yb.shape}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
