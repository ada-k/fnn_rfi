"""Utility functions for experiments: model factory, dataset factory, seeding, checkpoints.

Designed to be lightweight and CPU-friendly for rapid iteration on this laptop.
"""
import json
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision

from ..models.fnn import FNN
from ..models.cnn import SimpleCNN
from ..models.rnn import LSTMClassifier
from ..models.transformer import TinyVisionTransformer
from ..models.unet import UNet
from ..data.synthetic_spectrogram import SyntheticSpectrogramDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def save_checkpoint(state: dict, outdir: str, name: str = "checkpoint.pth"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    torch.save(state, path)
    return path


def get_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "fnn":
        return FNN(in_ch=3, num_classes=kwargs.get("num_classes", 10), stages=3, base_channels=kwargs.get("base_channels", 8), use_fib=kwargs.get("use_fib", True))
    if name == "cnn":
        return SimpleCNN(in_ch=3, num_classes=kwargs.get("num_classes", 10), base_channels=kwargs.get("base_channels", 8))
    if name == "rnn":
        return LSTMClassifier(in_ch=3, hidden_size=kwargs.get("hidden_size", 128), num_layers=kwargs.get("num_layers", 1), num_classes=kwargs.get("num_classes", 10))
    if name == "transformer":
        return TinyVisionTransformer(in_ch=3, img_size=32, patch_size=4, emb_dim=kwargs.get("emb_dim", 64), depth=kwargs.get("depth", 2), num_heads=kwargs.get("num_heads", 4), num_classes=kwargs.get("num_classes", 10))
    if name == "unet":
        return UNet(in_ch=kwargs.get("in_ch", 1), base_ch=kwargs.get("base_ch", 8), depth=kwargs.get("depth", 3))
    raise ValueError(f"Unknown model: {name}")


def get_dataloader(name: str, batch_size: int = 32, subset: int = None) -> Tuple[DataLoader, int]:
    name = name.lower()
    if name == "cifar":
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        if subset and subset < len(ds):
            from torch.utils.data import Subset
            ds = Subset(ds, list(range(subset)))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
        return loader, 10
    if name == "synthetic":
        ds = SyntheticSpectrogramDataset(length=subset or 1024, h=64, w=64)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
        return loader, 1
    raise ValueError(f"Unknown dataset: {name}")
