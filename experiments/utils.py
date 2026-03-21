"""Utility functions for experiments: model factory, dataset factory, seeding, checkpoints.

Designed to be lightweight and CPU-friendly for rapid iteration.
"""
import json
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from ..models.cnn import SimpleCNN
from ..models.fnn import FNN
from ..models.fnn_seg import FNNSeg
from ..models.rnn import LSTMClassifier
from ..models.transformer import TinyVisionTransformer
from ..models.unet import UNet
from ..data.synthetic_spectrogram import SyntheticSpectrogramDataset


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: dict, outdir: str):
    """Persist experiment configuration as JSON."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def save_checkpoint(state: dict, outdir: str, name: str = "checkpoint.pth") -> str:
    """Save a PyTorch checkpoint and return its path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    torch.save(state, path)
    return path


def get_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Supported names: fnn, fnn_seg, cnn, rnn, transformer, unet.
    Extra keyword arguments are forwarded to the constructor.
    """
    name = name.lower()
    num_classes = kwargs.get("num_classes", 10)

    if name == "fnn":
        return FNN(
            in_ch=3,
            num_classes=num_classes,
            stages=3,
            base_channels=kwargs.get("base_channels", 8),
            use_fib=kwargs.get("use_fib", True),
        )
    if name == "fnn_seg":
        return FNNSeg(
            in_ch=kwargs.get("in_ch", 1),
            base_ch=kwargs.get("base_ch", 16),
            stages=kwargs.get("stages", 3),
            use_fib=kwargs.get("use_fib", True),
            max_depth=kwargs.get("max_depth", 3),
        )
    if name == "cnn":
        return SimpleCNN(
            in_ch=3,
            num_classes=num_classes,
            base_channels=kwargs.get("base_channels", 8),
        )
    if name == "rnn":
        return LSTMClassifier(
            in_ch=3,
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 1),
            num_classes=num_classes,
        )
    if name == "transformer":
        return TinyVisionTransformer(
            in_ch=3,
            img_size=32,
            patch_size=4,
            emb_dim=kwargs.get("emb_dim", 64),
            depth=kwargs.get("depth", 2),
            num_heads=kwargs.get("num_heads", 4),
            num_classes=num_classes,
        )
    if name == "unet":
        return UNet(
            in_ch=kwargs.get("in_ch", 1),
            base_ch=kwargs.get("base_ch", 8),
            depth=kwargs.get("depth", 3),
        )
    raise ValueError(f"Unknown model '{name}'. "
                     f"Choose from: fnn, fnn_seg, cnn, rnn, transformer, unet.")


def get_dataloader(
    name: str,
    batch_size: int = 16,
    subset: int = None,
    split: str = "train",
) -> Tuple[DataLoader, int]:
    """Return a DataLoader and the number of output classes / channels.

    Parameters
    ----------
    name       : ``"cifar"`` or ``"synthetic"``.
    batch_size : mini-batch size.
    subset     : if set, limits the dataset to this many samples.
    split      : ``"train"`` or ``"val"`` (ignored for synthetic, which
                 generates data on-the-fly with different seeds).
    """
    name = name.lower()

    if name == "cifar":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        is_train = (split == "train")
        ds = torchvision.datasets.CIFAR10(
            root="./data", train=is_train, download=True, transform=transform
        )
        if subset and subset < len(ds):
            from torch.utils.data import Subset
            ds = Subset(ds, list(range(subset)))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=2)
        return loader, 10

    if name == "synthetic":
        length = subset or (2048 if split == "train" else 512)
        ds = SyntheticSpectrogramDataset(length=length, h=64, w=64)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=2)
        return loader, 1  # binary mask output

    raise ValueError(f"Unknown dataset '{name}'. Choose from: cifar, synthetic.")
