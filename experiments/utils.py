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
from ..models.fnn_inpaint import FNNInpaint
from ..models.fnn_seg import FNNSeg
from ..models.resnet_baseline import ResNetClassifier, ResUNet
from ..models.rnn import LSTMClassifier
from ..models.transformer import TinyVisionTransformer
from ..models.unet import UNet
from ..data.synthetic_spectrogram import SyntheticSpectrogramDataset
from ..data.hera_dataset import HERADataset, build_hera_loaders


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU.

    MPS is skipped on Python < 3.11 due to known stability issues with
    backward passes on older interpreters.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    import sys
    if sys.version_info >= (3, 11):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


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

    Supported names:
        fnn, fnn_seg, fnn_inpaint, cnn, rnn, transformer,
        unet, resnet_cls, resunet.

    Extra keyword arguments are forwarded to the constructor where accepted:
        num_classes, base_channels, use_fib, drop_path_rate, use_separable,
        in_ch, base_ch, stages, max_depth, n_blocks, ...
    """
    name = name.lower()
    num_classes = kwargs.get("num_classes", 10)
    drop_path_rate = kwargs.get("drop_path_rate", 0.0)
    use_separable = kwargs.get("use_separable", False)

    if name == "fnn":
        return FNN(
            in_ch=kwargs.get("in_ch", 3),
            num_classes=num_classes,
            stages=kwargs.get("stages", 3),
            base_channels=kwargs.get("base_channels", 8),
            use_fib=kwargs.get("use_fib", True),
            drop_path_rate=drop_path_rate,
            use_separable=use_separable,
        )
    if name == "fnn_seg":
        return FNNSeg(
            in_ch=kwargs.get("in_ch", 1),
            base_ch=kwargs.get("base_ch", 16),
            stages=kwargs.get("stages", 3),
            use_fib=kwargs.get("use_fib", True),
            max_depth=kwargs.get("max_depth", 3),
            drop_path_rate=drop_path_rate,
            use_separable=use_separable,
        )
    if name == "fnn_inpaint":
        return FNNInpaint(
            in_ch=kwargs.get("in_ch", 1),
            base_ch=kwargs.get("base_ch", 16),
            stages=kwargs.get("stages", 3),
            use_fib=kwargs.get("use_fib", True),
            max_depth=kwargs.get("max_depth", 3),
            drop_path_rate=drop_path_rate,
            use_separable=use_separable,
        )
    if name == "cnn":
        return SimpleCNN(
            in_ch=kwargs.get("in_ch", 3),
            num_classes=num_classes,
            base_channels=kwargs.get("base_channels", 8),
        )
    if name == "rnn":
        return LSTMClassifier(
            in_ch=kwargs.get("in_ch", 3),
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 1),
            num_classes=num_classes,
        )
    if name == "transformer":
        return TinyVisionTransformer(
            in_ch=kwargs.get("in_ch", 3),
            img_size=kwargs.get("img_size", 32),
            patch_size=kwargs.get("patch_size", 4),
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
    if name == "resnet_cls":
        return ResNetClassifier(
            in_ch=kwargs.get("in_ch", 3),
            num_classes=num_classes,
            base_ch=kwargs.get("base_ch", 16),
            stages=kwargs.get("stages", 3),
            n_blocks=kwargs.get("n_blocks", 2),
        )
    if name == "resunet":
        return ResUNet(
            in_ch=kwargs.get("in_ch", 1),
            base_ch=kwargs.get("base_ch", 16),
            stages=kwargs.get("stages", 3),
            out_ch=kwargs.get("out_ch", 1),
            n_blocks=kwargs.get("n_blocks", 2),
        )
    raise ValueError(
        f"Unknown model '{name}'. "
        "Choose from: fnn, fnn_seg, fnn_inpaint, cnn, rnn, transformer, "
        "unet, resnet_cls, resunet."
    )


def get_dataloader(
    name: str,
    batch_size: int = 16,
    subset: int = None,
    split: str = "train",
) -> Tuple[DataLoader, int]:
    """Return a DataLoader and the number of output classes / channels.

    Parameters
    ----------
    name       : ``"cifar"``, ``"synthetic"``, or ``"synthetic_inpaint"``.
    batch_size : mini-batch size.
    subset     : if set, limits the dataset to this many samples.
    split      : ``"train"`` or ``"val"`` (for synthetic, different seeds).

    Returns
    -------
    (loader, num_classes)
        num_classes is 10 for CIFAR and 1 for segmentation datasets.
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
        loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=0)
        return loader, 10

    if name == "synthetic":
        seed = 0 if split == "train" else 10000
        length = subset or (2048 if split == "train" else 512)
        ds = SyntheticSpectrogramDataset(length=length, h=64, w=64, seed=seed)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
        return loader, 1  # binary mask output

    if name == "synthetic_inpaint":
        seed = 0 if split == "train" else 10000
        length = subset or (2048 if split == "train" else 512)
        ds = SyntheticSpectrogramDataset(
            length=length, h=64, w=64, seed=seed, return_clean=True
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
        return loader, 1

    # --- Real HERA / LOFAR datasets ---
    # Expects name like "hera", "hera_aof", or "lofar"
    # with an optional `pkl_path` keyword in kwargs.
    # Use get_hera_loaders() for the full train/val/test triplet.
    _ZENODO = "/Users/ada/uel/data/zenodo"
    _PKL_MAP = {
        "hera":     f"{_ZENODO}/HERA_28-03-2023_all.pkl",
        "hera_aof": f"{_ZENODO}/HERA_AOF_20-07-2023_all.pkl",
        "lofar":    f"{_ZENODO}/LOFAR_Full_RFI_dataset.pkl",
    }
    if name in _PKL_MAP:
        pkl_path = _PKL_MAP[name]
        patch_size = 256
        ds = HERADataset(
            pkl_path=pkl_path,
            split=split,
            patch_size=patch_size,
            stride=patch_size if split != "train" else patch_size // 2,
            log_amplitude=True,
            normalize=True,
        )
        if subset and subset < len(ds):
            from torch.utils.data import Subset
            ds = Subset(ds, list(range(subset)))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
        return loader, 1  # binary mask output

    raise ValueError(
        f"Unknown dataset '{name}'. "
        "Choose from: cifar, synthetic, synthetic_inpaint, hera, hera_aof, lofar."
    )
