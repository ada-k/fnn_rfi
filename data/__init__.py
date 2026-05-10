"""Data package: synthetic generation, RFI injection, split utilities."""

from .synthetic_spectrogram import SyntheticSpectrogramDataset, MORPHOLOGIES, INJECTORS
from .inject_rfi import inject_rfi, inject_rfi_batch
from .prepare_splits import prepare_splits, load_split_manifest

__all__ = [
    "SyntheticSpectrogramDataset",
    "MORPHOLOGIES",
    "INJECTORS",
    "inject_rfi",
    "inject_rfi_batch",
    "prepare_splits",
    "load_split_manifest",
]
