"""PyTorch Dataset for the Zenodo HERA/LOFAR RFI pickle files.

These pickles come from:
  Charl du Toit, Trienko Grobler, Danie Ludick (2023).
  "A Comparison framework for deep learning RFI detection algorithms
  in radio astronomy." Zenodo. https://doi.org/10.5281/zenodo.8275061

Pickle format (list of 4 NumPy arrays):
  [0] X_train  (N_train, H, W, 2)  float32  — amplitude + phase
  [1] y_train  (N_train, H, W, 1)  bool     — binary RFI mask
  [2] X_test   (N_test,  H, W, 2)  float32
  [3] y_test   (N_test,  H, W, 1)  bool

Channel 0: log-power amplitude (roughly 0–300 range, telescope-dependent)
Channel 1: phase in radians (-π to π)

Usage
-----
    from fnn_rfi.data.hera_dataset import HERADataset
    from torch.utils.data import DataLoader

    train_ds = HERADataset(
        pkl_path="data/zenodo/HERA_28-03-2023_all.pkl",
        split="train",
        patch_size=256,   # crop to 256×256 patches
        stride=128,       # overlap stride for patch extraction
        log_amplitude=True,
        normalize=True,
        val_fraction=0.1, # fraction of train set held out for validation
        seed=42,
    )
    val_ds = HERADataset(..., split="val", ...)
    test_ds = HERADataset(..., split="test", ...)

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    for imgs, masks in loader:
        # imgs:  (B, 2, patch_size, patch_size)  float32
        # masks: (B, 1, patch_size, patch_size)  float32  (0/1)
        ...
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _log(msg: str):
    """Print with ISO timestamp prefix, flushed immediately."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_pkl(pkl_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a Zenodo RFI pickle once and return the 4 raw arrays.

    Use this when constructing multiple splits (train/val/test) so the
    3 GB file is read from disk only once.

    Returns:
        X_train_full, y_train_full, X_test, y_test  (all NumPy arrays)
    """
    pkl_path = Path(pkl_path)
    _log(f"Loading pickle: {pkl_path.name}  ({pkl_path.stat().st_size / 1e9:.1f} GB) …")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    X_train_full, y_train_full, X_test, y_test = raw[0], raw[1], raw[2], raw[3]
    _log(f"Pickle loaded — train {X_train_full.shape}, test {X_test.shape}")
    return X_train_full, y_train_full, X_test, y_test


class HERADataset(Dataset):
    """Dataset wrapping the du Toit Zenodo HERA/LOFAR RFI pickle files.

    Pass either `pkl_path` (file is loaded internally) OR pre-loaded arrays
    via `arrays=(X_train_full, y_train_full, X_test, y_test)` to avoid
    re-reading the 3 GB file for every split.

    Args:
        pkl_path:      Path to the .pkl file (ignored if `arrays` is given).
        arrays:        Pre-loaded tuple (X_train_full, y_train_full, X_test, y_test).
                       Use load_pkl() to obtain this once, then pass to all splits.
        split:         One of "train", "val", "test".
        patch_size:    Spatial crop size in pixels. None = full 512×512 frames.
        stride:        Patch extraction stride (default = patch_size → no overlap).
        log_amplitude: Apply log1p to amplitude channel (strongly recommended).
        normalize:     Standardise each channel to zero-mean unit-variance.
        val_fraction:  Fraction of training samples reserved for validation.
        seed:          Random seed for the train/val split.
        amplitude_stats: Pre-computed (mean, std) for amplitude. Pass from training
                         set to val/test so they use the same normalisation.
        phase_stats:   Pre-computed (mean, std) for phase channel.
    """

    def __init__(
        self,
        pkl_path: Optional[str] = None,
        arrays: Optional[Tuple] = None,
        split: str = "train",
        patch_size: Optional[int] = 256,
        stride: Optional[int] = None,
        log_amplitude: bool = True,
        normalize: bool = True,
        val_fraction: float = 0.1,
        seed: int = 42,
        amplitude_stats: Optional[Tuple[float, float]] = None,
        phase_stats: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert pkl_path is not None or arrays is not None, \
            "Provide either pkl_path or pre-loaded arrays."

        # --- load pickle (only if arrays not already supplied) ---
        if arrays is not None:
            X_train_full, y_train_full, X_test, y_test = arrays
        else:
            X_train_full, y_train_full, X_test, y_test = load_pkl(pkl_path)

        # --- train / val split ---
        n_total = len(X_train_full)
        n_val = max(1, int(n_total * val_fraction))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_total)
        val_idx   = idx[:n_val]
        train_idx = idx[n_val:]

        # Fancy indexing (val/train) already produces a writable copy.
        # Test split needs an explicit copy so in-place transforms don't corrupt
        # the shared `arrays` reference passed from build_hera_loaders.
        if split == "test":
            X, y = X_test.copy(), y_test
        elif split == "val":
            X, y = X_train_full[val_idx], y_train_full[val_idx]
        else:
            X, y = X_train_full[train_idx], y_train_full[train_idx]

        n_channels = X.shape[-1]
        _log(f"Split '{split}': {len(X)} frames  channels={n_channels} → building patches …")

        # --- amplitude transform (in-place — no extra 7 GB copy) ---
        if log_amplitude:
            np.log1p(X[..., 0], out=X[..., 0])

        # --- normalisation (in-place — stats from X directly, no re-index of full array) ---
        if normalize:
            if amplitude_stats is None:
                # X[..., 0] is already log-transformed at this point
                amplitude_stats = (float(X[..., 0].mean()), float(X[..., 0].std()) + 1e-8)
                _log(f"Amplitude stats computed: mean={amplitude_stats[0]:.4f}  std={amplitude_stats[1]:.4f}")

            X[..., 0] -= amplitude_stats[0]
            X[..., 0] /= amplitude_stats[1]

            if n_channels > 1:
                if phase_stats is None:
                    phase_stats = (float(X[..., 1].mean()), float(X[..., 1].std()) + 1e-8)
                    _log(f"Phase stats computed:     mean={phase_stats[0]:.4f}  std={phase_stats[1]:.4f}")
                X[..., 1] -= phase_stats[0]
                X[..., 1] /= phase_stats[1]
            else:
                _log("Single-channel data (amplitude only) — skipping phase normalisation.")

        self.amplitude_stats = amplitude_stats
        self.phase_stats = phase_stats

        # Keep X as float32 (pickle dtype) with no redundant copy.
        # Keep y as bool to save ~5 GB RAM per split; converted to float32 in __getitem__.
        self.X = X if X.dtype == np.float32 else X.astype(np.float32)
        self.y = y  # bool mask

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.patches = self._build_patch_index()
        _log(f"Split '{split}' ready: {len(self.patches)} patches  "
             f"(patch={patch_size}px  stride={self.stride}px)")

    def _build_patch_index(self):
        """Return list of (frame_idx, row_start, col_start) tuples."""
        N, H, W, _ = self.X.shape
        ps = self.patch_size

        if ps is None:
            # whole-frame mode
            return [(i, 0, 0) for i in range(N)]

        st = self.stride
        patches = []
        for i in range(N):
            for r in range(0, H - ps + 1, st):
                for c in range(0, W - ps + 1, st):
                    patches.append((i, r, c))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        frame_i, r, c = self.patches[idx]
        ps = self.patch_size

        if ps is None:
            img = self.X[frame_i]      # (H, W, 2)
            msk = self.y[frame_i]      # (H, W, 1)
        else:
            img = self.X[frame_i, r:r + ps, c:c + ps, :]   # (ps, ps, 2)
            msk = self.y[frame_i, r:r + ps, c:c + ps, :]   # (ps, ps, 1)

        # channels-last → channels-first for PyTorch
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))   # (C, H, W) float32
        msk = torch.from_numpy(msk.astype(np.float32).transpose(2, 0, 1).copy())  # (1, H, W) float32

        return img, msk


def build_hera_loaders(
    pkl_path: str,
    batch_size: int = 8,
    patch_size: int = 256,
    stride: int = 128,
    log_amplitude: bool = True,
    normalize: bool = True,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
):
    """Load the pickle ONCE then build train/val/test DataLoaders.

    Returns (train_loader, val_loader, test_loader, stats).
    stats = {"amplitude_stats": (mean, std), "phase_stats": (mean, std)}
    """
    from torch.utils.data import DataLoader

    # Load the 3 GB pickle a single time — shared by all three splits
    _log(f"build_hera_loaders: {Path(pkl_path).name}  "
         f"patch={patch_size}  stride={stride}  batch={batch_size}  seed={seed}")
    arrays = load_pkl(pkl_path)

    train_ds = HERADataset(
        arrays=arrays, split="train",
        patch_size=patch_size, stride=stride,
        log_amplitude=log_amplitude, normalize=normalize,
        val_fraction=val_fraction, seed=seed,
    )
    stats = {
        "amplitude_stats": train_ds.amplitude_stats,
        "phase_stats": train_ds.phase_stats,
    }
    val_ds = HERADataset(
        arrays=arrays, split="val",
        patch_size=patch_size, stride=patch_size,
        log_amplitude=log_amplitude, normalize=normalize,
        val_fraction=val_fraction, seed=seed,
        amplitude_stats=stats["amplitude_stats"],
        phase_stats=stats["phase_stats"],
    )
    test_ds = HERADataset(
        arrays=arrays, split="test",
        patch_size=patch_size, stride=patch_size,
        log_amplitude=log_amplitude, normalize=normalize,
        val_fraction=val_fraction, seed=seed,
        amplitude_stats=stats["amplitude_stats"],
        phase_stats=stats["phase_stats"],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    _log(f"Loaders ready — train={len(train_ds)} patches  "
         f"val={len(val_ds)} patches  test={len(test_ds)} patches")
    return train_loader, val_loader, test_loader, stats
