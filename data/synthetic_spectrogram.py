"""Synthetic RFI spectrogram dataset.

Generates 2-D time-frequency spectrograms (single polarisation, float32,
normalised to [0, 1]) with injected RFI contamination.  Generation is
fully deterministic given (seed, index), enabling reproducibility.

Dataset contract
----------------
Default (return_clean=False):
    returns (spectrogram, mask)
    spectrogram : float32 tensor (1, H, W) in [0, 1]
    mask        : float32 tensor (1, H, W) binary {0.0, 1.0}

With return_clean=True:
    returns (contaminated, mask, clean)
    contaminated : float32 tensor (1, H, W) — spectrogram with injected RFI
    mask         : float32 tensor (1, H, W) — binary RFI mask
    clean        : float32 tensor (1, H, W) — spectrogram without RFI

Five RFI morphology types
--------------------------
narrowband  : persistent CW-like emission in 1-3 frequency channels
time_burst  : broadband short-duration pulse
broadband   : wide-frequency moderate-duration emission
diagonal    : linearly drifting streak (satellite sweep / Doppler)
rectangle   : arbitrary rectangular RFI patch
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

MORPHOLOGIES: Tuple[str, ...] = (
    "narrowband",
    "time_burst",
    "broadband",
    "diagonal",
    "rectangle",
)


# ---------------------------------------------------------------------------
# Per-morphology injection functions
# Each function operates in-place on float32 (H, W) arrays for spec and mask.
# ---------------------------------------------------------------------------

def _inject_narrowband(
    spec: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    snr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = spec.shape
    n_chan = rng.randint(1, 4)
    f_centers = rng.randint(0, w, size=n_chan)
    f_width = rng.randint(1, 4)
    amp = max(float(snr * spec.std()), 1e-3)
    for fc in f_centers:
        f_lo = max(0, int(fc) - f_width // 2)
        f_hi = min(w, int(fc) + f_width // 2 + 1)
        spec[:, f_lo:f_hi] += amp
        mask[:, f_lo:f_hi] = 1.0
    return spec, mask


def _inject_time_burst(
    spec: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    snr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = spec.shape
    t_center = rng.randint(0, h)
    t_half = max(1, rng.randint(1, max(2, h // 10)))
    t_lo = max(0, t_center - t_half)
    t_hi = min(h, t_center + t_half + 1)
    amp = max(float(snr * spec.std()), 1e-3)
    spec[t_lo:t_hi, :] += amp
    mask[t_lo:t_hi, :] = 1.0
    return spec, mask


def _inject_broadband(
    spec: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    snr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = spec.shape
    t_lo = rng.randint(0, max(1, h - 2))
    t_span = rng.randint(2, max(3, h // 5))
    t_hi = min(h, t_lo + t_span)
    f_lo = rng.randint(0, max(1, w // 2))
    f_span = rng.randint(w // 3, w + 1)
    f_hi = min(w, f_lo + f_span)
    amp = max(float(snr * spec.std()), 1e-3)
    spec[t_lo:t_hi, f_lo:f_hi] += amp
    mask[t_lo:t_hi, f_lo:f_hi] = 1.0
    return spec, mask


def _inject_diagonal(
    spec: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    snr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = spec.shape
    f_start = rng.randint(0, w)
    f_end = rng.randint(0, w)
    streak_w = rng.randint(1, 4)
    amp = max(float(snr * spec.std()), 1e-3)
    for t in range(h):
        f_center = int(round(f_start + (f_end - f_start) * t / max(h - 1, 1)))
        f_lo = max(0, f_center - streak_w // 2)
        f_hi = min(w, f_center + streak_w // 2 + 1)
        spec[t, f_lo:f_hi] += amp
        mask[t, f_lo:f_hi] = 1.0
    return spec, mask


def _inject_rectangle(
    spec: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    snr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = spec.shape
    t_lo = rng.randint(0, max(1, h - 2))
    t_span = rng.randint(2, max(3, h // 4))
    t_hi = min(h, t_lo + t_span)
    f_lo = rng.randint(0, max(1, w - 2))
    f_span = rng.randint(2, max(3, w // 4))
    f_hi = min(w, f_lo + f_span)
    amp = max(float(snr * spec.std()), 1e-3)
    spec[t_lo:t_hi, f_lo:f_hi] += amp
    mask[t_lo:t_hi, f_lo:f_hi] = 1.0
    return spec, mask


INJECTORS = {
    "narrowband": _inject_narrowband,
    "time_burst": _inject_time_burst,
    "broadband": _inject_broadband,
    "diagonal": _inject_diagonal,
    "rectangle": _inject_rectangle,
}


# ---------------------------------------------------------------------------
# Background spectrogram generation
# ---------------------------------------------------------------------------

def _make_background(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    """Gaussian noise + mild bandpass tilt + per-row time-gain variation."""
    noise = rng.randn(h, w).astype(np.float32)
    bandpass = np.linspace(0.8, 1.2, w, dtype=np.float32)
    time_gain = (1.0 + 0.05 * rng.randn(h)).astype(np.float32)
    return noise * bandpass[np.newaxis, :] * time_gain[:, np.newaxis]


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2-D float32 array to [0, 1]."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticSpectrogramDataset(Dataset):
    """On-the-fly synthetic RFI spectrogram dataset.

    Parameters
    ----------
    length : int
        Number of samples in the dataset.
    h : int
        Spectrogram height (time bins).
    w : int
        Spectrogram width (frequency bins).
    snr_range : tuple[float, float]
        Min and max RFI signal-to-noise ratio.
    rfi_prob : float
        Probability that a sample contains at least one RFI event.
    morphologies : tuple[str, ...]
        RFI morphology types to sample from.
    seed : int
        Master seed; sample i is generated with seed + i.
    return_clean : bool
        If True, each item is (contaminated, mask, clean).
        If False (default), each item is (contaminated, mask).
    """

    def __init__(
        self,
        length: int = 2048,
        h: int = 64,
        w: int = 64,
        snr_range: Tuple[float, float] = (1.0, 10.0),
        rfi_prob: float = 0.8,
        morphologies: Tuple[str, ...] = MORPHOLOGIES,
        seed: int = 0,
        return_clean: bool = False,
    ):
        self.length = length
        self.h = h
        self.w = w
        self.snr_range = snr_range
        self.rfi_prob = rfi_prob
        self.morphologies = list(morphologies)
        self.seed = seed
        self.return_clean = return_clean

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.RandomState(self.seed + idx)

        # Generate clean background
        clean = _make_background(self.h, self.w, rng)
        spec = clean.copy()
        mask = np.zeros((self.h, self.w), dtype=np.float32)

        # Optionally inject RFI
        if rng.rand() < self.rfi_prob:
            n_rfi = rng.randint(1, 4)
            for _ in range(n_rfi):
                morph = rng.choice(self.morphologies)
                snr = float(rng.uniform(*self.snr_range))
                spec, mask = INJECTORS[morph](spec, mask, rng, snr)

        spec_t = torch.from_numpy(_normalize_01(spec)).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.return_clean:
            clean_t = torch.from_numpy(_normalize_01(clean)).unsqueeze(0)
            return spec_t, mask_t, clean_t
        return spec_t, mask_t


if __name__ == "__main__":
    ds = SyntheticSpectrogramDataset(length=16, h=64, w=64, return_clean=True)
    spec, mask, clean = ds[0]
    print(f"spec:  {spec.shape}  range=[{spec.min():.3f}, {spec.max():.3f}]")
    print(f"mask:  {mask.shape}  rfi_frac={mask.mean():.3f}")
    print(f"clean: {clean.shape} range=[{clean.min():.3f}, {clean.max():.3f}]")
