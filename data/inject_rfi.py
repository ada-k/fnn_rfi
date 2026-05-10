"""RFI injection utilities for creating controlled ground-truth datasets.

Applies synthetic RFI to existing clean spectrograms (real or synthetic)
with full ground truth.  Used by Phase 2 (data pipeline) and Phase 6
(downstream science evaluation).
"""
import numpy as np
from typing import List, Optional, Tuple

from .synthetic_spectrogram import INJECTORS, MORPHOLOGIES, _normalize_01


def inject_rfi(
    clean: np.ndarray,
    morphologies: Optional[Tuple[str, ...]] = None,
    snr_range: Tuple[float, float] = (1.0, 10.0),
    n_rfi_range: Tuple[int, int] = (1, 4),
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject synthetic RFI into a clean spectrogram.

    Parameters
    ----------
    clean : np.ndarray
        2-D float32 array of shape (H, W) — the clean spectrogram background.
    morphologies : tuple[str, ...], optional
        RFI types to sample from; defaults to all five morphologies.
    snr_range : tuple[float, float]
        Uniform distribution range for RFI signal-to-noise ratio.
    n_rfi_range : tuple[int, int]
        Range for the number of injected RFI events (uniform integer).
    rng : np.random.RandomState, optional
        Random state for reproducibility.  If None, a new one is created.

    Returns
    -------
    contaminated : np.ndarray — spectrogram with RFI injected, shape (H, W)
    mask         : np.ndarray — binary float32 mask, 1=RFI, shape (H, W)
    clean_norm   : np.ndarray — clean spectrogram normalised to [0, 1]
    """
    if rng is None:
        rng = np.random.RandomState()
    if morphologies is None:
        morphologies = MORPHOLOGIES

    spec = clean.copy().astype(np.float32)
    mask = np.zeros(spec.shape, dtype=np.float32)

    n_rfi = int(rng.randint(n_rfi_range[0], n_rfi_range[1]))
    for _ in range(n_rfi):
        morph = rng.choice(list(morphologies))
        snr = float(rng.uniform(*snr_range))
        spec, mask = INJECTORS[morph](spec, mask, rng, snr)

    return _normalize_01(spec), mask, _normalize_01(clean)


def inject_rfi_batch(
    clean_list: List[np.ndarray],
    seed: int = 0,
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Apply inject_rfi to a list of clean spectrograms.

    Parameters
    ----------
    clean_list : list of 2-D float32 arrays
    seed : int
        Master seed; sample i is processed with seed + i.
    **kwargs
        Forwarded to inject_rfi.

    Returns
    -------
    List of (contaminated, mask, clean_norm) tuples, same length as clean_list.
    """
    results = []
    for i, clean in enumerate(clean_list):
        rng = np.random.RandomState(seed + i)
        results.append(inject_rfi(clean, rng=rng, **kwargs))
    return results


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    clean = rng.randn(64, 64).astype(np.float32)
    contaminated, mask, clean_norm = inject_rfi(clean, rng=rng)
    print(f"contaminated: {contaminated.shape}  rfi_frac={mask.mean():.3f}")
    print(f"clean_norm:   {clean_norm.shape}")
