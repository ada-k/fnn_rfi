"""Spectrogram extraction from multiple radio-astronomy data formats.

Supports:
  .npy / .npz   — pre-extracted numpy arrays (primary format for this repo)
  .fits         — FITS spectrograms via astropy (optional dependency)
  .ms           — CASA Measurement Sets via python-casacore (optional)

All outputs are normalised float32 arrays of shape (H, W).

Astropy and python-casacore are soft dependencies — the module imports
gracefully without them and raises ImportError only when the relevant
format is requested.
"""
import os
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Format handlers
# ---------------------------------------------------------------------------

def load_npy(path: str) -> np.ndarray:
    """Load a .npy or .npz spectrogram (key='spec' or first array)."""
    if path.endswith(".npz"):
        data = np.load(path)
        key = "spec" if "spec" in data else list(data.keys())[0]
        arr = data[key].astype(np.float32)
    else:
        arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]  # drop channel dim if present
    return arr


def load_fits(path: str, hdu_index: int = 0, stokes_index: int = 0) -> np.ndarray:
    """Load a FITS file as a 2-D time-frequency spectrogram.

    Requires astropy. The 2-D plane is selected from the HDU array by taking
    stokes_index along the first non-time/frequency axis.
    """
    try:
        from astropy.io import fits  # type: ignore
    except ImportError as exc:
        raise ImportError("astropy is required to load FITS files: pip install astropy") from exc

    with fits.open(path) as hdul:
        data = hdul[hdu_index].data.astype(np.float32)

    # Squeeze trailing size-1 axes then handle common FITS axis orders
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[stokes_index]
    if data.ndim == 4:
        return data[stokes_index, 0]
    raise ValueError(f"Cannot reduce FITS array of shape {data.shape} to 2-D.")


def load_measurement_set(
    ms_path: str,
    column: str = "DATA",
    baseline: Tuple[int, int] = (0, 1),
    time_slice: Optional[slice] = None,
) -> np.ndarray:
    """Extract a single-baseline time-frequency spectrogram from a Measurement Set.

    Requires python-casacore. Returns amplitude of the first Stokes parameter
    averaged over the selected time range.

    Parameters
    ----------
    ms_path : str
        Path to the .ms directory.
    column : str
        Visibility column to read, e.g. 'DATA' or 'CORRECTED_DATA'.
    baseline : tuple[int, int]
        (ANTENNA1, ANTENNA2) indices to extract.
    time_slice : slice, optional
        Slice into the time axis.
    """
    try:
        from casacore.tables import table  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-casacore is required for MS support: "
            "pip install python-casacore  (also needs system casacore libs)"
        ) from exc

    t = table(ms_path, readonly=True)
    ant1 = t.getcol("ANTENNA1")
    ant2 = t.getcol("ANTENNA2")
    mask = (ant1 == baseline[0]) & (ant2 == baseline[1])
    data = t.getcol(column)[mask]  # (N_time, N_freq, N_pol)
    t.close()

    if time_slice is not None:
        data = data[time_slice]

    # Take amplitude of first polarisation
    spec = np.abs(data[:, :, 0]).astype(np.float32)
    return spec


# ---------------------------------------------------------------------------
# Normalisation / preprocessing
# ---------------------------------------------------------------------------

def preprocess_spectrogram(
    spec: np.ndarray,
    clip_sigma: float = 5.0,
) -> np.ndarray:
    """Normalise a raw spectrogram for use as model input.

    Steps:
    1. Clip extreme values at ``clip_sigma`` standard deviations.
    2. Scale to [0, 1].

    Parameters
    ----------
    spec : np.ndarray
        Raw 2-D float32 spectrogram (H, W).
    clip_sigma : float
        Sigma threshold for clipping.
    """
    mu = float(np.median(spec))
    sigma = float(np.std(spec))
    lo = mu - clip_sigma * sigma
    hi = mu + clip_sigma * sigma
    spec = np.clip(spec, lo, hi)
    if hi > lo:
        spec = (spec - lo) / (hi - lo)
    else:
        spec = np.zeros_like(spec)
    return spec.astype(np.float32)


def extract_patches(
    spec: np.ndarray,
    patch_h: int = 64,
    patch_w: int = 64,
    stride_h: int = 32,
    stride_w: int = 32,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Cut a large spectrogram into overlapping patches.

    Returns a list of (patch_array, (row_offset, col_offset)) tuples.
    Patches that extend beyond the spectrogram edge are zero-padded.
    """
    h, w = spec.shape
    patches = []
    for r in range(0, h, stride_h):
        for c in range(0, w, stride_w):
            patch = np.zeros((patch_h, patch_w), dtype=np.float32)
            r_end = min(r + patch_h, h)
            c_end = min(c + patch_w, w)
            patch[: r_end - r, : c_end - c] = spec[r:r_end, c:c_end]
            patches.append((patch, (r, c)))
    return patches


# ---------------------------------------------------------------------------
# Batch extraction helper
# ---------------------------------------------------------------------------

def extract_from_directory(
    data_dir: str,
    out_dir: str,
    fmt: str = "npy",
    patch_h: int = 64,
    patch_w: int = 64,
    stride_h: int = 32,
    stride_w: int = 32,
) -> List[str]:
    """Extract and patch all spectrograms in a directory.

    Parameters
    ----------
    data_dir : str
        Source directory containing raw spectrograms.
    out_dir : str
        Destination directory for extracted patches.
    fmt : str
        Input format: 'npy', 'fits', or 'ms'.
    patch_h, patch_w : int
        Patch dimensions.
    stride_h, stride_w : int
        Stride between patch centres.

    Returns
    -------
    List of output .npy file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    loaders = {"npy": load_npy, "fits": load_fits}
    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: npy, fits.")

    ext_map = {"npy": (".npy", ".npz"), "fits": (".fits", ".fit")}
    exts = ext_map[fmt]

    out_paths = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(exts):
            continue
        src = os.path.join(data_dir, fname)
        raw = loader(src)
        spec = preprocess_spectrogram(raw)
        patches = extract_patches(spec, patch_h, patch_w, stride_h, stride_w)
        stem = os.path.splitext(fname)[0]
        for patch, (r, c) in patches:
            out_name = f"{stem}_r{r:04d}_c{c:04d}.npy"
            out_path = os.path.join(out_dir, out_name)
            np.save(out_path, patch)
            out_paths.append(out_path)

    print(f"Extracted {len(out_paths)} patches from {data_dir} → {out_dir}")
    return out_paths


if __name__ == "__main__":
    # Quick self-test with a random array
    rng = np.random.RandomState(0)
    fake_spec = rng.randn(128, 256).astype(np.float32)
    spec_pp = preprocess_spectrogram(fake_spec)
    patches = extract_patches(spec_pp, patch_h=64, patch_w=64)
    print(f"Preprocessed spec range: [{spec_pp.min():.3f}, {spec_pp.max():.3f}]")
    print(f"Patches: {len(patches)}, first patch shape: {patches[0][0].shape}")
