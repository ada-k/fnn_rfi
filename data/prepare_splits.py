"""Dataset split preparation utilities.

Creates reproducible train/val/test splits from a list of file paths or
observation IDs.  Splits are saved as JSON manifests with content hashes so
splits can be versioned and reproduced exactly.

Usage
-----
    from data.prepare_splits import prepare_splits, load_split_manifest

    manifest = prepare_splits(
        paths=spec_paths,
        out_dir="data/splits/lofar_v1",
        ratios=(0.70, 0.15, 0.15),
        seed=42,
    )
"""
import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Tuple


def _hash_list(items: List[str]) -> str:
    """Return a short SHA-256 digest of a sorted list of strings."""
    combined = "\n".join(sorted(items)).encode()
    return hashlib.sha256(combined).hexdigest()[:12]


def prepare_splits(
    paths: List[str],
    out_dir: str,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    group_by: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Split a flat list of file paths into train / val / test.

    Splits are stratified by ``group_by`` labels if provided (e.g. observation
    IDs), ensuring full observations stay in one split to avoid data leakage.

    Parameters
    ----------
    paths : list of str
        All spectrogram file paths (or arbitrary IDs).
    out_dir : str
        Directory where the JSON manifest is saved.
    ratios : tuple[float, float, float]
        (train, val, test) fractions; must sum to 1.
    seed : int
        RNG seed for reproducibility.
    group_by : list of str, optional
        If provided, must be the same length as ``paths``.  Splitting is done
        at the group level so that no group straddles splits.

    Returns
    -------
    dict with keys 'train', 'val', 'test' mapping to lists of paths.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(seed)

    if group_by is not None:
        assert len(group_by) == len(paths)
        # Group paths by their group label
        groups: Dict[str, List[str]] = {}
        for p, g in zip(paths, group_by):
            groups.setdefault(g, []).append(p)
        keys = sorted(groups.keys())
        rng.shuffle(keys)
        # Split groups
        n = len(keys)
        n_train = max(1, round(n * ratios[0]))
        n_val = max(1, round(n * ratios[1]))
        train_keys = keys[:n_train]
        val_keys = keys[n_train: n_train + n_val]
        test_keys = keys[n_train + n_val:]
        splits = {
            "train": [p for k in train_keys for p in groups[k]],
            "val": [p for k in val_keys for p in groups[k]],
            "test": [p for k in test_keys for p in groups[k]],
        }
    else:
        items = list(paths)
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * ratios[0]))
        n_val = max(1, round(n * ratios[1]))
        splits = {
            "train": items[:n_train],
            "val": items[n_train: n_train + n_val],
            "test": items[n_train + n_val:],
        }

    manifest = {
        "seed": seed,
        "ratios": list(ratios),
        "content_hash": _hash_list(paths),
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    manifest_path = os.path.join(out_dir, "splits.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Splits saved to {manifest_path}  "
        f"train={len(splits['train'])}  val={len(splits['val'])}  "
        f"test={len(splits['test'])}"
    )
    return splits


def load_split_manifest(manifest_path: str) -> Dict[str, List[str]]:
    """Load a previously saved split manifest.

    Returns the 'splits' dict with 'train', 'val', 'test' keys.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest["splits"]


if __name__ == "__main__":
    dummy_paths = [f"spec_{i:04d}.npy" for i in range(100)]
    dummy_groups = [f"obs_{i // 10:02d}" for i in range(100)]  # 10 obs of 10 patches
    splits = prepare_splits(
        dummy_paths,
        out_dir="/tmp/test_splits",
        group_by=dummy_groups,
    )
    print({k: len(v) for k, v in splits.items()})
