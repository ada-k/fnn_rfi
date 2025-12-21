"""Simple wrapper to ingest AOFlagger outputs and convert them to training masks.

This is intentionally small: AOFlagger outputs can be many formats. The wrapper supports:
- numpy `.npy` masks (copied into `fnn_rfi/data/labels/`)
- numpy `.npz` archives with array named `mask`
- plain binary files saved as bytes matching spectrogram size (requires `--shape`)

Extend this as needed to parse AOFlagger-specific outputs (MS files, XML masks, etc.).
"""
import argparse
import os
import numpy as np


def ensure_labels_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_npy(src: str, dst_dir: str):
    arr = np.load(src)
    fname = os.path.basename(src)
    out = os.path.join(dst_dir, fname if fname.endswith('.npy') else fname + '.npy')
    np.save(out, arr)
    print('Saved', out)


def extract_npz(src: str, dst_dir: str):
    data = np.load(src)
    if 'mask' in data:
        arr = data['mask']
        out = os.path.join(dst_dir, os.path.basename(src).replace('.npz', '.npy'))
        np.save(out, arr)
        print('Saved', out)
    else:
        print('No `mask` array found in', src)


def load_binary(src: str, shape, dtype, dst_dir: str):
    arr = np.fromfile(src, dtype=dtype)
    arr = arr.reshape(shape)
    out = os.path.join(dst_dir, os.path.basename(src) + '.npy')
    np.save(out, arr)
    print('Saved', out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--out', default='fnn_rfi/data/labels')
    p.add_argument('--shape', type=int, nargs=2, help='H W for raw binary inputs')
    p.add_argument('--dtype', default='uint8', help='dtype for raw binary inputs')
    args = p.parse_args()

    ensure_labels_dir(args.out)

    for src in args.inputs:
        if src.endswith('.npy'):
            copy_npy(src, args.out)
        elif src.endswith('.npz'):
            extract_npz(src, args.out)
        else:
            if args.shape is None:
                print('Unknown format for', src, '- provide --shape for raw binary files')
                continue
            load_binary(src, tuple(args.shape), args.dtype, args.out)


if __name__ == '__main__':
    main()
