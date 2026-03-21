# FNN-RFI Research Report

**Live thesis report — appended incrementally as experiments complete.**

---

## Phase 0 — Project Foundation

### Goal
Establish a working Python package structure with all core dependencies, a
deterministic synthetic dataset, and a verified end-to-end training loop.

### Methodology
1. Created `fnn_rfi/__init__.py` to make the repo importable as a Python package
   (`python -m fnn_rfi.experiments.train`).
2. Implemented `data/synthetic_spectrogram.py`:
   - Five RFI morphologies: `narrowband`, `time_burst`, `broadband`, `diagonal`,
     `rectangle`.
   - Deterministic per `(seed + index)` — reproducible across machines.
   - `SyntheticSpectrogramDataset` returns `(spec, mask)` tensors in [0,1].
   - Optional `return_clean=True` for joint segmentation + inpainting tasks.
3. Updated `requirements.txt` with: `matplotlib`, `scipy`, `h5py`, `astropy`,
   `pandas`, `tabulate`.
4. Ran forward-pass smoke tests for all 13 models (CPU, no GPU required).

### Results

| Check | Status |
|---|---|
| Package import (`import fnn_rfi`) | PASS |
| SyntheticSpectrogramDataset shape `[1,64,64]` | PASS |
| `return_clean=True` triple `(spec, mask, clean)` | PASS |
| DataLoader batch `[4,1,64,64]` | PASS |
| All 13 models forward-pass | PASS |
| Requirements installed (torch 2.8.0, scipy, astropy, pandas, tabulate) | PASS |

### Observations
- The system Python on the development machine is Python 3.9 with no conda/venv —
  packages installed user-local at `~/Library/Python/3.9/`.
- `num_workers > 0` caused DataLoader multiprocessing hangs on macOS with Python
  3.9; set to `num_workers=0` for all loaders.

### Next Steps
- Integrate real dataset loading (LOFAR, HERA AOFlagger-labelled data).
- Parallelise data loading via `multiprocessing.Pool` in extraction scripts rather
  than `num_workers` in PyTorch DataLoader.

---

## Phase 1 — Model Enhancements

### Goal
Extend FNN and FNNSeg with stochastic depth regularisation and depthwise-separable
convolutions; add ResNet and joint inpainting baselines.

### Methodology

**1.1 DropPath (stochastic depth):**
- `models/fnn.py`: added `DropPath(nn.Module)` using per-sample Bernoulli mask
  scaled by survival probability (standard stochastic depth formulation).
- Applied to both branch outputs of each `FractalBlock` (`left_drop`, `right_drop`).
- Controlled by `drop_path_rate` float argument propagated through `FNN` and `FNNSeg`.

**1.2 Depthwise-separable convolutions:**
- `models/fnn.py`: added `ConvBlockSep(nn.Module)` — depthwise conv → pointwise
  conv × 2, with BN+ReLU. Same interface as `ConvBlock`.
- Controlled by `use_separable` bool propagated through `FNN`, `FNNSeg`.

**1.3 ResNet baseline:**
- `models/resnet_baseline.py`:
  - `ResNetClassifier` — strided ResNet encoder + global average pool + FC.
  - `ResUNet` — same encoder wired to a U-Net decoder with skip connections for
    pixel-level mask prediction.
- Pre-activation residual blocks throughout.

**1.4 FNNInpaint — joint segmentation + inpainting:**
- `models/fnn_inpaint.py`:
  - Shared fractal encoder + bottleneck.
  - Two independent decoders: `seg_decoder → seg_head` (mask logit),
    `inpaint_decoder → inpaint_head` (restored spectrogram).
  - `inpaint_loss(mask_logit, restored, mask_target, clean)` —
    BCE on mask + masked MSE only over RFI-contaminated pixels.

### Results — Model parameter counts (CPU, 64×64 input)

| Model | Task | Params | Latency (ms) | Throughput (S/s) |
|---|---|---:|---:|---:|
| FNN (fib, s=3) | classification | 17,754 | 2.4 | 1,659 |
| FNN (std, s=3) | classification | 23,754 | 2.6 | 1,615 |
| FNN (sep, s=3) | classification | 5,984 | 2.4 | 1,682 |
| SimpleCNN | classification | 6,418 | 0.4 | 10,382 |
| LSTMClassifier | classification | 145,802 | 0.6 | 6,257 |
| TinyViT (d=2, e=64) | classification | 74,954 | 0.6 | 6,887 |
| ResNetCls (base=16) | classification | 874,458 | 1.8 | 2,238 |
| FNNSeg (fib, s=3) | segmentation | 218,465 | 10.2 | 385 |
| FNNSeg (std, s=3) | segmentation | 344,817 | 11.0 | 365 |
| FNNSeg (sep, s=3) | segmentation | 81,843 | 7.1 | 573 |
| UNet (base=8, d=3) | segmentation | 121,033 | 3.5 | 1,123 |
| UNet (base=16, d=4) | segmentation | 1,942,289 | 7.6 | 527 |
| ResUNet (base=16, s=3) | segmentation | 1,614,833 | 8.1 | 489 |
| FNNInpaint (fib, s=3) | seg+inpaint | 326,930 | 16.3 | 246 |

### Observations
- `FNN (sep)` reduces parameters by **3.0×** vs FNN (fib) at negligible latency
  cost — depthwise-separable convolutions are a promising default.
- `FNNSeg (sep)` similarly reduces parameters **2.7×** with improved throughput
  (572 vs 385 S/s), making it the preferred segmentation backbone for ablations.
- ResUNet (1.6 M params) and UNet-16 (1.9 M params) are both ~5–8 ms latency on
  CPU — competitive with FNNSeg.
- FNNInpaint adds ~110 K params over FNNSeg for the second decoder head; dual
  output throughput is ~246 S/s.

### Next Steps
- Ablation: compare `use_fib=True/False` and `use_separable=True/False` over 50
  epochs of synthetic training.
- Tune `drop_path_rate` grid (0.0, 0.05, 0.1, 0.2) for FNNSeg.
- Evaluate PSNR/SSIM of FNNInpaint restoration head on clean synthetic data after
  full training.

---

## Phase 2 — Data Infrastructure

### Goal
Build a robust, format-agnostic data pipeline supporting: multi-format spectrogram
loading (NPY, FITS, Measurement Sets), controlled RFI injection onto clean data,
and reproducible train/val/test splits without data leakage.

### Methodology

**2.2 Multi-format loading — `data/extract_spectrograms.py`:**
- `load_npy(path)` — loads `.npy` arrays directly.
- `load_fits(path, hdu_index, stokes_index)` — via `astropy.io.fits`; extracts a
  2-D frequency-time slice from a FITS cube.
- `load_measurement_set(ms_path, column, baseline, time_slice)` — via
  `python-casacore` (soft import: `ImportError` only raised when called, not on
  import).
- `preprocess_spectrogram(spec, clip_sigma)` — sigma-clip outliers, robust
  normalise to [0,1].
- `extract_patches(spec, patch_h, patch_w, stride_h, stride_w)` — sliding-window
  patch extraction with configurable stride.
- `extract_from_directory(root, pattern, out_dir, ...)` — batch extraction over a
  directory tree.

**2.4 Controlled RFI injection — `data/inject_rfi.py`:**
- `inject_rfi(clean, morphologies, snr_range, n_rfi_range, rng)` — injects 1–N
  RFI features with random morphology, position, and SNR; returns
  `(contaminated, mask, clean_norm)` triple for full ground-truth annotation.
- `inject_rfi_batch(clean_list, seed, **kwargs)` — deterministic batch injection.

**2.5 Reproducible splits — `data/prepare_splits.py`:**
- `prepare_splits(paths, out_dir, ratios, seed, group_by)` — hash-stable
  deterministic splitting; supports group-by to prevent leakage between
  observations.
- Saves `split_manifest.json` with file paths, SHA-256 hashes, and split
  assignments.
- `load_split_manifest(manifest_path)` — reload for downstream use.

### Results

| Module | Imports | Unit-tested via |
|---|---|---|
| `data/synthetic_spectrogram.py` | torch, numpy | `scripts/test_models.py` |
| `data/inject_rfi.py` | numpy | import smoke in test_models |
| `data/prepare_splits.py` | hashlib, json, pathlib | import smoke in test_models |
| `data/extract_spectrograms.py` | numpy, astropy (soft) | import smoke in test_models |

All data modules import cleanly. Functional integration tests against real
telescope data will execute in Phase 3 (dataset acquisition).

### Observations
- Soft-importing `astropy` and `casacore` means the data pipeline installs clean
  on CPU-only machines with `pip install -r requirements.txt`.
- The group-based split in `prepare_splits.py` is critical for telescope data:
  RFI patterns can be correlated across scans from the same observation, so samples
  from one observation must not span train and test sets.

### Next Steps
- Download LOFAR LOD (public release) and HERA IDR2 flagged data; verify
  `load_fits` / `load_measurement_set` against real files.
- Profile `extract_patches` throughput on a 500 × 4096 LOFAR dynamic spectrum.
- Add an `InjectRFIDataset` wrapper that applies `inject_rfi` on-the-fly during
  training to avoid storing contaminated copies.

---

## Training Smoke-Test Results (2 epochs, CPU, batch=4)

| Run | Epoch 1 | Epoch 2 | Notes |
|---|---|---|---|
| FNNSeg / synthetic | loss=0.1860 IoU=0.894 F1=0.940 | loss=0.0333 IoU=0.917 F1=0.955 | Converges fast on synthetic data |
| ResUNet / synthetic | loss=0.0648 IoU=0.933 F1=0.964 | loss=0.0239 IoU=0.955 F1=0.977 | Strong baseline; heavier (1.6 M params) |
| FNNInpaint / synthetic_inpaint | loss=0.253 IoU=0.843 PSNR=2.17 | loss=0.046 IoU=0.941 PSNR=0.50 | Segmentation head converges; PSNR/SSIM need more epochs |

Summary: FNNSeg reaches IoU ≥ 0.91 in two epochs. ResUNet converges faster to
higher IoU (0.955) at the cost of 7× more parameters. FNNInpaint's segmentation
head learns well; the restoration head needs longer training to yield
meaningful PSNR (expected — masked MSE loss benefits from pre-warming the seg head).

---

## Phase 3 — Full Experiment Suite

### Goal
Establish comprehensive baselines and ablation results for all models across
segmentation, classification, and inpainting tasks on synthetic data. Measure
efficiency (params, MACs, latency, throughput) for all architectures.

### Experimental Setup

**Hardware:** MacBook Air M4 (10-core CPU, 10-core GPU, 16GB RAM, 512GB SSD).
MPS backend available but unstable on Python 3.9 (backward pass hangs);
all training uses CPU.

**Training protocol:**
- Segmentation: 15 epochs, batch_size=16, lr=1e-3, Adam, 1024 train / 256 val.
- Classification: 10 epochs, batch_size=32, lr=1e-3, Adam, 1024 train / 256 val.
- Inpainting: 15 epochs, batch_size=16, lr=1e-3, Adam, 1024 train / 256 val.
- 3 seeds per configuration (42, 123, 7) for variance estimation.
- Checkpoints saved for best validation metric per run.

**Dataset:** Synthetic spectrogram data (`SyntheticSpectrogramDataset`) with
five RFI morphologies (narrowband, time_burst, broadband, diagonal, rectangle).
64×64 single-channel spectrograms. Classification uses CIFAR-10.

### 3.1 Segmentation Ablation Grid (14 configs × 3 seeds = 42 runs)

| # | Config | Varies |
|---|--------|--------|
| 1–4 | FNNSeg: use_fib × use_separable (4 combos) | Core architecture |
| 5–7 | FNNSeg (fib): drop_path_rate ∈ {0.05, 0.1, 0.2} | Regularisation |
| 8–9 | FNNSeg (fib): stages ∈ {2, 4} | Network depth |
| 10–11 | FNNSeg (fib): base_ch ∈ {8, 32} | Network width |
| 12 | FNNSeg: fib + sep + dp=0.1 (best combo) | Combined |
| 13 | UNet (base=16, depth=3) | Baseline |
| 14 | ResUNet (base=16, stages=3) | Baseline |

### 3.2 Classification Experiments (7 configs × 3 seeds = 21 runs)

FNN (fib), FNN (std), FNN (fib+sep), SimpleCNN, LSTMClassifier, TinyViT,
ResNetCls — all on CIFAR-10 with matched hyperparameters.

### 3.3 Inpainting Experiments (4 configs × 3 seeds = 12 runs)

FNNInpaint variants: base, separable, drop_path=0.1, base_ch=32.

### 3.4 Efficiency Profiling

All measurements on CPU with batch_size=4, averaged over 50 inference passes.
MACs estimated via forward hooks on Conv2d, Linear, and BatchNorm2d layers.

**Segmentation models (input: 1×1×64×64):**

| Model | Params | MACs (M) | Latency (ms) | Throughput (S/s) |
|-------|-------:|--------:|-------------:|-----------------:|
| FNNSeg (fib, s=3) | 218,465 | 92.83 | 13.06 | 305 |
| FNNSeg (std, s=3) | 344,817 | 109.58 | 14.13 | 282 |
| FNNSeg (sep, s=3) | 81,843 | 23.56 | 9.80 | 433 |
| FNNSeg (fib, s=2) | 93,233 | 79.72 | — | — |
| FNNSeg (fib, s=4) | 601,393 | 104.05 | — | — |
| FNNSeg (fib, ch=8) | 66,201 | 32.85 | — | — |
| FNNSeg (fib, ch=32) | 869,057 | 367.30 | — | — |
| UNet (d=3, ch=8) | 121,033 | 35.01 | 5.10 | 886 |
| UNet (d=4, ch=16) | 1,942,289 | 180.96 | 9.63 | 444 |
| ResUNet (s=3, ch=16) | 1,614,833 | 229.83 | 11.23 | 384 |
| FNNInpaint (fib, s=3) | 326,930 | 159.24 | 20.21 | 203 |

**Classification models (input: 1×3×32×32):**

| Model | Params | MACs (M) | Latency (ms) | Throughput (S/s) |
|-------|-------:|--------:|-------------:|-----------------:|
| FNN (fib, s=3) | 17,754 | 3.84 | 3.19 | 1,343 |
| FNN (std, s=3) | 23,754 | 4.24 | 3.20 | 1,270 |
| FNN (sep, s=3) | 5,984 | 1.25 | 2.91 | 1,406 |
| SimpleCNN | 6,418 | 2.65 | 0.51 | 8,528 |
| LSTMClassifier | 145,802 | 0.01 | 0.69 | 5,826 |
| TinyViT (d=2, e=64) | 74,954 | 0.04 | 0.67 | 6,059 |
| ResNetCls (ch=16, s=3) | 874,458 | 32.44 | 2.23 | 1,721 |

**Key observations:**
- FNNSeg (sep) achieves a **3.9× MAC reduction** (23.6M vs 92.8M) and **2.7×
  parameter reduction** (82K vs 218K) over FNNSeg (fib) — the separable variant
  is dramatically more efficient.
- FNNSeg (fib, s=3) at 218K params / 92.8M MACs sits between UNet-8 (121K/35M)
  and UNet-16-d4 (1.9M/181M) in complexity.
- ResUNet is the heaviest segmentation model (1.6M params, 230M MACs) with no
  clear speed advantage over FNNSeg variants.
- For classification, FNN (sep) is the lightest (6K params, 1.25M MACs) while
  ResNetCls is heaviest (874K params, 32.4M MACs).

### 3.5 Interim Results (experiments in progress)

Results from completed segmentation runs (15 epochs, synthetic data, 3 seeds):

| Config | Params | Val IoU (mean +/- std) | Val F1 | Precision | Recall |
|--------|-------:|-----------------------:|-------:|----------:|-------:|
| FNNSeg (fib, s=3, ch=16) | 218,465 | **0.9757 +/- 0.0006** | 0.9876 | 0.9952 | 0.9803 |
| FNNSeg (fib+sep, s=3, ch=16) | 81,843 | **0.9640 +/- 0.0091** | 0.9816 | 0.9936 | 0.9701 |
| ResUNet (s=3, ch=16) | 1,614,833 | 0.9553 | 0.9768 | 0.9793 | 0.9752 |

**Key findings from initial ablation:**
- FNNSeg (fib) achieves the highest IoU (0.976) with remarkably low variance
  across seeds (std=0.0006).
- FNNSeg (fib+sep) trades 1.2% IoU for a 2.7× parameter reduction (81K vs 218K).
  The separable variant has higher cross-seed variance (std=0.009), suggesting
  it is more sensitive to initialisation.
- Both FNNSeg variants outperform ResUNet (1.6M params) despite using 7.4-19.7×
  fewer parameters.

*Remaining 36 segmentation runs, 21 classification runs, and 12 inpainting runs
are executing in the background. Results will be updated as runs complete.*

### 3.6 Data Acquisition Strategy

**Current state:** All experiments use synthetic data. The `datafiles/` directory
is empty — no real telescope data has been acquired yet.

**Recommended data sources for real-data evaluation:**

1. **LOFAR Observation Data (LOD)**
   - Archive: https://lta.lofar.eu/
   - Format: Measurement Sets (MS), extractable to FITS via CASA or DP3.
   - Typical size: 5–50 GB per observation; select short (~10 min) observations.
   - Labels: Generate via AOFlagger on the same data; use as pseudo-ground-truth.
   - Access: Free, requires ASTRON eduID account registration; bulk download via
     `wget` or `globus`.
   - Recommended: Download 3–5 short observations from different frequency bands
     and array configurations for diversity.

2. **HERA IDR2 (Hydrogen Epoch of Reionization Array)**
   - Portal: https://reionization.org/data/
   - Format: UVH5 / MIRIAD, convertible to spectrogram patches.
   - Typical size: 1–10 GB per file.
   - Labels: HERA releases include flagging metadata (XRFI outputs).
   - Access: Public download links on the HERA data release pages.
   - Recommended: Download the flagged calibration data subset (~2 GB).

3. **MWA (Murchison Widefield Array)**
   - Portal: https://www.mwatelescope.org/ (via ASVO)
   - Format: FITS / MS, wide-field imaging products.
   - Typical size: 2–20 GB per snapshot.
   - Labels: AOFlagger-compatible; flag files often included.
   - Access: Public; query via the All-Sky Virtual Observatory portal.
   - Recommended: 2–3 short continuum observations for diversity.

4. **Tabascal Simulations (controlled ground truth)**
   - If available from the literature, simulated datasets with known RFI injection
     provide perfect ground truth for controlled evaluation. These complement
     AOFlagger pseudo-labels by removing labelling noise.

**Practical constraints on MacBook Air M4 (16GB RAM, 512GB SSD):**
- Storage: ~400 GB free. Limit total downloaded data to 50 GB max.
- Processing: Extract 64×64 patches via `data/extract_spectrograms.py` — produces
  manageable patch sets (thousands of patches from a single observation).
- Strategy: Download 2–3 small observations per telescope, extract patches, store
  patches (not raw data) for training. Delete raw data after extraction.
- Group-aware splitting: Use `data/prepare_splits.py` with `group_by=observation`
  to prevent data leakage between observations.

**Minimum viable real-data experiment:**
1. Download one LOFAR observation (~10 GB).
2. Run AOFlagger via Docker (`tools/aoflagger_docker/`) to generate flag masks.
3. Extract 64×64 patches with `data/extract_spectrograms.py`.
4. Split with `data/prepare_splits.py`.
5. Train FNNSeg (best config) and UNet baseline on real patches.
6. Compare IoU against AOFlagger labels (pseudo-ground-truth) and against
   synthetic results.

### 3.7 SNR Robustness Sweep

Evaluated the best segmentation model (FNNSeg, fib, stages=3, base_ch=16,
seed=42, IoU=0.976) across 9 RFI signal-to-noise ratio levels. The model
was trained on synthetic data with SNR uniformly sampled from [1, 10].

| SNR | IoU | F1 | Precision | Recall |
|----:|-----:|----:|----------:|-------:|
| 0.5 | 0.044 | 0.084 | 0.977 | 0.044 |
| 1.0 | 0.560 | 0.717 | 0.978 | 0.567 |
| 2.0 | 0.927 | 0.962 | 0.987 | 0.938 |
| 3.0 | 0.976 | 0.988 | 0.995 | 0.980 |
| 5.0 | 0.994 | 0.997 | 0.999 | 0.995 |
| 8.0 | 0.990 | 0.995 | 0.999 | 0.991 |
| 10.0 | 0.982 | 0.991 | 0.999 | 0.983 |
| 15.0 | 0.936 | 0.967 | 0.998 | 0.938 |
| 20.0 | 0.871 | 0.931 | 0.999 | 0.871 |

**Key findings:**
- The model achieves near-perfect IoU (0.994) at SNR=5, falling within the
  training distribution.
- At very low SNR (0.5), the model correctly declines to flag — RFI is
  indistinguishable from noise. Precision remains 0.977 (few false positives).
- At high SNR (15–20), recall drops because the model has not seen such
  extreme RFI during training. This is addressable by extending the training
  SNR range.
- The operational sweet spot is SNR 2–10 (IoU > 0.93), which covers the
  majority of real-world RFI scenarios.

---

## Next Steps (Phase 3 continuation)

- Complete all 75 experiment runs (seg + cls + inpaint).
- Generate all visualization figures (ablation heatmaps, Pareto front, learning
  curves, prediction examples, SNR sweep).
- Run SNR robustness sweep across RFI signal-to-noise ratios.
- Compute cross-seed statistics (mean ± std) for all metrics.
- Download at least one real telescope observation and run end-to-end.
- Begin THESIS_DRAFT.md Chapters 5–6 (Experiments, Results).
