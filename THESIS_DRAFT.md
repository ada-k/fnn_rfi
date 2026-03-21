# Thesis Draft: Fractal Neural Networks for Radio Frequency Interference Detection and Mitigation

## Chapter 1 — Introduction

### 1.1 Problem Statement
Radio frequency interference (RFI) is a persistent challenge in radio astronomy,
contaminating scientific observations across all modern telescopes (LOFAR, HERA,
MWA, SKA precursors). As telescope sensitivity and data volumes grow exponentially,
manual flagging becomes impractical, and classical algorithms (e.g., AOFlagger)
struggle with evolving RFI environments. Machine learning approaches offer
promising automation but often require architectures with millions of parameters
and extensive labelled data.

### 1.2 Contributions
This thesis proposes the **Fractal Neural Network (FNN)**, a novel architecture
that uses Fibonacci-scaled channel widths and recursive fractal blocks for
efficient parameter usage. Key contributions:

1. **FNN architecture** with Fibonacci channel scheduling and self-similar fractal
   blocks, demonstrating competitive segmentation performance with significantly
   fewer parameters than U-Net baselines.
2. **Separable fractal convolutions** that reduce FNN parameters by 2.7× with
   minimal accuracy loss.
3. **Joint RFI segmentation and spectral restoration** via FNNInpaint, a dual-head
   architecture that simultaneously detects and removes RFI.
4. **Comprehensive evaluation** on synthetic data with ablation studies, efficiency
   profiling, and robustness analysis across RFI signal-to-noise ratios.

---

## Chapter 3 — Methods

### 3.1 Fractal Block Architecture
The FractalBlock is a self-similar computational unit with configurable recursion
depth. At depth `d`, a block consists of two parallel paths:
- **Left path**: a `ConvBlock` (or `ConvBlockSep` for the separable variant)
  that applies two 3×3 convolutions with BatchNorm and ReLU.
- **Right path**: two sub-FractalBlocks at depth `d-1`, producing diverse
  feature representations at different scales.

At the base case (depth 1), both paths are simple `ConvBlock`s. Path outputs are
merged via a learned 1×1 convolution, enabling the network to learn the optimal
combination of shallow and deep features.

### 3.2 Fibonacci Channel Scheduling
Standard encoder-decoder networks double channels at each downsampling stage
(16→32→64→128...), leading to rapid parameter growth. FNN uses the Fibonacci
sequence (1, 1, 2, 3, 5, 8, 13...) as channel multipliers:
- Stage 1: `base_ch × 1` = 16
- Stage 2: `base_ch × 1` = 16
- Stage 3: `base_ch × 2` = 32

This sublinear growth produces lighter models (218K vs 345K params for 3-stage
segmentation) while maintaining representational capacity through the fractal
block's parallel paths.

### 3.3 Stochastic Depth (DropPath)
DropPath regularisation randomly drops entire fractal block branches during
training, forcing the network to maintain useful representations at every depth
level. Each branch is independently dropped with probability `drop_path_rate`,
with outputs scaled by the survival probability during inference.

### 3.4 Depthwise-Separable Convolutions
`ConvBlockSep` replaces standard 3×3 convolutions with depthwise-separable
convolutions (depthwise conv → pointwise conv), reducing the parameter count
per convolutional layer by approximately `k²` (≈9× for 3×3 kernels). When
applied to FNNSeg, this yields 81K params vs 218K — a 2.7× reduction.

### 3.5 FNNSeg — Segmentation Architecture
FNNSeg follows a U-Net-like encoder-decoder structure:
- **Encoder**: `stages` fractal blocks with 2×2 max-pooling between stages.
- **Bottleneck**: A fractal block at the deepest resolution.
- **Decoder**: Transposed convolution upsampling with skip connections from the
  encoder, followed by fractal blocks at each resolution level.
- **Head**: A 1×1 convolution producing per-pixel binary logit (RFI / clean).

### 3.6 FNNInpaint — Joint Detection and Restoration
FNNInpaint shares the encoder and bottleneck with FNNSeg but adds a second
decoder for spectral restoration:
- **Segmentation decoder** → binary mask logit.
- **Restoration decoder** → reconstructed clean spectrogram.
- **Loss**: BCE on the mask + masked MSE on the restored spectrogram (only
  computing reconstruction loss over contaminated pixels).

### 3.7 Evaluation Metrics
- **Segmentation**: IoU (Jaccard), Dice/F1, pixel precision, pixel recall.
- **Restoration**: PSNR (dB), SSIM (structural similarity).
- **Efficiency**: parameter count, MACs (multiply-accumulate operations),
  CPU inference latency (ms), throughput (samples/s).

---

## Chapter 4 — Data

### 4.1 Synthetic Dataset
The `SyntheticSpectrogramDataset` generates 64×64 single-channel spectrograms
with controlled RFI injection. Five morphologies are supported:
1. **Narrowband**: vertical bands (persistent interference at fixed frequencies).
2. **Time burst**: horizontal bands (transient broadband events).
3. **Broadband**: wide-band sustained interference.
4. **Diagonal**: chirp-like interference sweeping across frequency with time.
5. **Rectangle**: localized patches of contamination.

Each sample is generated deterministically from `seed + index`, ensuring
reproducibility. RFI signal-to-noise ratio is sampled uniformly from a
configurable range (default [1, 10]).

### 4.2 Real Data Strategy
Real telescope data from LOFAR (Long Term Archive), HERA (IDR2), and MWA
(ASVO) archives is available for download. AOFlagger flag masks serve as
pseudo-ground-truth labels. The data pipeline (`data/extract_spectrograms.py`)
supports FITS, NPY, and Measurement Set formats with sigma-clip preprocessing
and sliding-window patch extraction.

---

## Chapter 5 — Experiments

### 5.1 Experimental Setup
All experiments run on a MacBook Air M4 (10-core CPU, 16GB RAM) using
PyTorch 2.8.0 on CPU. Segmentation uses Adam (lr=1e-3), batch_size=16,
15 epochs, 1024 train / 256 val samples. Classification uses batch_size=32,
10 epochs on CIFAR-10.

### 5.2 Ablation Design
14 segmentation configurations test:
- Fibonacci vs standard channel scheduling
- Separable vs standard convolutions
- Drop-path rates: {0, 0.05, 0.1, 0.2}
- Network depth (stages): {2, 3, 4}
- Network width (base_ch): {8, 16, 32}

Each configuration runs with 3 seeds (42, 123, 7) for variance estimation.

### 5.3 Baselines
- **UNet** (depth=3, base_ch=16): established segmentation baseline.
- **ResUNet** (stages=3, base_ch=16): residual U-Net with pre-activation blocks.
- **SimpleCNN, LSTM, TinyViT, ResNet**: classification baselines on CIFAR-10.

---

## Chapter 6 — Results

*(Results tables will be populated as experiments complete. See
`RESEARCH_REPORT.md` for interim results.)*

---

## Chapter 7 — Discussion

### Limitations
- All experiments on synthetic data; real telescope data evaluation pending.
- CPU-only training limits dataset scale and epoch count.
- Synthetic RFI morphologies are simplified compared to real-world interference.

### Future Work
- Evaluate on real LOFAR/HERA/MWA observations with AOFlagger labels.
- Scale to larger spectrograms (256×256, 512×512) with GPU training.
- Explore semi-supervised learning to leverage unlabelled telescope data.
- Integrate FNNInpaint into existing radio astronomy pipelines (CASA, DP3).

---

## Chapter 8 — Conclusion

The Fractal Neural Network demonstrates that biologically-inspired Fibonacci
channel scheduling combined with self-similar fractal blocks can achieve
competitive RFI segmentation performance (IoU > 0.97) with significantly
fewer parameters than conventional architectures. The separable variant
(FNNSeg-sep) reduces parameters by 2.7× while maintaining IoU > 0.96.

Notes
- Keep figures and code references versioned in the repo; save experiment config files alongside results.
- I will populate the Methods and Experiments sections as we implement models and run experiments.
