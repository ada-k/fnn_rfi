# fnn_rfi — Fractal Neural Networks for Radio-Frequency Interference Detection

A research codebase investigating whether compact, self-similar Fractal/Fibonacci Neural Networks (FNNs) can match or exceed standard deep-learning baselines for RFI detection and restoration on two-dimensional time–frequency spectrograms.

## Project overview

Radio-frequency interference (RFI) degrades sensitivity across modern radio telescopes.  This project:

1. Implements an FNN family (fractal depth + Fibonacci channel scaling) for 2D spectrogram segmentation and classification.
2. Benchmarks it against established baselines: U-Net, ResNet-derived classifiers, MobileNet/EfficientNet-inspired blocks, and a compact Vision Transformer.
3. Uses AOFlagger (containerised via Docker) to produce operational baseline masks from real data, and a synthetic injection pipeline for controlled ground-truth experiments.
4. Evaluates both pixel-level metrics (IoU, Dice, F1) and downstream scientific impact (transient recovery, power-spectrum bias).

---

## Repository structure

```
fnn_rfi/
├── data/
│   └── synthetic_spectrogram.py   # On-the-fly RFI injection dataset (5 morphology types)
├── models/
│   ├── fnn.py                     # FNN classifier (fractal blocks + Fibonacci channels)
│   ├── fnn_seg.py                 # FNNSeg: fractal U-Net for segmentation
│   ├── unet.py                    # Standard U-Net baseline
│   ├── cnn.py                     # SimpleCNN baseline
│   ├── rnn.py                     # LSTM baseline
│   └── transformer.py             # Tiny Vision Transformer baseline
├── experiments/
│   ├── train.py                   # Unified training script (train + val + metrics)
│   ├── evaluate.py                # Metrics: IoU, Dice, F1, PSNR, SSIM
│   └── utils.py                   # Model factory, dataloader factory, seeding
├── scripts/
│   ├── benchmark.py               # Parameter counts + CPU latency comparison table
│   ├── run_all.sh                 # End-to-end experiment runner
│   ├── aoflagger_wrapper.py       # Converts AOFlagger outputs to training masks
│   ├── test_models.py             # Quick forward-pass smoke test
│   ├── train_seg.py               # Segmentation training helper
│   └── train_toy.py               # Toy training helper
└── tools/
    └── aoflagger_docker/          # Dockerfile + helper to build AOFlagger in a container
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `tqdm`, `scikit-image`.

---

## Quick start

### Smoke test — verify all models run
```bash
cd fnn_rfi
python scripts/test_models.py
```

### Train FNNSeg on the synthetic dataset
```bash
cd fnn_rfi
python -m experiments.train --model fnn_seg --dataset synthetic --epochs 10 --batch-size 16
```

### Train U-Net baseline
```bash
python -m experiments.train --model unet --dataset synthetic --epochs 10 --batch-size 16
```

### Classification smoke (CIFAR-10 subset)
```bash
python -m experiments.train --model fnn --dataset cifar --epochs 5 --subset 1024
```

### Run the full parameter + latency benchmark
```bash
python scripts/benchmark.py
```

### Run all experiments in sequence
```bash
bash scripts/run_all.sh
```

Outputs land in `runs/<model>_<dataset>_<timestamp>/` and include:
- `config.json` — full hyperparameter record
- `ckpt_epoch<N>.pth` — checkpoint per epoch
- `results.json` — loss + metrics history

---

## Models

| Model | Task | Key idea |
|---|---|---|
| `FNN` | Classification | Fractal depth + Fibonacci channel scaling |
| `FNNSeg` | Segmentation | Fractal U-Net; same scaling law applied to encoder/decoder |
| `UNet` | Segmentation | Standard U-Net baseline |
| `SimpleCNN` | Classification | 3-layer conv baseline |
| `LSTMClassifier` | Classification | Row-as-timestep LSTM |
| `TinyVisionTransformer` | Classification | Patch-based ViT (compact) |

---

## Evaluation metrics

Segmentation: **IoU**, **Dice**, **precision**, **recall**, **F1**  
Restoration: **PSNR**, **SSIM** (scikit-image; pure-tensor fallback if unavailable)

All metrics are implemented in `experiments/evaluate.py` and logged automatically during training.

---

## Synthetic RFI types

The dataset generator (`data/synthetic_spectrogram.py`) injects five morphological types:

- **narrowband** — persistent CW-like emission in 1–3 frequency channels
- **time_burst** — broadband short-duration pulse
- **broadband** — wide-frequency moderate-duration emission
- **diagonal** — linearly drifting streak (satellite sweep / Doppler)
- **rectangle** — arbitrary rectangular RFI patch

---

## AOFlagger integration

AOFlagger is used to produce operational baseline masks from real spectrogram data.  A Dockerised build is provided to avoid host toolchain issues:

```bash
cd tools/aoflagger_docker
bash build_and_run.sh
```

Once masks are produced, convert them to `.npy` training labels with:

```bash
python scripts/aoflagger_wrapper.py path/to/mask.npy --out data/labels/
```

---

## References

Key literature cited in the research proposal (full APA list in `docs/Research_Proposal.docx`):

- Offringa et al. (2010, 2015, 2023) — AOFlagger methodology and operational evaluations  
- Akeret et al. (2017) — CNN/U-Net for RFI segmentation  
- Vafaei Sadr et al. (2020) — ResNet-based detection on MeerKAT  
- Yang et al. (2020) — residual detection for FAST  
- Larsson et al. (2016) — FractalNet architecture  
- Tan & Le (2019) — EfficientNet scaling  
- Howard et al. (2017) — MobileNet efficiency blocks  
- Gu et al. (2024) — multi-scale attention U-Net for RFI  
- Zhang et al. (2024) — joint detection + restoration (RFI-DRUnet)  
- Pagano et al. (2023) — inpainting residuals and EoR impact  
