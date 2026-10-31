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
├── LICENSE                        # Apache-2.0
├── README.md
├── requirements.txt
├── __init__.py
├── benchmark_results.json         # Parameter-count + CPU-latency benchmark output (scripts/benchmark.py)
├── doc/
│   ├── PAPER_ARXIV.md             # Condensed paper (Markdown source)
│   ├── PAPER_ARXIV.tex            # Condensed paper (LaTeX, arXiv-ready)
│   ├── PAPER_ARXIV.html           # Condensed paper (HTML, for easy local preview)
│   ├── THESIS_DRAFT_v2.md         # Full UEL dissertation draft
│   └── ada_kibet_research_proposal-2.pdf
├── data/
│   ├── synthetic_spectrogram.py   # On-the-fly RFI injection dataset (5 morphology types)
│   ├── inject_rfi.py              # Per-morphology injection functions
│   ├── hera_dataset.py            # HERA-PP/HERA-AOF dataset loader
│   ├── extract_spectrograms.py    # Raw-archive to spectrogram extraction
│   └── prepare_splits.py          # Train/val/test split construction
├── models/
│   ├── fnn.py                     # FNN classifier (fractal blocks + Fibonacci channels)
│   ├── fnn_seg.py                 # FNNSeg: fractal U-Net for segmentation
│   ├── fnn_inpaint.py             # FNNInpaint: dual-head detection + restoration variant
│   ├── unet.py                    # Standard U-Net baseline
│   ├── resnet_baseline.py         # ResU-Net / ResNet-classifier baselines
│   ├── cnn.py                     # SimpleCNN baseline
│   ├── rnn.py                     # LSTM baseline
│   └── transformer.py             # Tiny Vision Transformer baseline
├── experiments/
│   ├── train.py                   # Unified training script (train + val + metrics)
│   ├── train_inpaint.py           # FNNInpaint training loop
│   ├── run_all_experiments.py     # Experiment-grid driver
│   ├── run_grid.py                # Hyperparameter grid runner
│   ├── evaluate.py                # Metrics: IoU, Dice, F1, PSNR, SSIM
│   ├── visualize.py                # Plotting helpers
│   └── utils.py                   # Model factory, dataloader factory, seeding
├── scripts/
│   ├── train_hera.py              # Real-telescope (HERA/LOFAR) training entrypoint
│   ├── run_hera_experiments.sh    # Full 27-run HERA/LOFAR experiment driver
│   ├── run_snr_sweep.py           # Synthetic-data SNR robustness sweep
│   ├── aggregate_results.py       # Collects per-run results.json into aggregated_results.json
│   ├── eval_hera.py               # Evaluation on held-out HERA/LOFAR test sets
│   ├── count_flops.py             # FLOPs/parameter-count accounting
│   ├── benchmark.py               # Parameter counts + CPU latency comparison table
│   ├── extract_benchmarks.py      # Parses du Toit et al. (2023) benchmark CSV
│   ├── generate_arch_diagrams.py  # Architecture diagram figures
│   ├── generate_raster_plots.py   # Result bar-chart/plot figures
│   ├── generate_thesis_plots.py   # Full thesis figure set (figs/thesis/)
│   ├── pareto_plot.py             # Parameter-efficiency Pareto plots
│   ├── rerun_failed.py            # Re-runs any failed/incomplete experiment configs
│   ├── aoflagger_wrapper.py       # Converts AOFlagger outputs to training masks
│   ├── test_models.py             # Quick forward-pass smoke test
│   ├── test_hera_loader.py        # Dataset-loader smoke test
│   ├── train_seg.py               # Segmentation training helper
│   ├── train_toy.py               # Toy training helper
│   ├── interim_viz.py             # Interim/in-progress visualisation (superseded by generate_thesis_plots.py)
│   └── run_all.sh                 # End-to-end experiment runner
├── figs/                          # Generated figures (synthetic-data ablation/classification/inpainting), figs/thesis/ for the real-data study
├── results/hera/                  # Per-run results.json for all 27 runs; best_model.pt checkpoints for 24/27 (the 3 CPU-trained HERA-PP seed-0 runs were not checkpointed, see Section 3.7 of the paper)
├── logs/hera/                     # Per-run training logs
└── tools/
    └── aoflagger_docker/          # Dockerfile + helper to build AOFlagger in a container
```

Scripts under `scripts/` that write figures (`generate_thesis_plots.py`, `generate_arch_diagrams.py`, `generate_raster_plots.py`) and benchmark JSONs (`eval_hera.py`, `extract_benchmarks.py`) are written to be run with `fnn_rfi/` as a path prefix, i.e. from the parent directory of this repository (so that, for example, `generate_thesis_plots.py` writes to `fnn_rfi/figs/thesis/`). Running them from inside the repository root itself will create a duplicate nested `fnn_rfi/figs/` directory — if you see one, it's a stray artifact of running from the wrong working directory and can be deleted.

The condensed paper (`doc/PAPER_ARXIV.*`) is the arXiv-submission version of this work; `doc/THESIS_DRAFT_v2.md` is the full UEL dissertation it is derived from. An arXiv-ready submission bundle (`main.tex` + all referenced figures) can be regenerated at any time from `doc/PAPER_ARXIV.tex` and is not committed to version control (see `.gitignore`).

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

Key literature cited in the research proposal (full APA list in `doc/ada_kibet_research_proposal-2.pdf`; full reference list for the paper itself in `doc/PAPER_ARXIV.md`):

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
