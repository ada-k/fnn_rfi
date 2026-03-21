````markdown
# Progress Log

This file tracks actions performed and next steps for the `fnn_rfi` project.

Completed (local repo changes)
- Added `WORKPLAN.md` — overall 8-week plan and milestones.
- Added `FNN_SPEC.md` — FNN design notes and experiment ideas.
- Added `DATA_SOURCES.md` — public radio-astronomy archives and labeling notes.
- Added `PROPOSAL.md` and `THESIS_DRAFT.md` — skeletons for proposal and thesis.
- Implemented a minimal PyTorch prototype:
  - `models/fnn.py` — FractalBlock, ConvBlock, `FNN` class (Fibonacci-driven widths)
  - `models/__init__.py`
  - `scripts/train_toy.py` — small CIFAR10 smoke-test runner
  - `requirements.txt`
- Added `LIT_REVIEW.md` and `LIT_SUMMARIES.md` for literature tracking.

Environment & running notes
- You indicated you're on Arch Linux on an HP laptop without a GPU.
- Earlier run attempt failed with: ModuleNotFoundError: No module named 'torch'.
- Recommended local run steps (using a Python venv named `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r fnn_rfi/requirements.txt
python3 fnn_rfi/scripts/train_toy.py --batch 64 --subset 256
```

If your venv is elsewhere, replace `.venv/bin/python3` accordingly.

 Next actions (in progress / planned)
 - Fetch and summarise ~20 key papers (FractalNet variants, Fibonacci architecture usage, RFI ML papers).
 - Implement baseline models (CNN, RNN/LSTM, Transformer) with matching parameter budgets.
 - Run synthetic experiments and validate training stability on CIFAR10 and small spectrogram data.
 - Collect labeled RFI datasets and prepare data-prep scripts.

 Recent additions (experiment harness & AOFlagger guidance):
 - Added a lightweight unified training harness:
   - `experiments/utils.py` (model/dataset factories, seeding, checkpoint/config helpers)
   - `experiments/train.py` (CLI runner: pick model/dataset, epochs, batch-size, saves checkpoints)
   These let you run quick CPU experiments locally, e.g.:

   ```bash
   python -m experiments.train --model fnn --dataset cifar --epochs 2 --batch-size 32
   python -m experiments.train --model unet --dataset synthetic --epochs 1 --batch-size 8
   ```

 - Added AOFlagger guidance to `DATA_README.md` (see below) to help obtain and run flagging for label generation.
 - Added AOFlagger guidance to `DATA_README.md` (see below) to help obtain and run flagging for label generation.

Recent additions (AOFlagger container + wrapper):
- Added a reproducible container build under `tools/aoflagger_docker/`:
  - `Dockerfile` — builds `casacore` and `AOFlagger` in Ubuntu 22.04
  - `build_and_run.sh` — helper to build/run the container and mount local data
  - `README.md` — usage notes and run examples
- Added `scripts/aoflagger_wrapper.py` — utility to convert AOFlagger-produced `.npy`/`.npz`/raw binaries into `fnn_rfi/data/labels/` `.npy` masks.

Next immediate actions:
- Build the AOFlagger container locally with `tools/aoflagger_docker/build_and_run.sh build` and run it mounting your data folder.
- Use `scripts/aoflagger_wrapper.py` to convert masks produced by AOFlagger (or place `.npy` masks directly into `fnn_rfi/data/labels/`).
Validation run (completed):
- Ran `experiments.train` smoke test: `fnn` on `cifar` (subset=64, batch=8, epochs=1). Output saved to `runs/fnn_cifar_1766322601` and config stored there.

I logged this run and saved the checkpoint/config in the run folder above.

Notes on reproducibility
- All code includes docstrings and explanatory comments. Keep this file updated with runs and seed values.

Recent run (smoke test)
- Environment: `.venv` Python used when running the script on 2025-12-21.
- Command: `python3 fnn_rfi/scripts/train_toy.py --batch 64 --subset 256`
- Result: CIFAR10 subset training ran successfully (one epoch smoke test). Sample output:
  - `Model params: 17856`
  - `batch 0 loss 2.3172`
  - `avg loss: 2.2989`

Baseline implementations (added)
- `models/cnn.py` — `SimpleCNN` baseline (small conv net)
- `models/rnn.py` — `LSTMClassifier` baseline (reshape images to sequences)
- `models/transformer.py` — `TinyVisionTransformer` (patch-based)
- `scripts/test_models.py` — smoke-test that runs forward passes for FNN + baselines

Smoke test results (forward-only)
- Command: `python3 fnn_rfi/scripts/test_models.py`
- Output summary:
  - `FNN: params=17,856 out_shape=torch.Size([2, 16])`
  - `CNN: params=6,418 out_shape=torch.Size([2, 10])`
  - `RNN(LSTM): params=145,802 out_shape=torch.Size([2, 10])`
  - `Transformer: params=74,954 out_shape=torch.Size([2, 10])`

---

## Phase 3 — Full Experiment Suite (in progress, March 2026)

### Environment
- MacBook Air M4, 10-core CPU, 16GB RAM, macOS
- Python 3.9 (system), PyTorch 2.8.0
- MPS backend disabled (unstable on Python 3.9), all training on CPU
- ~400 GB disk free

### Experiments launched
- **Segmentation**: 14 configs x 3 seeds = 42 runs (FNNSeg ablation + UNet/ResUNet baselines)
- **Classification**: 7 configs x 3 seeds = 21 runs (FNN variants + CNN/RNN/Transformer/ResNet)
- **Inpainting**: 4 configs x 3 seeds = 12 runs (FNNInpaint variants)
- Total: 75 runs, executing via `experiments/run_all_experiments.py`

### Infrastructure added
- `experiments/run_grid.py` — generic experiment grid runner with resume support
- `experiments/visualize.py` — 11 visualization functions for thesis figures
- `experiments/run_all_experiments.py` — master orchestrator for all experiments
- `scripts/count_flops.py` — MAC counter using forward hooks
- `scripts/aggregate_results.py` — cross-seed results aggregation
- `scripts/run_snr_sweep.py` — SNR robustness evaluation
- `scripts/interim_viz.py` — generate visualizations from partial results
- `experiments/utils.py` updated with `get_device()` (CUDA > MPS > CPU)

### Results so far
- FNNSeg (fib, s=3): IoU=0.976 (1 seed), 218K params
- FNNSeg (fib+sep, s=3): IoU=0.964+/-0.009 (3 seeds), 81K params — 2.7x fewer params
- UNet and ResUNet baselines available for comparison
- SNR sweep completed: peak IoU=0.994 at SNR=5, graceful degradation at extremes
- Efficiency profiling: 18 model variants benchmarked (params, MACs, latency, throughput)

### Figures generated
- `figs/spectrogram_samples.png` — synthetic dataset examples
- `figs/fnn_architecture.png` — channel schedule + fractal block diagram
- `figs/seg_predictions_interim.png` — model predictions vs ground truth
- `figs/multi_metric_best.png` — IoU/F1/precision/recall learning curves
- `figs/learning_curves_interim.png` — cross-run comparison
- `figs/snr_sweep.png` — robustness across RFI SNR levels

### Report updated
- `RESEARCH_REPORT.md` Phase 3 section added with efficiency tables, data strategy,
  interim results, and SNR robustness analysis

````
