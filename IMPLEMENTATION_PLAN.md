# Implementation Plan -- FNN for RFI Detection

This plan is derived from a thorough review of: the research proposal (baseline), proposal feedback, and all existing repo code and documentation. It identifies gaps between the proposal and current state, and lays out all work in dependency-ordered phases.

---

## Current State Assessment

**What exists and is usable (keep):**
- `models/fnn.py` -- FractalBlock, ConvBlock, FNN classifier (Fibonacci channel scaling). Functional.
- `models/fnn_seg.py` -- FNNSeg (fractal U-Net encoder-decoder). Functional.
- `models/unet.py` -- UNet baseline. Functional.
- `models/cnn.py` -- SimpleCNN baseline. Functional.
- `models/rnn.py` -- LSTMClassifier baseline. Functional.
- `models/transformer.py` -- TinyVisionTransformer baseline. Functional.
- `experiments/train.py` -- Unified training loop with checkpointing. Functional.
- `experiments/evaluate.py` -- IoU, Dice, precision/recall/F1, PSNR, SSIM. Functional.
- `experiments/utils.py` -- Model/dataset factories, seeding, checkpoint helpers. Functional (except synthetic data import).
- `scripts/benchmark.py` -- Param count + latency table. Functional.
- `scripts/run_all.sh` -- End-to-end runner. Functional.
- `tools/aoflagger_docker/` -- Dockerfile and build/run helper. Can build once Docker is available.
- `scripts/aoflagger_wrapper.py` -- Converts AOFlagger masks to .npy. Functional.
- `benchmark_results.json` -- Existing benchmark snapshot.

**What is missing or broken (must build):**
1. `data/` directory and `data/synthetic_spectrogram.py` -- referenced in `experiments/utils.py` but the directory does not exist. The synthetic RFI generator is the foundation for all training before real data arrives.
2. Real data acquisition and preprocessing pipeline -- no real radio data has been downloaded or processed.
3. Drop-path regularisation -- mentioned in proposal and FNN_SPEC but not implemented in `FractalBlock`.
4. Separable (depthwise-separable) convolution option -- proposal mentions MobileNet/EfficientNet-inspired blocks.
5. Inpainting/restoration model or head -- proposal objective 4 (restoration/inpainting, measure fidelity).
6. ResNet-derived baseline -- proposal lists it as a baseline but it is not implemented.
7. Science-level evaluation: transient injection/recovery (TPR/FPR) and power-spectrum bias assessment.
8. Ablation infrastructure -- controlled experiments over scaling, regularisation, separable convs, channel schedules.
9. Cross-dataset and label-noise robustness tests.
10. FLOPs counting and peak memory profiling.
11. Statistical rigour -- multi-seed runs and 95% CI reporting.
12. Human-validated subset workflow (small manual labeling protocol).
13. Pipeline/architecture diagrams (feedback requirement).

---

## Phase 0: Foundation and Environment (prerequisite for everything)

**Goal:** Working environment, reproducible setup, synthetic data pipeline operational.

### 0.1 Environment setup
- Create/verify virtual environment with `requirements.txt`.
- Add missing dependencies to `requirements.txt` that will be needed later: `matplotlib`, `scipy`, `h5py`, `astropy`, `pandas`, `fvcore` (for FLOPs), `tabulate`.
- Confirm all existing smoke tests pass (`scripts/test_models.py`, `scripts/benchmark.py`).

### 0.2 Create `data/` package and synthetic RFI generator
- Create `data/__init__.py`.
- Create `data/synthetic_spectrogram.py` with `SyntheticSpectrogramDataset`:
  - Input: single-channel 2D spectrogram (time x frequency).
  - Background: Gaussian noise with optional bandpass slope and time-varying gain to mimic real spectrograms.
  - Five RFI morphology types (as listed in README): narrowband, time_burst, broadband, diagonal, rectangle.
  - Controllable parameters: SNR of injected RFI, duty cycle (fraction of pixels contaminated), morphology mixture weights.
  - Output: (spectrogram_tensor, binary_mask_tensor) pairs.
  - Deterministic when seeded.
- Verify `experiments/utils.py` import of `SyntheticSpectrogramDataset` resolves correctly.
- Run `experiments/train.py --model fnn_seg --dataset synthetic --epochs 1 --batch-size 4` as smoke test.

### 0.3 Verify existing models and training
- Run `scripts/test_models.py` to confirm all model forward passes still work.
- Run `scripts/benchmark.py` to regenerate `benchmark_results.json`.
- Run one epoch of segmentation training (FNNSeg, UNet) on synthetic data.
- Run one epoch of classification training (FNN, CNN, RNN, ViT) on CIFAR subset.
- Confirm checkpoint saving/loading round-trips.

**Checkpoint:** All models train, synthetic data generates plausible spectrograms and masks, all scripts run without error.

---

## Phase 1: Model Enhancements (proposal-required architecture work)

**Goal:** Implement all architectural features described in the proposal.

### 1.1 Drop-path regularisation in FractalBlock
- Modify `FractalBlock` in `models/fnn.py`:
  - Add a `drop_path_rate` parameter (default 0.0 = disabled).
  - During training, randomly drop one of the two child paths with probability `drop_path_rate` (the other path is scaled to compensate).
  - During eval, use both paths (standard drop-path semantics from Larsson et al., 2016).
- Propagate the parameter through `FNN` and `FNNSeg` constructors.
- Add `--drop-path` CLI option to `experiments/train.py`.

### 1.2 Separable convolution option
- Add a `ConvBlockSep` variant in `models/fnn.py` that uses depthwise-separable convolutions (depthwise conv + pointwise 1x1 conv) instead of standard convolutions.
- Add a `use_separable` flag to `FractalBlock`, `FNN`, and `FNNSeg` constructors.
- This directly addresses the MobileNet/EfficientNet-inspired efficiency angle from the proposal.

### 1.3 ResNet-derived baseline
- Create `models/resnet_baseline.py`:
  - A small ResNet (e.g. ResNet-18-style) with configurable depth, adapted for spectrogram classification.
  - Match parameter budget roughly to other baselines for fair comparison.
- Register in `models/__init__.py` and `experiments/utils.py` model factory.

### 1.4 Inpainting / restoration head
- Create `models/fnn_inpaint.py`:
  - Extends `FNNSeg` with a restoration head: instead of outputting a binary mask, output a restored spectrogram.
  - Architecture: shared fractal encoder, two decoder heads -- one for mask prediction (segmentation), one for cleaned spectrogram (restoration).
  - Loss: mask head uses BCE; restoration head uses MSE or L1 on the masked regions (only penalize where RFI was injected).
- Alternative simpler approach: train `FNNSeg` to output a 2-channel output (mask + restored value), with a combined loss. Start with this simpler version.
- Register in model factory.
- Add `--task inpaint` option to training script, or create a dedicated `experiments/train_inpaint.py` if the logic diverges significantly.

**Checkpoint:** All models pass forward/backward, ablation flags (drop-path, separable) function, inpainting model produces plausible outputs on synthetic data.

---

## Phase 2: Data Acquisition and Preprocessing

**Goal:** Obtain real radio astronomy data and produce AOFlagger-labelled training sets.

### 2.1 Dataset inventory and download
- Identify 1-2 publicly accessible datasets with manageable size. Priority candidates:
  - **LOFAR**: Public observations via LTA. Select 2-3 short observations with known RFI contamination.
  - **HERA**: Public data releases (hera-team GitHub or Zenodo). Often come with flagging metadata.
  - **MWA**: ASVO archive, public observations.
- Document exact observation IDs, download commands, and license terms in `DATA_SOURCES.md`.
- Download selected observations into `data/raw/` (or external storage if large).

### 2.2 Spectrogram extraction
- Create `data/extract_spectrograms.py`:
  - Read measurement sets or UVFITS files using `casacore` / `astropy`.
  - Extract 2D time-frequency spectrograms (per baseline or averaged) and save as `.npy` arrays.
  - Normalize: subtract bandpass, clip outliers, scale to [0, 1].
  - Output: `data/spectrograms/<dataset>/<obs_id>_<baseline>.npy`.
- Document the extraction parameters (averaging, polarization selection, frequency range).

### 2.3 AOFlagger label generation
- Build AOFlagger Docker container: `tools/aoflagger_docker/build_and_run.sh build`.
- Run AOFlagger on each observation to produce flag masks.
- Use `scripts/aoflagger_wrapper.py` to convert masks to `.npy` format in `data/labels/`.
- Document the AOFlagger strategy/configuration used (default or custom Lua script).

### 2.4 Controlled synthetic RFI injection on real data
- Create `data/inject_rfi.py`:
  - Take a clean (or cleaned) real spectrogram.
  - Inject synthetic RFI with known ground truth (morphology, location, SNR).
  - Output: (contaminated spectrogram, ground truth mask, original clean spectrogram) triples.
  - This provides perfect ground truth for restoration evaluation, complementing the noisy AOFlagger labels.

### 2.5 Dataset splits and versioning
- Create `data/prepare_splits.py`:
  - Split data into train/val/test (e.g. 70/15/15 by observation to avoid data leakage).
  - Save split indices as JSON manifests in `data/splits/`.
  - Version the splits (hash or timestamp).
- Ensure synthetic and real datasets can be loaded through a unified `DataLoader` interface via `experiments/utils.py`.

### 2.6 Human-validated subset
- Select ~50-100 spectrogram patches.
- Create a simple annotation protocol (binary mask review in matplotlib or a minimal labeling tool).
- Save validated masks in `data/human_validated/`.
- Use these to calibrate label quality and estimate AOFlagger error rates.

**Checkpoint:** Reproducible data pipeline; train/val/test splits for at least one real dataset; synthetic injection pipeline operational; data loading confirmed through the experiment harness.

---

## Phase 3: Baseline Experiments

**Goal:** Establish performance baselines for all models on both synthetic and real data.

### 3.1 Synthetic dataset experiments (all models)
Run deterministic training for each model on synthetic data:

| Model | Task | Key config |
|-------|------|------------|
| FNNSeg (fib) | Segmentation | stages=3, base_ch=16, use_fib=True |
| FNNSeg (std) | Segmentation | stages=3, base_ch=16, use_fib=False |
| UNet (matched params) | Segmentation | base_ch tuned to match FNNSeg param count |
| ResNet-derived | Classification of RFI patches | Adapted for spectrogram input |
| SimpleCNN | Classification | Unchanged |
| LSTMClassifier | Classification | Unchanged |
| TinyViT | Classification | Unchanged |

- Train for sufficient epochs (e.g. 30-50) with early stopping.
- Use 3 seeds per experiment; record per-seed and mean +/- 95% CI.
- Log: train loss, val loss, IoU, Dice, F1, precision, recall per epoch.
- Save checkpoints for best val IoU.

### 3.2 Real dataset experiments
- Repeat 3.1 on real data with AOFlagger labels.
- Start with segmentation models (primary task).
- Report same metrics as 3.1.
- Note: classification baselines may need adaptation for real spectrogram patches (not CIFAR).

### 3.3 AOFlagger baseline
- Run AOFlagger on the test split.
- Compute IoU, Dice, F1 against human-validated subset (if available) or the held-out AOFlagger labels from the controlled synthetic injection experiments.
- This establishes the classical baseline that the proposal aims to match or exceed.

### 3.4 Update `scripts/run_all.sh`
- Extend to cover all experiments from 3.1-3.3.
- Parameterize seeds.
- Add output collection and table generation.

**Checkpoint:** Baseline numbers for all models on synthetic and (at least one) real dataset. AOFlagger baseline established. Results reproducible via `run_all.sh`.

---

## Phase 4: Ablation Studies

**Goal:** Systematic ablations to understand the contribution of each FNN design choice.

### 4.1 Ablation matrix
Design the following ablation experiments on FNNSeg (primary model), evaluated on both synthetic and real data:

| Ablation | What changes | What stays |
|----------|-------------|------------|
| No Fibonacci scaling | `use_fib=False` (standard doubling) | Everything else |
| No fractal block (plain conv) | Replace FractalBlock with stacked convs | Same channel schedule |
| Drop-path rate sweep | `drop_path_rate` in {0.0, 0.1, 0.2, 0.3} | FNNSeg fib |
| Separable convolutions | `use_separable=True` | FNNSeg fib |
| Fractal depth sweep | `max_depth` in {1, 2, 3, 4} | FNNSeg fib |
| Number of stages | `stages` in {2, 3, 4} | FNNSeg fib |
| Channel base sweep | `base_ch` in {8, 16, 32} | FNNSeg fib |

### 4.2 Create ablation runner
- Create `scripts/run_ablations.py`:
  - Reads an ablation config (YAML or dict-of-dicts).
  - Spawns training runs for each configuration.
  - Collects results into a single table.
  - Computes deltas relative to the full FNNSeg model.

### 4.3 Analysis
- For each ablation, report: IoU, Dice, F1, params, latency, FLOPs.
- Identify which components contribute most to performance.
- Identify the best efficiency-accuracy trade-off configuration.

**Checkpoint:** Ablation table with clear quantitative evidence for each design choice. Identification of the recommended FNNSeg configuration.

---

## Phase 5: Restoration / Inpainting Experiments

**Goal:** Evaluate RFI restoration (not just detection) as specified in proposal objective 4.

### 5.1 Inpainting training
- Train the inpainting model (from Phase 1.4) on synthetic injection data (Phase 2.4).
- Loss: combined segmentation + restoration loss.
- Baselines: simple interpolation (linear, median-filter) of masked regions; UNet with restoration head.

### 5.2 Restoration metrics
- PSNR and SSIM between restored and original clean spectrogram (on synthetic injection test set).
- Structured residual analysis: compute residuals (restored - clean) and check for bias in time/frequency.
- Already implemented in `experiments/evaluate.py`; extend with residual analysis utilities.

### 5.3 Results
- Table: PSNR, SSIM per model.
- Figures: example spectrograms showing original, contaminated, mask, restored.

**Checkpoint:** Inpainting model trained and evaluated. Restoration metrics computed and compared across methods.

---

## Phase 6: Downstream Science Evaluation

**Goal:** Quantify scientific impact of RFI treatment, as required by proposal objectives.

### 6.1 Transient injection/recovery
- Create `experiments/science_eval.py` or `scripts/transient_recovery.py`:
  - Inject synthetic astronomical transients (e.g. Gaussian pulses of varying widths and SNRs) into clean spectrograms.
  - Apply RFI contamination.
  - Run each detection/restoration method.
  - Attempt to recover the injected transient from the cleaned spectrogram.
  - Measure: True Positive Rate (TPR) and False Positive Rate (FPR) at various detection thresholds.
  - Plot: TPR vs FPR (ROC-style curve) per method.

### 6.2 Power-spectrum bias assessment
- Create `scripts/power_spectrum_analysis.py`:
  - Compute 1D or 2D power spectra of clean, contaminated, and restored spectrograms.
  - Measure bias: difference in power spectrum between restored and clean.
  - Measure variance: scatter across multiple realizations.
  - This directly addresses the EoR (Epoch of Reionization) power-spectrum bias concern from the proposal.

### 6.3 Results
- Tables: TPR/FPR at fixed thresholds per method.
- Figures: ROC curves, power spectrum comparison plots.

**Checkpoint:** Downstream science impact quantified. Evidence for whether FNN restoration reduces or introduces bias relative to baselines.

---

## Phase 7: Efficiency Profiling and Comparison

**Goal:** Complete the efficiency characterisation required by the proposal.

### 7.1 Extend `scripts/benchmark.py`
- Add FLOPs counting (use `fvcore.nn.FlopCountAnalysis` or `torch.profiler`).
- Add peak GPU/CPU memory measurement during training (forward + backward).
- Add throughput (samples/second) during inference.
- Add all new models (ResNet baseline, inpainting model).

### 7.2 Efficiency table
- Columns: Model, Params, FLOPs, Latency (ms), Peak Memory (MB), Throughput (samples/s).
- Compare FNNSeg variants against UNet and ResNet baselines at matched accuracy levels.

**Checkpoint:** Full efficiency table. Evidence for whether FNN achieves better accuracy-efficiency trade-off.

---

## Phase 8: Cross-Dataset and Robustness Tests

**Goal:** Evaluate generalization, as required by the proposal.

### 8.1 Cross-dataset transfer
- Train on dataset A (e.g. LOFAR), test on dataset B (e.g. HERA or MWA).
- Report metric degradation.
- Optionally fine-tune on small subset of target dataset.

### 8.2 Label-noise sensitivity
- Artificially corrupt a fraction of training labels (flip mask pixels).
- Train FNNSeg and UNet under {0%, 5%, 10%, 20%} label noise.
- Measure degradation in val metrics.
- This tests robustness to imperfect AOFlagger labels.

### 8.3 SNR sensitivity
- Evaluate all models on synthetic data with varying RFI SNR levels: {1, 3, 5, 10, 20} dB.
- Plot detection metric vs SNR.

**Checkpoint:** Robustness analysis complete. Understanding of where FNN degrades relative to baselines.

---

## Phase 9: Reporting, Reproducibility, and Packaging

**Goal:** Final deliverables matching the proposal's reproducibility commitments.

### 9.1 Results collection
- Create `scripts/collect_results.py`:
  - Parses all `runs/*/results.json` files.
  - Produces consolidated CSV tables and LaTeX-ready tables.
  - Generates standard plots: learning curves, bar charts, ROC curves.

### 9.2 Figures
- Pipeline/architecture diagram (address feedback): create a clear figure showing data flow from raw data through preprocessing, training, evaluation, to science metrics.
- FNN architecture diagram: show the fractal block structure, Fibonacci channel scaling.
- Result summary figures: comparison bar charts, ablation heatmap, efficiency Pareto plot.

### 9.3 Thesis integration
- Populate `THESIS_DRAFT.md` chapters with results and references.
- Methods chapter: describe FNN architecture, training protocol, data pipeline.
- Results chapter: all tables and figures from Phases 3-8.
- Discussion: interpret ablations, address feedback questions (why fractal over transformers, success thresholds).

### 9.4 Reproducibility packaging
- Verify `requirements.txt` is complete and pinned.
- Ensure `run_all.sh` reproduces all key results.
- Archive final model checkpoints.
- Document exact seeds, split hashes, AOFlagger versions.
- Container recipes updated and tested.

### 9.5 Address proposal feedback
- Add conceptual framework diagram (9.2).
- Define explicit success thresholds (e.g. FNNSeg IoU within 2% of UNet at <50% params).
- Add explicit ethical/data governance section to thesis (data licenses, no PII in radio data, responsible use of compute).
- Sharpen architecture comparison (why fractal over transformers: interpretable multi-scale, subnetwork extraction, controlled parameter growth vs attention compute cost).
- Deduplicate reference URLs.

**Checkpoint:** All deliverables complete. Thesis draft with results. Reproducible artifact package.

---

## Dependency Graph

```
Phase 0 (foundation)
  |
  +---> Phase 1 (model enhancements) --+
  |                                     |
  +---> Phase 2 (data acquisition) -----+
                                        |
                                        v
                                   Phase 3 (baselines)
                                        |
                              +---------+---------+
                              |         |         |
                              v         v         v
                         Phase 4    Phase 5    Phase 7
                        (ablation) (inpaint)  (efficiency)
                              |         |         |
                              +---------+---------+
                                        |
                                        v
                                   Phase 6 (science eval)
                                        |
                                        v
                                   Phase 8 (robustness)
                                        |
                                        v
                                   Phase 9 (reporting)
```

Phases 1 and 2 can run in parallel after Phase 0.
Phases 4, 5, and 7 can run in parallel after Phase 3.
Phase 6 depends on having trained models from 3-5.
Phase 8 depends on trained models and multiple datasets.
Phase 9 is final integration.

---

## Task Priority Summary

| Priority | Phase | Reason |
|----------|-------|--------|
| Critical | 0.2 | Synthetic data generator is missing; blocks all training |
| Critical | 0.1, 0.3 | Environment and smoke tests; blocks everything |
| High | 1.1, 1.2 | Drop-path and separable convs are core proposal claims |
| High | 2.1-2.3 | Real data needed for credible results |
| High | 3.1-3.3 | Baseline numbers needed before any ablation or comparison |
| High | 1.4 | Inpainting is a distinct proposal objective |
| Medium | 1.3 | ResNet baseline is useful but not the primary contribution |
| Medium | 4.1-4.3 | Ablations are essential for the thesis but depend on baselines |
| Medium | 5.1-5.3 | Restoration experiments depend on inpainting model |
| Medium | 6.1-6.2 | Science eval is a key differentiator but depends on models |
| Medium | 7.1-7.2 | Efficiency profiling extends existing benchmark |
| Lower | 8.1-8.3 | Robustness tests are thorough but secondary |
| Lower | 2.6 | Human validation is ideal but labor-intensive |
| Final | 9.x | Reporting is last, depends on all results |

---

## File Creation / Modification Summary

**New files to create:**
- `data/__init__.py`
- `data/synthetic_spectrogram.py`
- `data/extract_spectrograms.py`
- `data/inject_rfi.py`
- `data/prepare_splits.py`
- `models/resnet_baseline.py`
- `models/fnn_inpaint.py`
- `scripts/run_ablations.py`
- `scripts/collect_results.py`
- `scripts/transient_recovery.py`
- `scripts/power_spectrum_analysis.py`
- `experiments/science_eval.py` (or merge into evaluate.py)

**Existing files to modify:**
- `models/fnn.py` -- add drop-path, separable conv option
- `models/fnn_seg.py` -- propagate drop-path, separable conv flags
- `models/__init__.py` -- register new models
- `experiments/utils.py` -- register new models and datasets in factories
- `experiments/train.py` -- add CLI flags for drop-path, separable, inpainting task
- `experiments/evaluate.py` -- add residual analysis, FLOPs, memory utilities
- `scripts/benchmark.py` -- add FLOPs, memory, new models
- `scripts/run_all.sh` -- extend to cover all experiments
- `requirements.txt` -- add new dependencies

**Existing files to keep as-is:**
- `models/unet.py`, `models/cnn.py`, `models/rnn.py`, `models/transformer.py` -- functional baselines
- `scripts/aoflagger_wrapper.py` -- functional
- `scripts/test_models.py`, `scripts/train_toy.py`, `scripts/train_seg.py` -- smoke tests
- `tools/aoflagger_docker/*` -- functional container setup
- All documentation files (update content in Phase 9)
