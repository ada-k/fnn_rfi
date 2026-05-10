# Research Strategy: Two-Track Structure & Thesis Roadmap

**Date:** 3 May 2026  
**Status:** Active planning document  

---

## The Two Contributions

This thesis makes two distinct contributions. They are presented in one document for the thesis, but are separable for future journal submission.

---

### Contribution 1 — FNN: A Fractal Neural Network Architecture

**Core claim:** Fibonacci channel scheduling applied to a fractal encoder-decoder produces a parameter-efficient model with competitive accuracy and strong anytime-depth properties.

**What to show:**
- The architectural novelty: fractal block expansion rule, Fibonacci scheduling, drop-path.
- Ablation: fib vs. uniform channels, depth effect, drop-path sensitivity, separable convolutions.
- Efficiency Pareto: params vs. IoU, FLOPs vs. IoU — FNNSeg dominates or matches ResUNet/UNet at far lower param count.
- Anytime-depth: earlier exits preserve most accuracy (useful for deployment at different compute budgets).
- Generalisation: tested on both synthetic spectrograms and real HERA/LOFAR data.

**Evidence already in-hand (synthetic experiments):**
- FNNSeg (fib, s=3, ch=16): IoU ≈ 0.976, 218K params
- FNNSeg (fib+sep): IoU ≈ 0.972, 82K params
- UNet (ch=16, d=3): IoU ≈ 0.971, 482K params
- ResUNet (s=3, ch=16): IoU ≈ 0.974, 1.6M params
- FNNSeg achieves comparable IoU with 2–7× fewer parameters → Pareto efficiency claim

**Paper title (if split):** "FNN: A Fractal Neural Network with Fibonacci Channel Scheduling for Efficient Visual Segmentation"  
**Target venue:** ICLR / NeurIPS (architecture track) or IEEE Signal Processing Letters

---

### Contribution 2 — RFI Detection with FNN

**Core claim:** FNNSeg applied to real radio-astronomy data (HERA + LOFAR) achieves competitive detection performance against established methods while using significantly fewer parameters.

**What to show:**
- Real-world performance on HERA (simulated, pixel-perfect labels) and LOFAR (real, hand-labeled).
- Direct comparison against du Toit et al. benchmark numbers (results_all_models.csv published at Zenodo 8275061) — same datasets, same evaluation protocol, directly comparable.
- Cross-dataset generalisation: train on HERA, test on LOFAR.
- SNR robustness (already done on synthetic; port to real data).
- Comparison with AOFlagger baseline.

**Key reference to beat/match:** du Toit et al. (Zenodo 8275061) — the only published systematic comparison on exactly this dataset.

**Paper title (if split):** "Fractal Neural Networks for Radio Frequency Interference Detection in Radio Astronomy: Benchmarking on HERA and LOFAR"  
**Target venue:** MNRAS / A&A / URSI GASS / IAU symposium / IEEE TAES

---

## What NOT to Include (Simplification)

The feedback from the proposal was to sharpen the focus. The following items either distract from the two core contributions or dilute the story:

- **Classification task (CIFAR-10):** Results are weak (best acc ~0.39) because CIFAR-10 is not the target domain. Include only in the FNN architecture section as a sanity-check baseline, not as a headline result. Explicitly state this limitation.
- **Inpainting task:** Include briefly as a secondary capability of FNN (dual-head variant). Don't lead with it.
- **Too many model variants:** Focus on 3–4 key FNNSeg variants in paper. Full ablation table in appendix.
- **RNN / LSTM baselines for segmentation:** These don't make sense for 2D segmentation. Include only in the classification section if at all.

---

## Thesis Structure (Single Document)

```
Chapter 1  Introduction
    1.1  The RFI problem in radio astronomy
    1.2  Why deep learning for RFI detection
    1.3  The fractal architecture opportunity
    1.4  Contributions (two clearly stated)
    1.5  Thesis structure

Chapter 2  Background & Literature Review
    2.1  Radio frequency interference — sources and impact
    2.2  Classical RFI mitigation (AOFlagger, thresholding)
    2.3  Deep learning for RFI detection
         - Vafaei Sadr et al. (CNN baseline)
         - Gu et al. (attention UNet)
         - Mesarcik et al. (NLN)
         - du Toit et al. (comparison framework) ← CRITICAL, currently missing
    2.4  Fractal and recursive neural architectures
         - FractalNet (Larsson et al.)
         - Fibonacci sequences in ML
    2.5  Encoder-decoder segmentation networks (UNet family)
    2.6  Transformer architectures for segmentation (why not chosen?)

Chapter 3  Methodology — FNN Architecture (Contribution 1)
    3.1  Fractal block definition
    3.2  Fibonacci channel scheduling
    3.3  Drop-path regularisation
    3.4  Anytime-depth property
    3.5  Depthwise-separable variant (FNNSeg-sep)
    3.6  FNNInpaint: dual-head extension

Chapter 4  Methodology — Real-Data Pipeline (Contribution 2)
    4.1  Dataset description: HERA and LOFAR pickles
    4.2  Preprocessing: log-amplitude, phase, normalisation, patching
    4.3  Label quality: simulation masks vs. AOFlagger masks
    4.4  Training protocol: loss, optimiser, augmentation
    4.5  Evaluation protocol (matching du Toit et al.)

Chapter 5  Experiments — Architecture Validation (Synthetic)
    5.1  Synthetic spectrogram generation
    5.2  Ablation study: Fibonacci vs. uniform channels
    5.3  Ablation: depth (max_depth = 2, 3, 4)
    5.4  Ablation: drop-path rate
    5.5  Comparison with baselines (UNet, ResUNet, CNN)
    5.6  Parameter efficiency (Pareto plot)
    5.7  SNR robustness

Chapter 6  Experiments — RFI Detection (Real Data)
    6.1  HERA experiments (same label conditions as du Toit et al.)
    6.2  HERA-AOF experiments (AOFlagger labels)
    6.3  LOFAR experiments
    6.4  Cross-dataset generalisation (HERA → LOFAR)
    6.5  Comparison with published results (du Toit et al. benchmark)

Chapter 7  Results & Discussion
    7.1  FNN architecture efficiency vs. accuracy
    7.2  RFI detection against state of the art
    7.3  Limitations: CIFAR-10 classification, inpainting
    7.4  Ethical considerations (data governance, deployment risks)
    7.5  Why fractal over transformers (answer proposal feedback)

Chapter 8  Conclusion
    8.1  Summary of contributions
    8.2  Future work: production deployment, cross-telescope generalisation
    8.3  Journal paper split plan

Appendix A  Full ablation tables
Appendix B  Hyper-parameter grids
Appendix C  Reproducibility checklist
```

---

## Immediate Next Steps (Ordered by Priority)

### Step 1: Validate the real-data loader (1–2 hours)
```bash
cd /Users/ada/uel
python3 fnn_rfi/scripts/test_hera_loader.py
```
Confirm HERADataset produces correct tensor shapes and mask values from the HERA pickle.

### Step 2: Run real-data pilot experiment (3–4 hours, CPU-bound)
Train FNNSeg (best synthetic variant: fib, s=3, ch=16) on HERA_28-03-2023_all.pkl.
Patch size 256×256, stride 128 → ~(1120-100)×9 ≈ 9180 train patches.  
Use a small subset first (256 patches) to check the pipeline before a full run.

```bash
python3 fnn_rfi/scripts/train_hera.py \
    --pkl data/zenodo/HERA_28-03-2023_all.pkl \
    --model fnn_seg --use_fib --patch 256 --epochs 30 --batch 8
```

### Step 3: Run all baselines on HERA (1–2 days, can background)
- FNNSeg (fib, s=3, ch=16) — 3 seeds
- FNNSeg (fib+sep) — 3 seeds
- UNet (ch=16, depth=3) — 3 seeds
- ResUNet (ch=16) — 3 seeds
*(This gives the comparison table matching du Toit et al.)*

### Step 4: Extract comparable numbers from results_all_models.csv
Parse `data/zenodo/results_all_models.csv` to extract test accuracy, F1, AUC for the best UNet models in du Toit's benchmark. These become the "published baseline" row in our comparison table.

### Step 5: Cross-dataset run (train HERA → test LOFAR)
Load LOFAR_Full_RFI_dataset.pkl as test-only; use HERA model to generate predictions.  
Report degradation vs. HERA-trained HERA-tested → measures domain gap.

### Step 6: Proposal gap analysis (2–3 hours writing)
Re-read the proposal and proposal_feedback.md.  
Create `PROPOSAL_GAP_ANALYSIS.md` mapping each feedback point to:
- What we have now that addresses it
- What still needs to be written in the report

Feedback points to address:
1. "No conceptual framework diagram" → create pipeline figure (fig: rfi_pipeline.png)
2. "No explicit success criteria" → state: FNNSeg ≥ UNet accuracy at ≤ 50% params
3. "Ethical/data governance explicit discussion" → add Chapter 7.4
4. "Why fractal over transformers" → add justification in Chapter 7.5
5. "Repeated URLs" → fix in reference list

### Step 7: Write report (1–2 weeks)
Working draft order (write the easy parts first):
1. Chapter 4 (data) — now have exact numbers
2. Chapter 5 (synthetic experiments) — results in hand
3. Chapter 3 (FNN architecture) — spec exists in FNN_SPEC.md
4. Chapter 6 (real-data experiments) — after Step 3
5. Chapter 7 (discussion) — must wait for all results
6. Chapter 2 (literature review) — fill in du Toit + others
7. Chapter 1 (introduction) — write last with full results in mind
8. Abstract — very last

---

## Comparison Table Template (Real Data)

The following is the target table for Chapter 6 (to be filled in after real-data runs):

| Model | #Params | FLOPs | Test Acc (HERA) | F1 (HERA) | AUC (HERA) | Test Acc (LOFAR) |
|-------|---------|-------|-----------------|------------|------------|-----------------|
| UNet (du Toit, 16f) | ~482K | — | — | — | — | — |
| UNet (du Toit, 32f) | — | — | — | — | — | — |
| Best model (du Toit) | — | — | ~0.997 | — | — | — |
| **FNNSeg-fib (ours)** | **218K** | — | TBD | TBD | TBD | TBD |
| **FNNSeg-sep (ours)** | **82K** | — | TBD | TBD | TBD | TBD |
| UNet-baseline (ours) | 482K | — | TBD | TBD | TBD | TBD |
| ResUNet-baseline (ours) | 1.6M | — | TBD | TBD | TBD | TBD |

Fill du Toit numbers from `data/zenodo/results_all_models.csv`.

---

## Key Narrative Points for Paper

1. **Efficiency vs. accuracy:** FNNSeg achieves near-UNet accuracy (IoU ~0.97) with 2–7× fewer parameters. This matters for telescope deployments where compute is constrained (SKA edge nodes, real-time flagging).

2. **Fractal justification:** Fractal blocks inherently provide multi-scale feature aggregation without attention mechanisms. Fibonacci scheduling balances channel capacity across scales without wasteful uniform allocation. Unlike transformers, FNN scales to 512×512 patches without quadratic attention cost.

3. **Real-world validation:** Previous fractal/FractalNet work only evaluated on CIFAR/ImageNet. This is the first application of a fractal architecture to real radio-astronomy data, demonstrating domain-specific effectiveness.

4. **Reproducibility:** All datasets are public (Zenodo CC-BY), code is open-source, exact train/test splits and seeds reported. This addresses the proposal feedback on missing success criteria and reproducibility.

5. **Ethical note:** HERA/LOFAR data is fully non-personal (electromagnetic spectra from space). No sensitive personal data. AOFlagger labels introduce label noise — we acknowledge and quantify this as a limitation.

---

## Journal Split Plan (Post-Thesis)

If splitting into two papers post-thesis:

**Paper 1 (Architecture):**  
- Venue: IEEE Transactions on Neural Networks and Learning Systems (TNNLS) or ICLR workshop
- Content: Fractal block, Fibonacci scheduling, ablations on synthetic + CIFAR-10, Pareto plots
- Key figure: FNN architecture diagram + efficiency Pareto frontier

**Paper 2 (RFI Application):**  
- Venue: Monthly Notices of the Royal Astronomical Society (MNRAS) or A&A
- Content: Real HERA/LOFAR results, comparison with du Toit benchmark, cross-dataset generalisation, deployment discussion
- Key figure: Comparison table + example predictions on LOFAR waterfall
