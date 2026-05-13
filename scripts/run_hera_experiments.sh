#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# HERA real-data experiments — run ONE command at a time.
#
# STATUS (as of 2026-05-10):
#   DONE (CPU, ~3 days each):
#     fnn_seg  seed=0  test_acc=0.999921  test_f1=0.996386  test_auc=0.999804
#     unet     seed=0  test_acc=0.999897  test_f1=0.995265  test_auc=0.999735
#   IN PROGRESS (CPU, leave running — will give CPU baseline for report):
#     resunet  seed=0  epoch 10/30  ~38 min/epoch on Mac CPU
#   TODO (run on GCP GPU — see GCP setup guide below):
#     fnn_seg  seed=1, seed=2
#     unet     seed=1, seed=2
#     resunet  seed=1, seed=2
#
# NOTE ON du Toit benchmark: data/zenodo/results_all_models.csv already
# contains published test results for RNET, UNET, DSC_RESUNET, RFI_NET etc.
# on the same HERA dataset. Use those directly in the comparison table —
# no need to re-run du Toit's architectures from scratch.
#
# ─────────────────────────────────────────────────────────────────────────────
# GCP GPU SETUP (one-time, ~30 min)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. CREATE ACCOUNT
#    Go to: https://console.cloud.google.com
#    Sign in with your Google account.
#    Click "Try for free" → add a credit card (you won't be charged until
#    the $300 free credit runs out, valid 90 days).
#
# 2. CREATE A PROJECT
#    Top bar → "Select a project" → "New Project"
#    Name: fnn-rfi-experiments
#
# 3. ENABLE COMPUTE ENGINE API
#    console.cloud.google.com/apis/library/compute.googleapis.com
#    Click "Enable".
#
# 4. REQUEST GPU QUOTA (T4 — cheapest GPU, ~$0.35/hr, or ~$0.11/hr spot)
#    console.cloud.google.com/iam-admin/quotas
#    Filter: "NVIDIA_T4_GPUS"  Region: us-central1
#    Request quota = 1 (approval usually instant for new accounts).
#
# 5. CREATE THE VM
#    console.cloud.google.com/compute/instances → "Create Instance"
#
#    Name:          fnn-rfi-gpu
#    Region:        us-central1 (cheapest)
#    Zone:          us-central1-a
#    Machine type:  n1-standard-8  (8 vCPU, 30 GB RAM)
#    GPU:           Click "Add GPU" → NVIDIA T4 × 1
#    Boot disk:     "Change" → Deep Learning on Linux (Debian 11, CUDA 12.3)
#                   Size: 200 GB SSD (needs ~20 GB for pickles + code)
#    Firewall:      tick "Allow HTTP" and "Allow HTTPS"
#
#    Click "Create". Wait ~2 minutes.
#
# 6. SSH INTO THE VM
#    In the Instances list, click the "SSH" button next to fnn-rfi-gpu.
#    A browser terminal opens. When asked "Do you want to install the Nvidia
#    driver?" → YES.  Wait for driver install (~5 min), then reconnect SSH.
#
#    Verify GPU:
#      nvidia-smi
#    You should see the T4 with ~15 GB VRAM.
#
# 7. TRANSFER YOUR CODE (on the VM)
#    Option A — if code is on GitHub:
#      git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
#      cd YOUR_REPO
#
#    Option B — if not on GitHub, zip on your Mac and gcloud scp:
#      # On Mac:
#      cd /Users/ada/uel && zip -r fnn_rfi_code.zip fnn_rfi/ --exclude "*/__pycache__/*"
#      gcloud compute scp fnn_rfi_code.zip fnn-rfi-gpu:~ --zone us-central1-a
#      # On VM:
#      unzip fnn_rfi_code.zip
#
# 8. DOWNLOAD DATASETS DIRECTLY FROM ZENODO (on the VM — faster than uploading from Mac)
#    mkdir -p data/zenodo && cd data/zenodo
#
#    # HERA pixel-perfect labels (3.3 GB):
#    wget "https://zenodo.org/records/8275061/files/HERA_28-03-2023_all.pkl"
#
#    # HERA AOFlagger labels (3.3 GB):
#    wget "https://zenodo.org/records/8275061/files/HERA_AOF_20-07-2023_all.pkl"
#
#    # Results CSV (103 MB):
#    wget "https://zenodo.org/records/8275061/files/results_all_models.csv"
#
#    # LOFAR (9.2 GB — only if needed, takes time):
#    wget "https://zenodo.org/records/8275061/files/LOFAR_Full_RFI_dataset.pkl"
#
#    cd ~/YOUR_REPO   # back to project root
#
# 9. INSTALL DEPENDENCIES (on VM — CUDA-enabled torch already in Deep Learning image)
#    pip install scikit-learn pandas
#    # PyTorch + torchvision are pre-installed in the Deep Learning VM image.
#    # Verify:
#    python3 -c "import torch; print(torch.cuda.is_available())"
#    # Should print: True
#
# 10. RUN THE EXPERIMENTS (on VM — GPU auto-detected by get_device())
#     mkdir -p logs/hera
#     Then copy-paste the GPU commands from the section below.
#
# ─────────────────────────────────────────────────────────────────────────────
# COST ESTIMATE
#   n1-standard-8 + T4:  ~$0.35/hr  standard  |  ~$0.11/hr spot (preemptible)
#   Each 30-epoch run on T4: ~1–2 hours (vs 3 days on CPU → ~40× speedup)
#   6 remaining runs × 2 hrs = ~12 hrs = ~$4.20 standard / ~$1.30 spot
#   You have $300 free credit → effectively free.
#
#   To use spot pricing, add --provisioning-model=SPOT when creating the VM
#   (spot VMs can be preempted, so add --restart-on-failure=false).
#   For short jobs (<2 hrs each), spot is fine.
#
# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS (same as CPU — GPU is auto-selected via get_device())
#   --patch 256  --epochs 30  --batch 32   ← increase batch to 32 on GPU
#   --base_ch 16  --depth 3
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════
# SECTION A — COMPLETED ON CPU (Mac)  [DO NOT RE-RUN]
# ════════════════════════════════════════════════════════════════════

# [DONE] 1/9  FNNSeg  seed=0  CPU  test_acc=0.999921  test_f1=0.996386
# nohup python3 fnn_rfi/scripts/train_hera.py \
#   --model fnn_seg --base_ch 16 --depth 3 \
#   --patch 256 --epochs 30 --batch 8 --seed 0 \
#   > logs/hera/fnn_seg_ch16_seed0.log 2>&1

# [DONE] 4/9  UNet  seed=0  CPU  test_acc=0.999897  test_f1=0.995265
# nohup python3 fnn_rfi/scripts/train_hera.py \
#   --model unet --base_ch 16 --depth 3 \
#   --patch 256 --epochs 30 --batch 8 --seed 0 \
#   > logs/hera/unet_ch16_seed0.log 2>&1

# ════════════════════════════════════════════════════════════════════
# SECTION B — IN PROGRESS ON CPU (Mac)  [LEAVE RUNNING]
# ════════════════════════════════════════════════════════════════════

# [RUNNING] 7/9  ResUNet  seed=0  CPU  epoch 10/30  ~15 hrs remaining
# nohup python3 fnn_rfi/scripts/train_hera.py \
#   --model resunet --base_ch 16 --depth 3 \
#   --patch 256 --epochs 30 --batch 8 --seed 0 \
#   > logs/hera/resunet_ch16_seed0.log 2>&1

# ════════════════════════════════════════════════════════════════════
# SECTION C — RUN ON GCP GPU  (copy-paste one at a time on the VM)
# ════════════════════════════════════════════════════════════════════
# batch=32 on GPU (T4 has 15GB VRAM, 32×(2×256×256×4B) = ~128MB — fits easily)
# get_device() in train_hera.py auto-selects CUDA if available — no code change needed.

# ── 2/9  FNNSeg  seed=1  [GPU] ───────────────────────────────────────────────
nohup python3 fnn_rfi/scripts/train_hera.py \
  --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/fnn_seg_ch16_seed1.log 2>&1

# ── 3/9  FNNSeg  seed=2  [GPU] ───────────────────────────────────────────────
nohup python3 fnn_rfi/scripts/train_hera.py \
  --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/fnn_seg_ch16_seed2.log 2>&1

# ── 5/9  UNet  seed=1  [GPU] ─────────────────────────────────────────────────
nohup python3 fnn_rfi/scripts/train_hera.py \
  --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/unet_ch16_seed1.log 2>&1

# ── 6/9  UNet  seed=2  [GPU] ─────────────────────────────────────────────────
nohup python3 fnn_rfi/scripts/train_hera.py \
  --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/unet_ch16_seed2.log 2>&1

# ── 8/9  ResUNet  seed=1  [GPU] ──────────────────────────────────────────────
nohup python3 fnn_rfi/scripts/train_hera.py \
  --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/resunet_ch16_seed1.log 2>&1

# ── 9/9  ResUNet  seed=2  [GPU] ──────────────────────────────────────────────
nohup python3 fnn_rfi/scripts/train_hera.py \
  --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/resunet_ch16_seed2.log 2>&1

# ════════════════════════════════════════════════════════════════════
# SECTION D — HERA_AOF experiments  (same split, AOFlagger labels)
# Run after Section C completes. Compares DNN labels vs AOFlagger labels.
# ════════════════════════════════════════════════════════════════════
PKL_AOF="data/zenodo/HERA_AOF_20-07-2023_all.pkl"

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 0 \
  > logs/hera/aof_fnn_seg_ch16_seed0.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/aof_fnn_seg_ch16_seed1.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/aof_fnn_seg_ch16_seed2.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 0 \
  > logs/hera/aof_unet_ch16_seed0.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/aof_unet_ch16_seed1.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/aof_unet_ch16_seed2.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 0 \
  > logs/hera/aof_resunet_ch16_seed0.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/aof_resunet_ch16_seed1.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_AOF --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/aof_resunet_ch16_seed2.log 2>&1

# ════════════════════════════════════════════════════════════════════
# SECTION E — LOFAR experiments  (different telescope, generalization test)
#
# NOTE: LOFAR pickle is 9.2 GB. Download on VM with:
#   wget "https://zenodo.org/records/8275061/files/LOFAR_Full_RFI_dataset.pkl" \
#        -O data/zenodo/LOFAR_Full_RFI_dataset.pkl
#
# IMPORTANT: Check LOFAR frame dimensions BEFORE running:
#   python3 -c "
#   import pickle
#   d = pickle.load(open('data/zenodo/LOFAR_Full_RFI_dataset.pkl','rb'))
#   print('X_train:', d[0].shape, 'X_test:', d[2].shape)
#   "
# If frames are 512×512 → use --patch 256 (same as HERA)
# If frames are smaller (e.g. 256×256) → use --patch 128 or omit --patch for full frame
# ════════════════════════════════════════════════════════════════════
PKL_LOF="data/zenodo/LOFAR_Full_RFI_dataset.pkl"

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 0 \
  > logs/hera/lofar_fnn_seg_ch16_seed0.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/lofar_fnn_seg_ch16_seed1.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model fnn_seg --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/lofar_fnn_seg_ch16_seed2.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 0 \
  > logs/hera/lofar_unet_ch16_seed0.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/lofar_unet_ch16_seed1.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model unet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/lofar_unet_ch16_seed2.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 0 \
  > logs/hera/lofar_resunet_ch16_seed0.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 1 \
  > logs/hera/lofar_resunet_ch16_seed1.log 2>&1

nohup python3 fnn_rfi/scripts/train_hera.py \
  --pkl $PKL_LOF --model resunet --base_ch 16 --depth 3 \
  --patch 256 --epochs 30 --batch 32 --seed 2 \
  > logs/hera/lofar_resunet_ch16_seed2.log 2>&1

# ════════════════════════════════════════════════════════════════════
# SECTION F — Cross-dataset generalization  (train on HERA, test on LOFAR)
# Run AFTER Section C (HERA models trained). Uses best checkpoint from HERA run.
# Requires a separate evaluation script — see scripts/eval_cross_dataset.py
# ════════════════════════════════════════════════════════════════════
