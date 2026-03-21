#!/usr/bin/env bash
# run_all.sh — trains every model on the synthetic dataset and runs the
# benchmark.  Execute from the fnn_rfi/ directory:
#
#   bash scripts/run_all.sh
#
# Results land in runs/<model>_synthetic_<timestamp>/ and benchmark_results.json.

set -euo pipefail
# Move to the parent of fnn_rfi/ so package imports resolve correctly
cd "$(dirname "$0")/../.."

echo "========================================================"
echo " FNN-RFI  —  full experiment run"
echo "========================================================"

# ---- segmentation models (primary task) ------------------------------------
echo ""
echo "--- Segmentation: FNNSeg (Fibonacci) ---"
python3 -m fnn_rfi.experiments.train \
    --model fnn_seg --dataset synthetic \
    --epochs 10 --batch-size 16 --lr 1e-3 --seed 42

echo ""
echo "--- Segmentation: FNNSeg (separable convs) ---"
python3 -m fnn_rfi.experiments.train \
    --model fnn_seg --dataset synthetic \
    --epochs 10 --batch-size 16 --lr 1e-3 --seed 42 --separable

echo ""
echo "--- Segmentation: FNNSeg (drop-path 0.1) ---"
python3 -m fnn_rfi.experiments.train \
    --model fnn_seg --dataset synthetic \
    --epochs 10 --batch-size 16 --lr 1e-3 --seed 42 --drop-path 0.1

echo ""
echo "--- Segmentation: UNet baseline ---"
python3 -m fnn_rfi.experiments.train \
    --model unet --dataset synthetic \
    --epochs 10 --batch-size 16 --lr 1e-3 --seed 42

echo ""
echo "--- Segmentation: ResUNet baseline ---"
python3 -m fnn_rfi.experiments.train \
    --model resunet --dataset synthetic \
    --epochs 10 --batch-size 16 --lr 1e-3 --seed 42

# ---- inpainting model -------------------------------------------------------
echo ""
echo "--- Inpainting: FNNInpaint ---"
python3 -m fnn_rfi.experiments.train_inpaint \
    --epochs 20 --batch-size 16 --lr 1e-3 --seed 42

# ---- classification smoke tests (CIFAR, small subset) ----------------------
echo ""
echo "--- Classification smoke: FNNSeg (Fibonacci) ---"
python3 -m fnn_rfi.experiments.train \
    --model fnn_seg --dataset synthetic \
    --epochs 10 --batch-size 16 --lr 1e-3 --seed 42

echo ""
echo "--- Classification smoke: FNN ---"
python3 -m fnn_rfi.experiments.train \
    --model fnn --dataset cifar \
    --epochs 5 --batch-size 32 --subset 1024 --seed 42

echo ""
echo "--- Classification smoke: SimpleCNN ---"
python3 -m fnn_rfi.experiments.train \
    --model cnn --dataset cifar \
    --epochs 5 --batch-size 32 --subset 1024 --seed 42

echo ""
echo "--- Classification smoke: LSTM ---"
python3 -m fnn_rfi.experiments.train \
    --model rnn --dataset cifar \
    --epochs 5 --batch-size 32 --subset 1024 --seed 42

echo ""
echo "--- Classification smoke: TinyViT ---"
python3 -m fnn_rfi.experiments.train \
    --model transformer --dataset cifar \
    --epochs 5 --batch-size 32 --subset 1024 --seed 42

echo ""
echo "--- Classification smoke: ResNetClassifier ---"
python3 -m fnn_rfi.experiments.train \
    --model resnet_cls --dataset cifar \
    --epochs 5 --batch-size 32 --subset 1024 --seed 42

# ---- benchmark -------------------------------------------------------------
echo ""
echo "--- Parameter & latency benchmark ---"
python3 -m fnn_rfi.scripts.benchmark

echo ""
echo "========================================================"
echo " All runs complete.  Results in runs/ and benchmark_results.json"
echo "========================================================"
