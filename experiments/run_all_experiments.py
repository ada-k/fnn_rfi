"""Master experiment script: runs all Phase 3 experiments and generates
visualizations and report content.

Usage:
    python3 -m fnn_rfi.experiments.run_all_experiments

Runs on MPS if available, CPU fallback. Designed for MacBook Air M4 (16GB).
"""
import json
import os
import sys
import time

import torch

# Ensure repo root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fnn_rfi.experiments.run_grid import run_single, _run_name
from fnn_rfi.experiments.utils import get_device, set_seed
from fnn_rfi.experiments.visualize import (
    plot_spectrogram_samples,
    plot_segmentation_examples,
    plot_inpaint_examples,
    plot_learning_curves,
    plot_ablation_heatmap,
    plot_pareto,
    plot_metric_bars,
    plot_fnn_architecture,
    run_snr_sweep,
    make_results_table,
)

DEVICE = get_device()
FIGS_DIR = os.path.join(ROOT, "fnn_rfi", "figs")
RUNS_DIR = os.path.join(ROOT, "runs")
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


def _fig(name):
    return os.path.join(FIGS_DIR, name)


def _runs(name):
    return os.path.join(RUNS_DIR, name)


# ==================================================================
# EXPERIMENT DEFINITIONS
# ==================================================================

# --- Segmentation Ablation Grid (16 configs) ---
SEG_EXPERIMENTS = []

# Core ablation: use_fib x use_separable (4 combos) with default settings
for fib in [True, False]:
    for sep in [True, False]:
        SEG_EXPERIMENTS.append({
            "model": "fnn_seg", "use_fib": fib, "use_separable": sep,
            "drop_path_rate": 0.0, "stages": 3, "base_ch": 16, "max_depth": 3,
        })

# Drop-path sweep (fib=True, sep=False baseline)
for dp in [0.05, 0.1, 0.2]:
    SEG_EXPERIMENTS.append({
        "model": "fnn_seg", "use_fib": True, "use_separable": False,
        "drop_path_rate": dp, "stages": 3, "base_ch": 16, "max_depth": 3,
    })

# Stages sweep {2, 4} (stage=3 already in core ablation)
for st in [2, 4]:
    SEG_EXPERIMENTS.append({
        "model": "fnn_seg", "use_fib": True, "use_separable": False,
        "drop_path_rate": 0.0, "stages": st, "base_ch": 16, "max_depth": 3,
    })

# Channel base sweep {8, 32} (16 already in core)
for ch in [8, 32]:
    SEG_EXPERIMENTS.append({
        "model": "fnn_seg", "use_fib": True, "use_separable": False,
        "drop_path_rate": 0.0, "stages": 3, "base_ch": ch, "max_depth": 3,
    })

# Best combo: fib + sep + drop-path
SEG_EXPERIMENTS.append({
    "model": "fnn_seg", "use_fib": True, "use_separable": True,
    "drop_path_rate": 0.1, "stages": 3, "base_ch": 16, "max_depth": 3,
})

# Baselines
SEG_EXPERIMENTS.append({
    "model": "unet", "base_ch": 16, "depth": 3,
})
SEG_EXPERIMENTS.append({
    "model": "resunet", "base_ch": 16, "stages": 3,
})

# --- Classification experiments ---
CLS_EXPERIMENTS = []

for fib in [True, False]:
    CLS_EXPERIMENTS.append({
        "model": "fnn", "use_fib": fib, "use_separable": False,
        "drop_path_rate": 0.0, "stages": 3, "base_channels": 16,
    })

CLS_EXPERIMENTS.append({
    "model": "fnn", "use_fib": True, "use_separable": True,
    "drop_path_rate": 0.0, "stages": 3, "base_channels": 16,
})

for m in ["cnn", "rnn", "transformer", "resnet_cls"]:
    CLS_EXPERIMENTS.append({"model": m})

# --- Inpainting experiments ---
INPAINT_EXPERIMENTS = [
    {"model": "fnn_inpaint", "use_fib": True, "use_separable": False,
     "drop_path_rate": 0.0, "stages": 3, "base_ch": 16},
    {"model": "fnn_inpaint", "use_fib": True, "use_separable": True,
     "drop_path_rate": 0.0, "stages": 3, "base_ch": 16},
    {"model": "fnn_inpaint", "use_fib": True, "use_separable": False,
     "drop_path_rate": 0.1, "stages": 3, "base_ch": 16},
    {"model": "fnn_inpaint", "use_fib": True, "use_separable": False,
     "drop_path_rate": 0.0, "stages": 3, "base_ch": 32},
]

SEEDS = [42, 123, 7]


# ==================================================================
# RUN FUNCTIONS
# ==================================================================

def run_experiment_set(experiments, seeds, epochs, batch_size, lr,
                       dataset, out_root, subset=None, val_size=512):
    """Run a list of experiment configs across seeds. Returns results list."""
    results = []
    total = len(experiments) * len(seeds)
    idx = 0
    for cfg in experiments:
        for seed in seeds:
            idx += 1
            rname = _run_name(cfg, seed)
            outdir = os.path.join(out_root, rname)

            # Skip completed
            rfile = os.path.join(outdir, "results.json")
            if os.path.exists(rfile):
                print(f"  [{idx}/{total}] SKIP {rname}")
                with open(rfile) as f:
                    prev = json.load(f)
                results.append({
                    "run_name": rname,
                    "model": cfg["model"],
                    "config": prev.get("config", cfg),
                    "params": prev.get("config", {}).get("params", 0),
                    "final_metrics": prev["history"][-1] if prev.get("history") else {},
                    "outdir": outdir,
                })
                continue

            print(f"  [{idx}/{total}] {rname} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                summary = run_single(
                    cfg, seed, epochs, batch_size, lr, dataset,
                    out_root, DEVICE, subset, val_size,
                )
                dt = time.time() - t0
                iou = summary["final_metrics"].get("val_iou", "")
                acc = summary["final_metrics"].get("val_acc", "")
                info = f"IoU={iou:.4f}" if isinstance(iou, float) else (
                    f"Acc={acc:.4f}" if isinstance(acc, float) else "")
                print(f"{dt:.1f}s  {info}")
                results.append(summary)
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({"run_name": rname, "error": str(e), "model": cfg["model"]})

    return results


def generate_visualizations(seg_results, cls_results, inp_results):
    """Generate all visualization figures."""
    print("\n--- Generating visualizations ---")

    # 1. Spectrogram samples
    from fnn_rfi.data.synthetic_spectrogram import SyntheticSpectrogramDataset
    ds = SyntheticSpectrogramDataset(length=8, h=64, w=64, seed=0)
    plot_spectrogram_samples(ds, n_samples=4, out_path=_fig("spectrogram_samples.png"))
    print("  spectrogram_samples.png")

    # 2. Architecture diagram
    plot_fnn_architecture(stages=3, use_fib=True, base_ch=16,
                          out_path=_fig("fnn_architecture.png"))
    plot_fnn_architecture(stages=4, use_fib=True, base_ch=16,
                          out_path=_fig("fnn_architecture_4stage.png"))
    print("  fnn_architecture.png")

    # 3. Segmentation results
    if seg_results:
        # Ablation heatmaps
        plot_ablation_heatmap(seg_results, "use_fib", "use_separable",
                              metric="val_iou",
                              out_path=_fig("ablation_fib_vs_sep.png"))
        print("  ablation_fib_vs_sep.png")

        plot_ablation_heatmap(seg_results, "drop_path_rate", "stages",
                              metric="val_iou",
                              out_path=_fig("ablation_dp_vs_stages.png"))
        print("  ablation_dp_vs_stages.png")

        plot_ablation_heatmap(seg_results, "base_ch", "stages",
                              metric="val_iou",
                              out_path=_fig("ablation_ch_vs_stages.png"))
        print("  ablation_ch_vs_stages.png")

        # Pareto plot
        plot_pareto(seg_results, x_key="params", y_metric="val_iou",
                    out_path=_fig("pareto_seg.png"))
        print("  pareto_seg.png")

        # Metric bars
        plot_metric_bars(seg_results, metric="val_iou", group_by="model",
                         out_path=_fig("seg_iou_by_model.png"))
        print("  seg_iou_by_model.png")

        plot_metric_bars(seg_results, metric="val_f1", group_by="model",
                         out_path=_fig("seg_f1_by_model.png"))
        print("  seg_f1_by_model.png")

        # Learning curves: best FNNSeg vs UNet vs ResUNet
        curve_runs = []
        curve_labels = []
        for r in seg_results:
            if r.get("config", {}).get("seed") == 42 and "error" not in r:
                rpath = os.path.join(r.get("outdir", ""), "results.json")
                if os.path.exists(rpath):
                    curve_runs.append(rpath)
                    curve_labels.append(r["run_name"])
        if curve_runs:
            plot_learning_curves(curve_runs[:8], curve_labels[:8],
                                 metric="val_iou",
                                 out_path=_fig("learning_curves_seg.png"))
            print("  learning_curves_seg.png")

        # Results table
        make_results_table(seg_results, out_path=_fig("seg_results_table.txt"))
        print("  seg_results_table.txt")

        # Seg prediction examples from best model
        best = max(
            [r for r in seg_results if "error" not in r],
            key=lambda r: r.get("final_metrics", {}).get("val_iou", 0),
            default=None,
        )
        if best and os.path.exists(os.path.join(best["outdir"], "best.pth")):
            from fnn_rfi.experiments.utils import get_model
            model_cfg = dict(best["config"])
            model_name = model_cfg.pop("model", "fnn_seg")
            for k in ["seed", "epochs", "lr", "batch_size", "dataset", "params",
                       "subset", "val_size"]:
                model_cfg.pop(k, None)
            model = get_model(model_name, **model_cfg)
            ckpt = torch.load(os.path.join(best["outdir"], "best.pth"),
                              map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(DEVICE)
            ds_val = SyntheticSpectrogramDataset(length=8, h=64, w=64, seed=999)
            plot_segmentation_examples(model, ds_val, n_samples=4,
                                       device=str(DEVICE),
                                       out_path=_fig("seg_predictions.png"))
            print("  seg_predictions.png")

            # SNR sweep
            snr_path, snr_data = run_snr_sweep(
                model, [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0],
                device=str(DEVICE), n_samples=256,
                out_path=_fig("snr_sweep.png"),
            )
            print("  snr_sweep.png")
            with open(_fig("snr_sweep_data.json"), "w") as f:
                json.dump({str(k): v for k, v in snr_data.items()}, f, indent=2)

    # 4. Classification results
    if cls_results:
        plot_metric_bars(cls_results, metric="val_acc", group_by="model",
                         out_path=_fig("cls_acc_by_model.png"))
        print("  cls_acc_by_model.png")

    # 5. Inpainting results
    if inp_results:
        plot_metric_bars(inp_results, metric="val_iou", group_by="run_name",
                         out_path=_fig("inpaint_iou_bars.png"))
        print("  inpaint_iou_bars.png")

        plot_metric_bars(inp_results, metric="val_psnr", group_by="run_name",
                         out_path=_fig("inpaint_psnr_bars.png"))
        print("  inpaint_psnr_bars.png")

        # Inpaint examples from best model
        best_inp = max(
            [r for r in inp_results if "error" not in r],
            key=lambda r: r.get("final_metrics", {}).get("val_iou", 0),
            default=None,
        )
        if best_inp and os.path.exists(os.path.join(best_inp["outdir"], "best.pth")):
            from fnn_rfi.models.fnn_inpaint import FNNInpaint
            cfg = dict(best_inp["config"])
            m = FNNInpaint(
                in_ch=1,
                base_ch=cfg.get("base_ch", 16),
                stages=cfg.get("stages", 3),
                use_fib=cfg.get("use_fib", True),
                use_separable=cfg.get("use_separable", False),
                drop_path_rate=cfg.get("drop_path_rate", 0.0),
            )
            ckpt = torch.load(os.path.join(best_inp["outdir"], "best.pth"),
                              map_location="cpu", weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            m.to(DEVICE)
            ds_clean = SyntheticSpectrogramDataset(length=8, h=64, w=64, seed=999,
                                                   return_clean=True)
            plot_inpaint_examples(m, ds_clean, n_samples=4, device=str(DEVICE),
                                  out_path=_fig("inpaint_predictions.png"))
            print("  inpaint_predictions.png")


# ==================================================================
# MAIN
# ==================================================================

def main():
    print("=" * 70)
    print(" FNN-RFI Phase 3: Full Experiment Suite")
    print(f" Device: {DEVICE}")
    print(f" Seeds: {SEEDS}")
    print("=" * 70)

    all_results = {}

    # ---- 1. Segmentation ablation ----
    print("\n=== SEGMENTATION EXPERIMENTS ===")
    print(f"  {len(SEG_EXPERIMENTS)} configs x {len(SEEDS)} seeds "
          f"= {len(SEG_EXPERIMENTS) * len(SEEDS)} runs")
    seg_results = run_experiment_set(
        SEG_EXPERIMENTS, SEEDS, epochs=15, batch_size=16, lr=1e-3,
        dataset="synthetic", out_root=RUNS_DIR, subset=1024, val_size=256,
    )
    all_results["segmentation"] = seg_results

    # ---- 2. Classification ----
    print("\n=== CLASSIFICATION EXPERIMENTS ===")
    print(f"  {len(CLS_EXPERIMENTS)} configs x {len(SEEDS)} seeds")
    cls_results = run_experiment_set(
        CLS_EXPERIMENTS, SEEDS, epochs=10, batch_size=32, lr=1e-3,
        dataset="cifar", out_root=RUNS_DIR, subset=1024, val_size=256,
    )
    all_results["classification"] = cls_results

    # ---- 3. Inpainting ----
    print("\n=== INPAINTING EXPERIMENTS ===")
    print(f"  {len(INPAINT_EXPERIMENTS)} configs x {len(SEEDS)} seeds")
    inp_results = run_experiment_set(
        INPAINT_EXPERIMENTS, SEEDS, epochs=15, batch_size=16, lr=1e-3,
        dataset="synthetic_inpaint", out_root=RUNS_DIR, subset=1024, val_size=256,
    )
    all_results["inpainting"] = inp_results

    # ---- 4. Save combined ----
    combined_path = os.path.join(RUNS_DIR, f"all_results_{int(time.time())}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined_path}")

    # ---- 5. Visualizations ----
    generate_visualizations(seg_results, cls_results, inp_results)

    print("\n" + "=" * 70)
    print(" All experiments and visualizations complete.")
    print(f" Figures: {FIGS_DIR}")
    print(f" Runs: {RUNS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
