"""
Generate two architecture diagrams for the thesis:
  fig19_fnnseg_arch.png  — FNNSeg U-Net scaffold with Fibonacci channels
  fig20_fractal_block.png — FractalBlock recursion tree (depth=3)

Run from the project root:
    python3 fnn_rfi/scripts/generate_arch_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUTDIR = "fnn_rfi/figs/thesis"
os.makedirs(OUTDIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _box(ax, x, y, w, h, label, sublabel=None, facecolor="#2563EB",
         textcolor="white", fontsize=8, alpha=0.92, radius=0.08):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         facecolor=facecolor, edgecolor="white",
                         linewidth=0.8, zorder=3, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y + (h*0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)
    if sublabel:
        ax.text(x, y - h*0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=textcolor, zorder=4, style="italic")

def _arrow(ax, x0, y0, x1, y1, color="#374151", lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def _skip(ax, x0, y0, x1, y1, label="skip", color="#F97316"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.6, linestyle="dashed",
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)
    mx, my = (x0+x1)/2, (y0+y1)/2
    ax.text(mx, my + 0.12, label, ha="center", va="bottom",
            fontsize=7, color=color, fontstyle="italic")

# ══════════════════════════════════════════════════════════════
# FIG 19 — FNNSeg Architecture
# ══════════════════════════════════════════════════════════════
def fig_fnnseg_arch():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    C_ENC  = "#2563EB"
    C_BTN  = "#7C3AED"
    C_DEC  = "#059669"
    C_IO   = "#374151"
    C_POOL = "#64748B"
    C_SKIP = "#F97316"
    C_CONN = "#1F2937"

    bw, bh = 2.0, 0.62

    # Classic U-shape:
    # Left column (encoder):  x=2.5  y = 8.5, 7.0, 5.5, 4.0
    # Bottom (bottleneck):    x=7.0  y = 2.5
    # Right column (decoder): x=11.5 y = 4.0, 5.5, 7.0, 8.5

    enc_x = 2.5
    dec_x = 11.5
    enc_ys = [8.5, 7.0, 5.5]      # 3 encoder stages
    dec_ys = [5.5, 7.0, 8.5]      # 3 decoder stages (reverse order)
    btn_y  = 3.0
    enc_ch = [16, 32, 48]
    dec_ch = [48, 32, 16]
    enc_depths = [2, 3, 3]

    # Input and output
    _box(ax, 0.7, 8.5, 1.1, bh, "Input",
         "512×512\n2-ch", facecolor=C_IO, fontsize=7.5)
    _box(ax, 13.3, 8.5, 1.1, bh, "Output",
         "512×512\n1-ch RFI", facecolor=C_IO, fontsize=7.5)

    # Encoder stages
    for i, (y, ch, d) in enumerate(zip(enc_ys, enc_ch, enc_depths)):
        _box(ax, enc_x, y, bw, bh,
             "FractalBlock", f"{ch} ch  |  depth={d}",
             facecolor=C_ENC, fontsize=8)
        if i == 0:
            _arrow(ax, 0.7 + 0.55, 8.5, enc_x - bw/2, 8.5, color=C_CONN)
        else:
            # downward arrow from pool to this stage
            _arrow(ax, enc_x, enc_ys[i-1] - bh/2 - 0.45 + 0.2,
                   enc_x, y + bh/2, color=C_CONN)
        # MaxPool arrow going down (between enc stages)
        if i < len(enc_ys) - 1:
            _box(ax, enc_x - 0.05, y - bh/2 - 0.3, 1.5, 0.4,
                 "MaxPool 2×2", facecolor=C_POOL, fontsize=7)
            _arrow(ax, enc_x, y - bh/2, enc_x, y - bh/2 - 0.14, color=C_CONN)

    # Arrow from last encoder to bottleneck
    _box(ax, enc_x - 0.05, enc_ys[-1] - bh/2 - 0.3, 1.5, 0.4,
         "MaxPool 2×2", facecolor=C_POOL, fontsize=7)
    _arrow(ax, enc_x, enc_ys[-1] - bh/2, enc_x, enc_ys[-1] - bh/2 - 0.14, color=C_CONN)
    # diagonal to bottleneck
    _arrow(ax, enc_x, enc_ys[-1] - bh/2 - 0.44,
           7.0 - bw/2*0.8, btn_y + bh/2, color=C_CONN)

    # Bottleneck
    _box(ax, 7.0, btn_y, bw*1.3, bh*1.1,
         "Bottleneck", "FractalBlock  96 ch  |  depth=3",
         facecolor=C_BTN, fontsize=9)

    # Arrow from bottleneck to first decoder
    _arrow(ax, 7.0 + bw*0.65, btn_y,
           dec_x - bw/2 - 0.7, dec_ys[0] - bh*0.3, color=C_CONN)

    # Decoder stages
    for i, (y, ch, d) in enumerate(zip(dec_ys, dec_ch, enc_depths[::-1])):
        _box(ax, dec_x, y, bw, bh,
             "FractalBlock", f"{ch} ch  |  depth={d}",
             facecolor=C_DEC, fontsize=8)
        # ConvTranspose2d box (to the left of decoder)
        _box(ax, dec_x - bw/2 - 0.6, y, 1.0, 0.45,
             "ConvT2d\n×2 up", facecolor=C_POOL, fontsize=6.5)
        _arrow(ax, dec_x - bw/2 - 0.1, y, dec_x - bw/2, y, color=C_CONN)
        if i > 0:
            _arrow(ax, dec_x, dec_ys[i-1] + bh/2,
                   dec_x, y - bh/2, color=C_CONN)

    # Output head
    _arrow(ax, dec_x + bw/2, 8.5, 13.3 - 0.55, 8.5, color=C_CONN)

    # Skip connections (horizontal, through the U)
    skip_pairs = [(enc_ys[0], dec_ys[2]),   # stage 0: enc y=8.5 → dec y=8.5
                  (enc_ys[1], dec_ys[1]),   # stage 1: enc y=7.0 → dec y=7.0
                  (enc_ys[2], dec_ys[0])]   # stage 2: enc y=5.5 → dec y=5.5
    for ey, dy in skip_pairs:
        ax.annotate("", xy=(dec_x - bw/2 - 0.6 - 0.5, dy),
                    xytext=(enc_x + bw/2, ey),
                    arrowprops=dict(arrowstyle="->", color=C_SKIP,
                                    lw=1.8, linestyle="dashed"),
                    zorder=2)
    ax.text(7.0, 6.35, "skip connections\n(concat to decoder input)",
            ha="center", va="center", fontsize=8, color=C_SKIP,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF7ED",
                      edgecolor=C_SKIP, alpha=0.8))

    # Channel/depth table at the bottom
    ax.text(7.0, 1.9, "Fibonacci channel schedule  (base_ch=16, stages=3)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#1E3A5F")
    cols = ["Stage", "Fib factor", "Channels", "FractalBlock depth"]
    rows_data = [
        ("Encoder 0", "F\u2082=1", "16", "2"),
        ("Encoder 1", "F\u2083=2", "32", "3"),
        ("Encoder 2", "F\u2084=3", "48", "3"),
        ("Bottleneck", "2\u00d748", "96", "3"),
        ("Decoder 2",  "\u2014",    "48", "3"),
        ("Decoder 1",  "\u2014",    "32", "3"),
        ("Decoder 0",  "\u2014",    "16", "2"),
    ]
    col_xs = [4.1, 5.7, 7.0, 9.0]
    row_h = 0.27
    base_y = 1.6
    for j, cx in enumerate(col_xs):
        ax.text(cx, base_y, cols[j], ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#1E3A5F", alpha=0.9))
    for i, row in enumerate(rows_data):
        ry = base_y - (i+1)*row_h
        bg = "#EDE9FE" if "Bottleneck" in row[0] else "#DBEAFE"
        for j, (cx, val) in enumerate(zip(col_xs, row)):
            ax.text(cx, ry, val, ha="center", va="center", fontsize=7.5,
                    color="#1E3A5F",
                    bbox=dict(boxstyle="round,pad=0.08", facecolor=bg, alpha=0.7))

    # Legend
    patches = [
        mpatches.Patch(facecolor=C_ENC,  label="Encoder FractalBlock"),
        mpatches.Patch(facecolor=C_BTN,  label="Bottleneck FractalBlock"),
        mpatches.Patch(facecolor=C_DEC,  label="Decoder FractalBlock"),
        mpatches.Patch(facecolor=C_POOL, label="Pool / Upsample"),
        mpatches.Patch(facecolor=C_SKIP, label="Skip connection (concat)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=8.5,
              bbox_to_anchor=(0.0, 0.0), framealpha=0.95)

    ax.set_title("FNNSeg Architecture: Fibonacci U-Net with Recursive FractalBlocks\n"
                 "Parameters: 218,609 (2-ch HERA)  |  218,465 (1-ch LOFAR)",
                 fontsize=11, fontweight="bold", pad=10)

    path = f"{OUTDIR}/fig19_fnnseg_arch.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════
# FIG 20 — FractalBlock Recursion Tree (depth=3)
# ══════════════════════════════════════════════════════════════
def fig_fractal_block():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    bw, bh = 2.0, 0.55
    C_LEAF  = "#16A34A"
    C_MID   = "#2563EB"
    C_ROOT  = "#7C3AED"
    C_MERGE = "#DC2626"
    C_SKIP  = "#F97316"

    # ── depth=3 tree layout ──────────────────────────────────
    # Root: FractalBlock(in, out, depth=3) at x=6.5, y=7
    # Level 2: Left branch (depth=2) at x=3.5, y=5.5
    #          Right branch (depth=2) at x=9.5, y=5.5
    # Level 1 (leaves): depth=2 has two depth=1 branches each
    #   Left-left:  x=2.0, y=3.8  (leaf ConvBlock)
    #   Left-right: x=5.0, y=3.8  (leaf ConvBlock)
    #   Right-left: x=8.0, y=3.8  (leaf ConvBlock)
    #   Right-right:x=11.0,y=3.8  (leaf ConvBlock)
    # Merge nodes at level 2 midpoints
    # Merge node at root

    root_xy  = (6.5, 7.0)
    l2_left  = (3.0, 5.3)
    l2_right = (10.0, 5.3)
    l1_ll    = (1.5, 3.5)
    l1_lr    = (4.5, 3.5)
    l1_rl    = (8.5, 3.5)
    l1_rr    = (11.5, 3.5)
    merge_l  = (3.0, 1.8)
    merge_r  = (10.0, 1.8)
    merge_root = (6.5, 0.5)

    def _node(xy, label, sub, color, lw=bw, lh=bh):
        _box(ax, xy[0], xy[1], lw, lh, label, sub, facecolor=color, fontsize=8)

    # Root
    _node(root_xy, "FractalBlock", "in_ch → out_ch  |  depth=3", C_ROOT, bw*1.3)

    # Depth-2 nodes
    _node(l2_left,  "FractalBlock", "out_ch → out_ch  |  depth=2", C_MID)
    _node(l2_right, "FractalBlock", "out_ch → out_ch  |  depth=2", C_MID)

    # Leaf nodes (depth=1 = ConvBlock)
    for xy, label in [(l1_ll, "ConvBlock\ndepth=1"), (l1_lr, "ConvBlock\ndepth=1"),
                       (l1_rl, "ConvBlock\ndepth=1"), (l1_rr, "ConvBlock\ndepth=1")]:
        _node(xy, "ConvBlock", "3×3 BN SiLU ×2", C_LEAF, bw*0.95)

    # Merge nodes
    for xy, label in [(merge_l, "Merge\n(cat + 1×1 conv)"),
                       (merge_r, "Merge\n(cat + 1×1 conv)"),
                       (merge_root, "Merge\n(cat + 1×1 conv)")]:
        _node(xy, label, None, C_MERGE, bw*0.95, bh*0.9)

    # Arrows: root → children
    _arrow(ax, root_xy[0], root_xy[1] - bh/2, l2_left[0], l2_left[1] + bh/2)
    _arrow(ax, root_xy[0], root_xy[1] - bh/2, l2_right[0], l2_right[1] + bh/2)

    # left depth-2 → leaves
    _arrow(ax, l2_left[0], l2_left[1] - bh/2, l1_ll[0], l1_ll[1] + bh/2)
    _arrow(ax, l2_left[0], l2_left[1] - bh/2, l1_lr[0], l1_lr[1] + bh/2)

    # right depth-2 → leaves
    _arrow(ax, l2_right[0], l2_right[1] - bh/2, l1_rl[0], l1_rl[1] + bh/2)
    _arrow(ax, l2_right[0], l2_right[1] - bh/2, l1_rr[0], l1_rr[1] + bh/2)

    # leaves → merge nodes
    _arrow(ax, l1_ll[0],  l1_ll[1] - bh/2,  merge_l[0], merge_l[1] + bh/2)
    _arrow(ax, l1_lr[0],  l1_lr[1] - bh/2,  merge_l[0], merge_l[1] + bh/2)
    _arrow(ax, l1_rl[0],  l1_rl[1] - bh/2,  merge_r[0], merge_r[1] + bh/2)
    _arrow(ax, l1_rr[0],  l1_rr[1] - bh/2,  merge_r[0], merge_r[1] + bh/2)

    # merge nodes → root merge
    _arrow(ax, merge_l[0], merge_l[1] - bh/2, merge_root[0], merge_root[1] + bh/2)
    _arrow(ax, merge_r[0], merge_r[1] - bh/2, merge_root[0], merge_root[1] + bh/2)

    # Drop-path annotation
    for xy in [l2_left, l2_right, l1_ll, l1_lr, l1_rl, l1_rr]:
        ax.text(xy[0] + bw*0.52, xy[1] + 0.05, "DropPath\n(rate=0.0)", fontsize=5.5,
                color="#6B7280", va="center")

    # Label each level
    for y_lv, lbl in [(7.0, "Level 3 (root): branches into two depth-2 sub-trees"),
                       (5.3, "Level 2: each branch further recurses into two depth-1 leaves"),
                       (3.5, "Level 1 (leaf): standard ConvBlock (3×3 BN SiLU ×2)"),
                       (1.8, "Level 2 merge: concatenate + 1×1 conv to project back to out_ch"),
                       (0.5, "Level 3 merge: final output tensor (out_ch channels)")]:
        ax.text(0.15, y_lv, lbl, va="center", fontsize=7.5, color="#374151",
                fontstyle="italic")

    # Overall title + info box
    ax.set_title("FractalBlock Recursion Tree — depth=3\n"
                 "Each non-leaf node spawns two parallel paths that are concatenated and "
                 "projected at each level",
                 fontsize=11, fontweight="bold", pad=8)

    ax.text(6.5, -0.3,
            "ConvBlock = Conv2d(3×3) → BN → SiLU → Conv2d(3×3) → BN → SiLU  |  "
            "Merge = cat([left, right], dim=1) → Conv2d(1×1) → out_ch",
            ha="center", va="top", fontsize=7.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F9FF", edgecolor="#93C5FD"))

    patches = [
        mpatches.Patch(facecolor=C_ROOT,  label="FractalBlock (root, depth=3)"),
        mpatches.Patch(facecolor=C_MID,   label="FractalBlock (depth=2)"),
        mpatches.Patch(facecolor=C_LEAF,  label="ConvBlock leaf (depth=1)"),
        mpatches.Patch(facecolor=C_MERGE, label="Merge (cat + 1×1 proj)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8.5, framealpha=0.9)

    path = f"{OUTDIR}/fig20_fractal_block.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating architecture diagrams...\n")
    fig_fnnseg_arch()
    fig_fractal_block()
    print("\nDone.")
