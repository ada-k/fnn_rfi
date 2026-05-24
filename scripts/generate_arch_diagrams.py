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
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
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

    bw, bh = 2.6, 0.80   # larger boxes

    # Classic U-shape in larger canvas (18×13)
    enc_x = 3.2
    dec_x = 14.8
    enc_ys = [11.0, 9.0, 7.0]     # encoder top → bottom
    dec_ys = [7.0,  9.0, 11.0]   # decoder bottom → top
    btn_y  = 4.5
    enc_ch = [16, 32, 48]
    dec_ch = [48, 32, 16]
    enc_depths = [2, 3, 3]

    # Input and output
    _box(ax, 0.85, 11.0, 1.3, bh, "Input",
         "512×512 · 2-ch", facecolor=C_IO, fontsize=9)
    _box(ax, 17.2, 11.0, 1.3, bh, "Output",
         "512×512 · 1-ch", facecolor=C_IO, fontsize=9)

    # Encoder stages
    for i, (y, ch, d) in enumerate(zip(enc_ys, enc_ch, enc_depths)):
        _box(ax, enc_x, y, bw, bh,
             "FractalBlock", f"{ch} ch  |  depth={d}",
             facecolor=C_ENC, fontsize=9)
        if i == 0:
            _arrow(ax, 0.85 + 0.65, 11.0, enc_x - bw/2, 11.0, color=C_CONN)
        else:
            _arrow(ax, enc_x, enc_ys[i-1] - bh/2 - 0.55,
                   enc_x, y + bh/2, color=C_CONN)
        pool_y = y - bh/2 - 0.38
        _box(ax, enc_x, pool_y, 1.8, 0.50,
             "MaxPool 2×2", facecolor=C_POOL, fontsize=8)
        _arrow(ax, enc_x, y - bh/2, enc_x, pool_y + 0.25, color=C_CONN)

    # diagonal enc → bottleneck
    _arrow(ax, enc_x, enc_ys[-1] - bh/2 - 0.55,
           9.0 - bw*0.5, btn_y + bh/2, color=C_CONN)

    # Bottleneck
    _box(ax, 9.0, btn_y, bw*1.4, bh*1.15,
         "Bottleneck", "FractalBlock  96 ch  |  depth=3",
         facecolor=C_BTN, fontsize=10)

    # diagonal bottleneck → dec
    _arrow(ax, 9.0 + bw*0.7, btn_y,
           dec_x - bw/2 - 0.85, dec_ys[0] - bh*0.35, color=C_CONN)

    # Decoder stages
    for i, (y, ch, d) in enumerate(zip(dec_ys, dec_ch, enc_depths[::-1])):
        _box(ax, dec_x, y, bw, bh,
             "FractalBlock", f"{ch} ch  |  depth={d}",
             facecolor=C_DEC, fontsize=9)
        _box(ax, dec_x - bw/2 - 0.75, y, 1.2, 0.55,
             "ConvT2d\n×2 up", facecolor=C_POOL, fontsize=7.5)
        _arrow(ax, dec_x - bw/2 - 0.15, y, dec_x - bw/2, y, color=C_CONN)
        if i > 0:
            _arrow(ax, dec_x, dec_ys[i-1] + bh/2,
                   dec_x, y - bh/2, color=C_CONN)

    # Output arrow
    _arrow(ax, dec_x + bw/2, 11.0, 17.2 - 0.65, 11.0, color=C_CONN)

    # Skip connections
    skip_pairs = [(enc_ys[0], dec_ys[2]),
                  (enc_ys[1], dec_ys[1]),
                  (enc_ys[2], dec_ys[0])]
    for ey, dy in skip_pairs:
        ax.annotate("", xy=(dec_x - bw/2 - 0.75 - 0.6, dy),
                    xytext=(enc_x + bw/2, ey),
                    arrowprops=dict(arrowstyle="->", color=C_SKIP,
                                    lw=2.0, linestyle="dashed"),
                    zorder=2)
    ax.text(9.0, 8.1, "skip connections (concat)",
            ha="center", va="center", fontsize=9, color=C_SKIP,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF7ED",
                      edgecolor=C_SKIP, alpha=0.9))

    # Channel/depth table at the bottom
    ax.text(9.0, 3.0, "Fibonacci channel schedule  (base_ch=16, stages=3)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#1E3A5F")
    cols = ["Stage", "Fib multiplier", "Channels", "FractalBlock depth"]
    rows_data = [
        ("Encoder 0", "F\u2082 = 1", "16", "2"),
        ("Encoder 1", "F\u2083 = 2", "32", "3"),
        ("Encoder 2", "F\u2084 = 3", "48", "3"),
        ("Bottleneck", "2 \u00d7 48",  "96", "3"),
        ("Decoder 2",  "\u2014",        "48", "3"),
        ("Decoder 1",  "\u2014",        "32", "3"),
        ("Decoder 0",  "\u2014",        "16", "2"),
    ]
    col_xs = [6.0, 7.8, 9.3, 11.2]
    row_h = 0.34
    base_y = 2.7
    for j, cx in enumerate(col_xs):
        ax.text(cx, base_y, cols[j], ha="center", va="center", fontsize=8,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#1E3A5F", alpha=0.9))
    for i, row in enumerate(rows_data):
        ry = base_y - (i+1)*row_h
        bg = "#EDE9FE" if "Bottleneck" in row[0] else "#DBEAFE"
        for j, (cx, val) in enumerate(zip(col_xs, row)):
            ax.text(cx, ry, val, ha="center", va="center", fontsize=8,
                    color="#1E3A5F",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=bg, alpha=0.7))

    # Legend
    patches = [
        mpatches.Patch(facecolor=C_ENC,  label="Encoder FractalBlock"),
        mpatches.Patch(facecolor=C_BTN,  label="Bottleneck FractalBlock"),
        mpatches.Patch(facecolor=C_DEC,  label="Decoder FractalBlock"),
        mpatches.Patch(facecolor=C_POOL, label="Pool / Upsample"),
        mpatches.Patch(facecolor=C_SKIP, label="Skip connection (concat)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              bbox_to_anchor=(0.0, 0.0), framealpha=0.95)

    ax.set_title("Figure 1 — FNNSeg Architecture: Fibonacci U-Net with Recursive FractalBlocks\n"
                 "Parameters: 218,609 (2-ch HERA)  |  218,465 (1-ch LOFAR)",
                 fontsize=12, fontweight="bold", pad=12)

    path = f"{OUTDIR}/fig19_fnnseg_arch.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════
# FIG 20 — FractalBlock Recursion Tree (depth=3)
# ══════════════════════════════════════════════════════════════
def fig_fractal_block():
    # Canvas is wider (22 units) so the left-margin labels never touch the boxes.
    fig, ax = plt.subplots(figsize=(22, 11))
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.8, 10.5)
    ax.axis("off")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    bw, bh = 2.6, 0.78
    C_LEAF  = "#16A34A"
    C_MID   = "#2563EB"
    C_ROOT  = "#7C3AED"
    C_MERGE = "#DC2626"

    # All nodes shifted +2 right so left margin (0–2.9) is free for annotations
    root_xy    = (11.0, 9.5)
    l2_left    = (6.5,  7.5)
    l2_right   = (15.5, 7.5)
    l1_ll      = (4.2,  5.5)
    l1_lr      = (8.8,  5.5)
    l1_rl      = (13.2, 5.5)
    l1_rr      = (17.8, 5.5)
    merge_l    = (6.5,  3.2)
    merge_r    = (15.5, 3.2)
    merge_root = (11.0, 1.0)

    def _node(xy, label, sub, color, lw=bw, lh=bh):
        _box(ax, xy[0], xy[1], lw, lh, label, sub, facecolor=color, fontsize=9)

    # Root
    _node(root_xy, "FractalBlock", "in_ch \u2192 out_ch  |  depth=3", C_ROOT, bw*1.4, bh*1.1)

    # Depth-2 nodes
    _node(l2_left,  "FractalBlock", "out_ch \u2192 out_ch  |  depth=2", C_MID)
    _node(l2_right, "FractalBlock", "out_ch \u2192 out_ch  |  depth=2", C_MID)

    # Leaf nodes
    for xy in [l1_ll, l1_lr, l1_rl, l1_rr]:
        _node(xy, "ConvBlock", "3\u00d73 \u2192 BN \u2192 SiLU  \u00d72", C_LEAF)

    # Merge nodes
    for xy in [merge_l, merge_r, merge_root]:
        _node(xy, "Merge", "cat + 1\u00d71 conv", C_MERGE, bw*1.1, bh*0.95)

    # Arrows: root → depth-2
    _arrow(ax, root_xy[0], root_xy[1] - bh*0.55, l2_left[0],  l2_left[1]  + bh/2)
    _arrow(ax, root_xy[0], root_xy[1] - bh*0.55, l2_right[0], l2_right[1] + bh/2)

    # depth-2 → leaves
    _arrow(ax, l2_left[0],  l2_left[1]  - bh/2, l1_ll[0], l1_ll[1] + bh/2)
    _arrow(ax, l2_left[0],  l2_left[1]  - bh/2, l1_lr[0], l1_lr[1] + bh/2)
    _arrow(ax, l2_right[0], l2_right[1] - bh/2, l1_rl[0], l1_rl[1] + bh/2)
    _arrow(ax, l2_right[0], l2_right[1] - bh/2, l1_rr[0], l1_rr[1] + bh/2)

    # leaves → merge
    _arrow(ax, l1_ll[0], l1_ll[1] - bh/2, merge_l[0], merge_l[1] + bh/2)
    _arrow(ax, l1_lr[0], l1_lr[1] - bh/2, merge_l[0], merge_l[1] + bh/2)
    _arrow(ax, l1_rl[0], l1_rl[1] - bh/2, merge_r[0], merge_r[1] + bh/2)
    _arrow(ax, l1_rr[0], l1_rr[1] - bh/2, merge_r[0], merge_r[1] + bh/2)

    # merges → root merge
    _arrow(ax, merge_l[0], merge_l[1] - bh/2, merge_root[0], merge_root[1] + bh/2)
    _arrow(ax, merge_r[0], merge_r[1] - bh/2, merge_root[0], merge_root[1] + bh/2)

    # DropPath labels — to the right of each node, clear of box border
    for xy in [l2_left, l2_right, l1_ll, l1_lr, l1_rl, l1_rr]:
        ax.text(xy[0] + bw*0.57, xy[1] + 0.06,
                "DropPath\n(rate=0.0)", fontsize=6.0, color="#6B7280",
                va="center", ha="left", linespacing=1.3)

    # Short left-margin level labels — kept well within the 0–2.9 unit margin
    for y_lv, lbl in [
        (9.5,  "Root (depth=3)"),
        (7.5,  "Depth-2 branches"),
        (5.5,  "Depth-1 leaves\n(ConvBlock)"),
        (3.2,  "Level-2 merge"),
        (1.0,  "Level-3 merge\n(final output)"),
    ]:
        ax.text(0.15, y_lv, lbl, va="center", fontsize=7.5, color="#374151",
                fontstyle="italic", ha="left", linespacing=1.3)

    ax.set_title("FractalBlock Recursion Tree \u2014 depth=3\n"
                 "Each non-leaf node spawns two parallel paths that are concatenated and projected at each level",
                 fontsize=12, fontweight="bold", pad=10)

    ax.text(11.0, -0.55,
            "ConvBlock = Conv2d(3\u00d73) \u2192 BN \u2192 SiLU \u2192 Conv2d(3\u00d73) \u2192 BN \u2192 SiLU"
            "  |  Merge = cat([left, right], dim=1) \u2192 Conv2d(1\u00d71) \u2192 out_ch",
            ha="center", va="top", fontsize=8.5, color="#1E3A5F",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0F9FF", edgecolor="#93C5FD"))

    patches = [
        mpatches.Patch(facecolor=C_ROOT,  label="FractalBlock (root, depth=3)"),
        mpatches.Patch(facecolor=C_MID,   label="FractalBlock (depth=2)"),
        mpatches.Patch(facecolor=C_LEAF,  label="ConvBlock leaf (depth=1)"),
        mpatches.Patch(facecolor=C_MERGE, label="Merge (cat + 1\u00d71 proj)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.9)

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
