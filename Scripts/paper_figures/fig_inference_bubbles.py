"""
Inference time vs. metric bubble plots (both tracks).

Produces:
  inference_bubble_track1.svg  — Track 1: F1 vs. inference time
  inference_bubble_track2.svg  — Track 2: balanced accuracy vs. inference time
  inference_bubbles_updated.pdf — combined 1×2 figure for the paper

Bubble size encodes peak VRAM usage; color encodes use of TTA / ensembling.

Paper figure: fig:inference_time
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from adjustText import adjust_text

from utils import (
    NAVY, MM_TO_INCH, FONT_SERIF,
    load_track1_results, load_track2_results,
    style_rcparams,
)

# Reference max VRAM for consistent scaling across both tracks
_MEM_MAX_REFERENCE_GB = 23.383


def _get_ablation_group(ablations, participant):
    abl = ablations.get(participant, [])
    has_ens = any("ensembling_1" in a for a in abl)
    has_tta = "TTA" in abl
    if has_ens and has_tta:
        return 0  # ensembling + TTA
    elif has_ens:
        return 1  # ensembling only
    elif has_tta:
        return 2  # TTA only
    return 3      # neither


def _bubble_plot(ax, participants, duration, metric_values, peak_mem,
                 ablations, metric_label, mem_ref_max=None,
                 xlim=None, ylim=None):
    """
    Draw a single bubble-plot panel onto *ax*.
    Returns the group_colors array for legend construction.
    """
    group_colors = plt.cm.tab10(np.linspace(0, 0.5, 4))
    group_labels = ["ensembling + TTA", "ensembling", "TTA", "neither"]

    dur  = np.array([duration[p]       for p in participants])
    mem  = np.array([peak_mem[p] / 1e3 for p in participants])  # MB → GB
    perf = np.array([metric_values[p]  for p in participants])

    if mem_ref_max is None:
        mem_ref_max = mem.max()
    mem_norm = (mem - mem.min()) / (mem_ref_max - mem.min() + 1e-9)
    sizes = 50 + mem_norm * 500

    texts = []
    for p, x, y, s, m in zip(participants, dur, perf, sizes, mem):
        c = group_colors[_get_ablation_group(ablations, p)]
        ax.scatter(x, y, s=s, color=c, edgecolors="white", linewidths=1.2,
                   alpha=0.85, zorder=3)
        texts.append(ax.annotate(
            f"{p}\n{m:.1f} GB", xy=(x, y),
            ha="center", va="center", fontsize=7,
            fontweight="bold", color="#333", zorder=4,
        ))

    ax.set_xlabel("Mean inference duration (s)", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#ffffff")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # VRAM size legend
    vram_handles = []
    for mv in [mem.min(), (mem.min() + mem_ref_max) / 2, mem_ref_max]:
        t = (mv - mem.min()) / (mem_ref_max - mem.min() + 1e-9)
        h = ax.scatter([], [], s=50 + t * 500, c="gray", alpha=0.5,
                       label=f"{mv:.1f} GB")
        vram_handles.append(h)
    vram_legend = ax.legend(handles=vram_handles, title="Peak VRAM",
                            loc="lower right", fontsize=8,
                            title_fontsize=8, framealpha=0.7)
    ax.add_artist(vram_legend)

    # Group color legend
    group_handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=group_colors[i], markersize=8,
                      label=group_labels[i])
        for i in range(4)
    ]
    ax.legend(handles=group_handles, title="Method",
              loc="upper right", fontsize=8, title_fontsize=8, framealpha=0.7)

    return group_colors


def plot_inference_bubbles(t1_data, t2_data,
                           save_path="inference_bubbles_updated.pdf"):
    """
    Combined 1×2 inference bubble figure.
    Paper figure: fig:inference_time
    """
    style_rcparams()
    plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8})

    fig, axes = plt.subplots(
        1, 2,
        figsize=(180 * MM_TO_INCH, 85 * MM_TO_INCH),
        dpi=300,
        gridspec_kw={"wspace": 0.45},
    )
    fig.patch.set_facecolor("#ffffff")

    # Track 1
    t1_participants = sorted(t1_data["f1_score"], key=lambda p: t1_data["f1_score"][p], reverse=True)
    _bubble_plot(
        axes[0], t1_participants,
        t1_data["duration"], t1_data["f1_score"], t1_data["peak_mem"],
        t1_data["ablations"],
        metric_label="$F_1$ score",
        mem_ref_max=_MEM_MAX_REFERENCE_GB,
        xlim=None, ylim=None,
    )
    axes[0].set_title("Track 1 — Mitosis Detection", fontsize=9, fontweight="bold", pad=6)

    # Track 2
    t2_participants = sorted(t2_data["balanced_accuracy"],
                             key=lambda p: t2_data["balanced_accuracy"][p], reverse=True)
    _bubble_plot(
        axes[1], t2_participants,
        t2_data["duration"], t2_data["balanced_accuracy"], t2_data["peak_mem"],
        t2_data["ablations"],
        metric_label="Balanced accuracy",
        mem_ref_max=_MEM_MAX_REFERENCE_GB,
        xlim=[-10, 25], ylim=[0.80, 0.94],
    )
    axes[1].set_title("Track 2 — Atypical MF Classification",
                      fontsize=9, fontweight="bold", pad=6)

    for ax, label in zip(axes, ["a", "b"]):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=NAVY, va="top", clip_on=False)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


def plot_inference_bubble_single(data, participants, metric_key, metric_label,
                                 save_path, xlim=None, ylim=None):
    """Save a standalone bubble plot for one track."""
    style_rcparams()
    fig, ax = plt.subplots(figsize=(120 * MM_TO_INCH, 95 * MM_TO_INCH), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    _bubble_plot(ax, participants, data["duration"], data[metric_key],
                 data["peak_mem"], data["ablations"], metric_label,
                 mem_ref_max=_MEM_MAX_REFERENCE_GB, xlim=xlim, ylim=ylim)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate inference bubble figures.")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()
    out = args.outdir

    print("Loading Track 1 results…")
    t1 = load_track1_results()
    print("Loading Track 2 results…")
    t2 = load_track2_results()

    # Combined figure for the paper
    plot_inference_bubbles(
        t1, t2,
        save_path=os.path.join(out, "inference_bubbles_updated.pdf"),
    )
