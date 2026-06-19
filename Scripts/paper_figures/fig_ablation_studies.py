"""
Ablation study figures — TTA and ensembling (both tracks).

Produces:
  ablation_studies.pdf — combined 2-panel figure (T1 top, T2 bottom)
                         showing the effect of removing TTA / ensembling.

Individual SVGs (also saved):
  ablation_t1_f1.svg                  — Track 1, overall F1
  ablation_t1_f1_hotspot.svg          — Track 1, hotspot F1
  ablation_t1_f1_random.svg           — Track 1, random F1
  ablation_t1_f1_challenging.svg      — Track 1, challenging F1
  ablation_t1_froc_hotspot.svg        — Track 1, hotspot FROC-AUC
  ablation_t1_froc_random.svg         — Track 1, random FROC-AUC
  ablation_t1_ap_random.svg           — Track 1, random AP
  ablation_t2_bacc.svg                — Track 2, balanced accuracy
  ablation_t2_auroc.svg               — Track 2, ROC AUC

Paper figure: fig:ablation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from utils import (
    NAVY, MM_TO_INCH, CB_COLORS, MARKERS,
    TRACK1_OUTPUT_PATH, TRACK2_OUTPUT_PATH,
    T1_PARTICIPANTS, T2_PARTICIPANTS,
    load_ablation_results, style_rcparams,
)

_X_LABELS_FULL = [
    "TTA $\\times$\nensembling $\\times$",
    "TTA ✓\nensembling $\\times$",
    "TTA $\\times$\nensembling ✓",
    "TTA ✓\nensembling ✓\n(submission)",
    "ensembling $\\times$",
    "ensembling ✓\n(submission)",
    "TTA $\\times$",
    "TTA ✓\n(submission)",
]


def plot_ablation(skey, participants, basepath, metric_label, ax=None,
                  save_path=None):
    """
    Draw the ablation plot for one metric/track onto *ax* (or a new figure).

    Shows individual participant trajectories plus group means for:
      - Complete ablation (positions 1–4): both TTA and ensembling switched off
      - Ensembling ablation (positions 5–6): only ensembling varied
      - TTA ablation (positions 7–8): only TTA varied

    Returns a dict with mean/median gains and sample counts.
    """
    (final_results, abl_tta, abl_ensembling, abl_tta_ensembling,
     abl_ensembling_all, abl_tta_ensembling_all) = load_ablation_results(
        skey, participants, basepath
    )

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(200 * MM_TO_INCH, 75 * MM_TO_INCH), dpi=300)

    all_ensembling, all_tta = [], []

    for i, p in enumerate(participants):
        color  = CB_COLORS[i % len(CB_COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        in_ensembling = p in abl_ensembling
        in_tta        = p in abl_tta

        if in_tta and in_ensembling:
            x = [1, 2, 3, 4]
            y = [abl_tta_ensembling[p], abl_ensembling[p],
                 abl_tta[p],            final_results[p]]
            ax.plot(x, y, color=color, marker=marker, markersize=7,
                    linewidth=1.2, alpha=0.6, label=p)
            avg_ens = [0.5 * (abl_tta_ensembling[p] + abl_ensembling[p]),
                       0.5 * (abl_tta[p] + final_results[p])]
            avg_tta = [0.5 * (abl_tta_ensembling[p] + abl_tta[p]),
                       0.5 * (abl_ensembling[p] + final_results[p])]
            ax.plot([5, 6], avg_ens, color=color, marker=marker, markersize=7,
                    linewidth=2, alpha=0.6)
            ax.plot([7, 8], avg_tta, color=color, marker=marker, markersize=7,
                    linewidth=2, alpha=0.6)
            all_ensembling.append(avg_ens)
            all_tta.append(avg_tta)

        elif in_ensembling:
            x = [5, 6]
            y = [abl_ensembling[p], final_results[p]]
            ax.plot(x, y, color=color, marker=marker, markersize=7,
                    linewidth=1.2, alpha=0.6, label=p)
            all_ensembling.append(y)

        elif in_tta:
            x = [7, 8]
            y = [abl_tta[p], final_results[p]]
            ax.plot(x, y, color=color, marker=marker, markersize=7,
                    linewidth=1.2, alpha=0.6, label=p)
            all_tta.append(y)

        else:
            continue

    # Mean lines and Δ annotations
    if all_ensembling:
        mean_ens = np.mean(all_ensembling, axis=0)
        delta    = np.mean(np.diff(all_ensembling, axis=1))
        ax.plot([5, 6], mean_ens, color="grey", linestyle=":", linewidth=1.5)
        sign = "+" if delta >= 0 else ""
        ax.text(5.5, mean_ens.mean(), f"{sign}{delta:.3f}",
                ha="center", va="top", fontsize=7)

    if all_tta:
        mean_tta = np.mean(all_tta, axis=0)
        delta    = np.mean(np.diff(all_tta, axis=1))
        ax.plot([7, 8], mean_tta, color="grey", linestyle=":", linewidth=1.5)
        sign = "+" if delta >= 0 else ""
        ax.text(7.5, mean_tta.mean(), f"{sign}{delta:.3f}",
                ha="center", va="top", fontsize=7)

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xticklabels(_X_LABELS_FULL, fontsize=6.5, rotation=90)
    ax.set_ylabel(metric_label, fontweight="bold", fontsize=8)
    ax.axvspan(0.5, 4.5, color="#f2f2f2", alpha=0.8, zorder=0)
    ax.axvspan(6.5, 8.5, color="#f2f2f2", alpha=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    y_lim = ax.get_ylim()
    ax.text(2.5, y_lim[1], "Complete Ablation",
            ha="center", va="bottom", fontweight="bold", size=7, alpha=0.6)
    ax.text(5.5, y_lim[1], f"Ablate Ensembling (n={len(all_ensembling)})",
            ha="center", va="bottom", size=7, fontweight="bold", alpha=0.6)
    ax.text(7.5, y_lim[1], f"Ablate TTA (n={len(all_tta)})",
            ha="center", va="bottom", fontweight="bold", size=7, alpha=0.6)

    ax.legend(title="Participants", bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=False, fontsize=6)

    summary = {}
    if all_ensembling:
        diffs = np.diff(np.array(all_ensembling), axis=1).flatten()
        summary["ens_mean"]   = float(np.mean(diffs))
        summary["ens_median"] = float(np.median(diffs))
        summary["n_ens"]      = len(all_ensembling)
    if all_tta:
        diffs = np.diff(np.array(all_tta), axis=1).flatten()
        summary["tta_mean"]   = float(np.mean(diffs))
        summary["tta_median"] = float(np.median(diffs))
        summary["n_tta"]      = len(all_tta)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Saved → {save_path}")

    return summary


def plot_ablation_studies_combined(outdir="."):
    """
    Combined figure for the paper: T1 overall F1 (top) + T2 balanced accuracy (bottom).
    Paper figure: fig:ablation
    """
    style_rcparams()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(200 * MM_TO_INCH, 150 * MM_TO_INCH),
        dpi=300,
        gridspec_kw={"hspace": 0.85},
    )
    fig.patch.set_facecolor("white")

    plot_ablation(
        skey="f1_score",
        participants=T1_PARTICIPANTS,
        basepath=TRACK1_OUTPUT_PATH,
        metric_label="$F_1$ score (all ROI types)",
        ax=axes[0],
    )
    axes[0].set_title("Track 1 — Mitosis Detection", fontsize=9,
                      fontweight="bold", color=NAVY, pad=6)

    plot_ablation(
        skey="overall_balanced_accuracy",
        participants=T2_PARTICIPANTS,
        basepath=TRACK2_OUTPUT_PATH,
        metric_label="Balanced accuracy",
        ax=axes[1],
    )
    axes[1].set_title("Track 2 — Atypical MF Classification", fontsize=9,
                      fontweight="bold", color=NAVY, pad=6)

    for ax, label in zip(axes, ["a", "b"]):
        ax.text(-0.06, 1.06, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=NAVY, va="top", clip_on=False)

    save_path = os.path.join(outdir, "ablation_studies.pdf")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate ablation study figures.")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()
    out = args.outdir

    style_rcparams()

    # Combined paper figure
    plot_ablation_studies_combined(outdir=out)
