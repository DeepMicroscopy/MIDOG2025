"""
Track 2 analysis figures.

Produces:
  ba_roc_auc_tumor_type_t2.pdf     — BA + ROC-AUC boxplots per tumor type
  t2_ss_by_domain.pdf              — sensitivity vs. specificity scatter by tumor (with IQR)
  ss_heatmap.pdf                   — specificity/sensitivity per tumor × team heatmap

Paper figures: fig:ba_roc_t2, fig:kappa_t2
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats
from adjustText import adjust_text
import pandas as pd

from utils import (
    NAVY, MM_TO_INCH, TUMOR_TYPES, FONT_SERIF,
    load_track2_results, style_rcparams, PRECOMPUTED_PATH,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

_COLOR_BA   = "#0072B2"
_COLOR_AUC  = "#009E73"
_COLOR_SENS = "#D55E00"
_COLOR_SPEC = "#CC79A7"


def _domain_medians(tumor_type_scores, key):
    return np.array([
        np.median(tumor_type_scores[key][k])
        for k in range(len(TUMOR_TYPES))
    ])


# ── BA + ROC boxplots ──────────────────────────────────────────────────────────

def plot_ba_roc_by_tumor(tumor_type_scores, save_path="ba_roc_auc_tumor_type_t2.pdf"):
    """
    Side-by-side boxplots: balanced accuracy (left) and ROC-AUC (right) per tumor type.
    Paper figure: fig:ba_roc_t2
    """
    style_rcparams()

    ba_data  = [tumor_type_scores["BA"][k]    for k in range(len(TUMOR_TYPES))]
    auc_data = [tumor_type_scores["AUROC"][k] for k in range(len(TUMOR_TYPES))]

    fig, axs = plt.subplots(1, 2, figsize=(90 * MM_TO_INCH * 2, 65 * MM_TO_INCH),
                            dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")

    axs[0].boxplot(ba_data, patch_artist=True,
                   boxprops=dict(facecolor=_COLOR_BA, alpha=0.5))
    axs[0].set_xticks(range(1, len(TUMOR_TYPES) + 1))
    axs[0].set_xticklabels(TUMOR_TYPES, rotation=90, fontsize=6.5)
    axs[0].set_ylabel("Balanced accuracy", fontsize=7)
    axs[0].set_ylim(0.4, 1.01)

    axs[1].boxplot(auc_data, patch_artist=True,
                   boxprops=dict(facecolor=_COLOR_AUC, alpha=0.5))
    axs[1].set_xticks(range(1, len(TUMOR_TYPES) + 1))
    axs[1].set_xticklabels(TUMOR_TYPES, rotation=90, fontsize=6.5)
    axs[1].set_ylabel("ROC AUC", fontsize=7)
    axs[1].set_ylim(0.4, 1.01)

    for ax in axs:
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    for ax, label in zip(axs, ["a", "b"]):
        ax.text(-0.15, 1.06, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=NAVY, va="top", clip_on=False)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Sensitivity vs. specificity scatter ────────────────────────────────────────

def plot_t2_ss_by_domain(tumor_type_scores, save_path="t2_ss_by_domain.pdf"):
    """
    Scatter: median sensitivity vs. median specificity per tumor type, with IQR bars.
    """
    style_rcparams()

    fig, ax = plt.subplots(figsize=(80 * MM_TO_INCH, 75 * MM_TO_INCH), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FFFFFF")

    cmap = plt.cm.get_cmap("tab20", len(TUMOR_TYPES))
    texts = []
    for i, tumor in enumerate(TUMOR_TYPES):
        spec = tumor_type_scores["specificity"].get(i, [])
        sens = tumor_type_scores["sensitivity"].get(i, [])
        if not spec or not sens:
            continue
        spec, sens = np.array(spec), np.array(sens)
        med_spec, med_sens = np.median(spec), np.median(sens)
        q1s, q3s = np.percentile(spec, 25), np.percentile(spec, 75)
        q1r, q3r = np.percentile(sens, 25), np.percentile(sens, 75)
        color = cmap(i)
        ax.plot([q1s, q3s], [med_sens, med_sens], color=color, linewidth=1.2, alpha=0.6, zorder=3)
        ax.plot([med_spec, med_spec], [q1r, q3r], color=color, linewidth=1.2, alpha=0.6, zorder=3)
        ax.scatter(med_spec, med_sens, s=55, color=color, edgecolors="white",
                   linewidths=0.5, zorder=5, label=tumor)
        texts.append(ax.annotate(tumor, (med_spec, med_sens),
                                  xytext=(5, 3), textcoords="offset points",
                                  fontsize=6, color=color, fontweight="bold", clip_on=True))

    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("Specificity (normal MF recall)", fontsize=7, labelpad=4)
    ax.set_ylabel("Sensitivity (atypical MF recall)", fontsize=7, labelpad=4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Sensitivity/Specificity by-team heatmap ────────────────────────────────────

def plot_ss_decomposition_heatmap(tumor_type_scores, participants,
                                   save_path="ss_heatmap.pdf"):
    """
    Heatmap: specificity (left) + sensitivity (right), rows = tumor types, cols = teams.
    """
    style_rcparams()
    n_tumors = len(TUMOR_TYPES)
    n_teams  = len(participants)

    spec_matrix = np.full((n_tumors, n_teams), np.nan)
    sens_matrix = np.full((n_tumors, n_teams), np.nan)

    for i in range(n_tumors):
        spec = tumor_type_scores["specificity"].get(i, [])
        sens = tumor_type_scores["sensitivity"].get(i, [])
        if spec:
            spec_matrix[i, :len(spec)] = spec
        if sens:
            sens_matrix[i, :len(sens)] = sens

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, matrix, title, cmap in zip(
        axes, [spec_matrix, sens_matrix], ["Specificity", "Sensitivity"], ["Blues", "Oranges"]
    ):
        im = ax.imshow(matrix, vmin=0.2, vmax=1.0, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_teams))
        ax.set_xticklabels(participants, fontsize=8, rotation=90)
        ax.set_yticks(range(n_tumors))
        ax.set_yticklabels(TUMOR_TYPES, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=8)
        for ii in range(n_tumors):
            for jj in range(n_teams):
                val = matrix[ii, jj]
                if not np.isnan(val):
                    ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if val > 0.5 else NAVY)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── BA/AUROC by-team heatmap ──────────────────────────────────────────────────

def plot_mm_decomposition_heatmap(tumor_type_scores, participants,
                                   save_path="mm_track2_heatmap.pdf"):
    """
    Heatmap: balanced accuracy (left) + ROC-AUC (right), rows = tumor types, cols = teams.
    """
    style_rcparams()
    n_tumors = len(TUMOR_TYPES)
    n_teams  = len(participants)

    ba_matrix   = np.full((n_tumors, n_teams), np.nan)
    auc_matrix  = np.full((n_tumors, n_teams), np.nan)

    for i in range(n_tumors):
        ba  = tumor_type_scores["BA"].get(i, [])
        auc = tumor_type_scores["AUROC"].get(i, [])
        if ba:
            ba_matrix[i, :len(ba)] = ba
        if auc:
            auc_matrix[i, :len(auc)] = auc

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, matrix, title, cmap, vmin in zip(
        axes, [ba_matrix, auc_matrix], ["Balanced Accuracy", "ROC AUC"],
        ["Greens", "Purples"], [0.2, 0.8]
    ):
        im = ax.imshow(matrix, vmin=vmin, vmax=1.0, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_teams))
        ax.set_xticklabels(participants, fontsize=8, rotation=90)
        ax.set_yticks(range(n_tumors))
        ax.set_yticklabels(TUMOR_TYPES, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=8)
        for ii in range(n_tumors):
            for jj in range(n_teams):
                val = matrix[ii, jj]
                if not np.isnan(val):
                    ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if val > 0.5 else NAVY)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Kappa vs. sens/spec (1×3) ──────────────────────────────────────────────────

def _plot_sens_spec_scatter(ax, sens_vals, spec_vals, kappa_vals, labels):
    cmap = cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.2, vmax=1.0)

    s_grid = np.linspace(0.5, 1.07, 300)
    for ba_target in [0.75, 0.80, 0.85, 0.90, 0.95]:
        spec_iso = 2 * ba_target - s_grid
        mask = (spec_iso >= 0.52) & (spec_iso <= 1.07)
        ax.plot(s_grid[mask], spec_iso[mask], color="#CCCCCC", linewidth=0.5,
                linestyle="--", zorder=1)
        x_label = 2 * ba_target - 1.06
        if 0.52 < x_label < 1.07:
            ax.text(x_label - 0.005, 1.0, f"BA={ba_target:.2f}",
                    fontsize=4.5, color="#AAAAAA", va="bottom", ha="center", rotation=-55)

    ax.plot([0.5, 1.05], [0.5, 1.05], color="#AAAAAA", linewidth=0.6, linestyle=":", zorder=1)

    colors = [cmap(norm(k)) for k in kappa_vals]
    for x, y, c in zip(sens_vals, spec_vals, colors):
        ax.scatter(x, y, color=c, s=35, edgecolors="white", linewidths=0.4, zorder=4)

    texts = []
    for x, y, name in zip(sens_vals, spec_vals, labels):
        texts.append(ax.text(x, y, name, fontsize=5.5, color=NAVY, clip_on=False, zorder=6))
    adjust_text(texts, ax=ax, expand=(1.5, 1.5),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.4, shrinkA=2, shrinkB=2))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=5.5)
    cbar.set_label(r"Cohen's $\kappa$ (R1 vs R2)", fontsize=6, labelpad=4, loc="top")

    ax.set_xlim(0.52, 1.07)
    ax.set_ylim(0.52, 1.07)
    ax.set_xlabel("Sensitivity (atypical MF recall)", fontsize=7, labelpad=4)
    ax.set_ylabel("Specificity (normal MF recall)", fontsize=7, labelpad=4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def _scatter_kappa_vs_metric_t2(ax, kappa_vals, metric_vals, labels,
                                  color, metric_name, show_ylabel=True):
    kappa_vals  = np.array(kappa_vals)
    metric_vals = np.array(metric_vals)
    rho, p_val  = stats.spearmanr(kappa_vals, metric_vals)

    slope, intercept, *_ = stats.linregress(kappa_vals, metric_vals)
    x_fit = np.linspace(kappa_vals.min(), kappa_vals.max(), 300)
    ax.plot(x_fit, np.clip(slope * x_fit + intercept, 0, 1),
            color="#888888", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)

    ax.scatter(kappa_vals, metric_vals, color=color, s=28,
               edgecolors="white", linewidths=0.4, zorder=4)

    texts = []
    for x, y, name in zip(kappa_vals, metric_vals, labels):
        texts.append(ax.text(x, y, name, fontsize=5.5, color=NAVY, clip_on=False, zorder=6))
    adjust_text(texts, x=kappa_vals, y=metric_vals, ax=ax,
                expand=(2.5, 2.0), force_points=(0.3, 0.5), force_text=(0.3, 0.5),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                lw=0.4, shrinkA=2, shrinkB=2),
                only_move={"points": "y", "texts": "xy"})

    p_str = r"$p$ < 0.001" if p_val < 0.001 else rf"$p$ = {p_val:.3f}"
    ax.text(0.97, 0.05,
            rf"$\rho_s$ = {rho:.2f}" + "\n" + p_str,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color=NAVY, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", linewidth=0.5, alpha=0.9))

    ax.set_xlabel(r"Cohen's $\kappa$ (R1 vs R2)", fontsize=7, labelpad=4)
    if show_ylabel:
        ax.set_ylabel(metric_name, fontsize=7, labelpad=4)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.52, 1.07)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def plot_t2_kappa_vs_sens_spec(tumor_type_scores, pivot, save_path="t2_figure_kappa_vs_sens_spec.pdf"):
    """
    1×3 figure:
      a) sensitivity vs. specificity (colored by kappa)
      b) kappa vs. sensitivity
      c) kappa vs. specificity
    Paper figure: fig:kappa_t2
    """
    style_rcparams()

    kappa_r1r2 = pivot["R1 vs R2"]
    kappa_vals = np.array([kappa_r1r2.get(t, np.nan) for t in TUMOR_TYPES])
    valid  = ~np.isnan(kappa_vals)
    labels = [TUMOR_TYPES[i] for i in range(len(TUMOR_TYPES)) if valid[i]]
    kappa_v = kappa_vals[valid]
    sens_v  = _domain_medians(tumor_type_scores, "sensitivity")[valid]
    spec_v  = _domain_medians(tumor_type_scores, "specificity")[valid]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(180 * MM_TO_INCH, 55 * MM_TO_INCH),
        dpi=300,
        gridspec_kw={"wspace": 0.60},
    )
    fig.patch.set_facecolor("white")

    _plot_sens_spec_scatter(axes[0], sens_v, spec_v, kappa_v, labels)
    _scatter_kappa_vs_metric_t2(axes[1], kappa_v, sens_v, labels,
                                 color=_COLOR_SENS,
                                 metric_name="Sensitivity (atypical MF)",
                                 show_ylabel=True)
    _scatter_kappa_vs_metric_t2(axes[2], kappa_v, spec_v, labels,
                                 color=_COLOR_SPEC,
                                 metric_name="Specificity (normal MF)",
                                 show_ylabel=True)

    for ax, label in zip(axes, ["a)", "b)", "c)"]):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=NAVY, va="top", clip_on=False)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


def plot_t2_kappa_vs_performance(tumor_type_scores, pivot, save_path="t2_kappa_vs_performance.pdf"):
    """
    2×2 figure: annotator kappa vs. BA, AUC, sensitivity, specificity.
    """
    style_rcparams()

    kappa_r1r2 = pivot["R1 vs R2"]
    kappa_vals = np.array([kappa_r1r2.get(t, np.nan) for t in TUMOR_TYPES])
    valid  = ~np.isnan(kappa_vals)
    labels = [TUMOR_TYPES[i] for i in range(len(TUMOR_TYPES)) if valid[i]]

    ba_vals   = _domain_medians(tumor_type_scores, "BA")
    auc_vals  = _domain_medians(tumor_type_scores, "AUROC")
    sens_vals = _domain_medians(tumor_type_scores, "sensitivity")
    spec_vals = _domain_medians(tumor_type_scores, "specificity")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(130 * MM_TO_INCH, 110 * MM_TO_INCH),
        dpi=300,
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0, 0], ba_vals,   _COLOR_BA,   "Balanced accuracy",        True),
        (axes[0, 1], auc_vals,  _COLOR_AUC,  "AUC-ROC",                  False),
        (axes[1, 0], sens_vals, _COLOR_SENS, "Sensitivity (atypical MF)", True),
        (axes[1, 1], spec_vals, _COLOR_SPEC, "Specificity (normal MF)",   False),
    ]

    for ax, metric_vals, color, name, show_ylabel in panels:
        _scatter_kappa_vs_metric_t2(ax, kappa_vals[valid], metric_vals[valid],
                                     labels, color=color, metric_name=name,
                                     show_ylabel=show_ylabel)

    axes[0, 1].set_ylim(0.85, 1)
    for ax, label in zip(axes.flat, ["a", "b", "c", "d"]):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=NAVY, va="top", clip_on=False)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Track 2 analysis figures.")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()
    out = args.outdir

    print("Loading Track 2 results…")
    t2 = load_track2_results()
    tts = t2["tumor_type_scores"]
    participants = list(t2["balanced_accuracy"].keys())

    # BA + ROC boxplots
    plot_ba_roc_by_tumor(tts, save_path=os.path.join(out, "ba_roc_auc_tumor_type_t2.pdf"))

    # Sensitivity vs. specificity scatter
    plot_t2_ss_by_domain(tts, save_path=os.path.join(out, "t2_ss_by_domain.pdf"))

    # By-team heatmaps
    plot_ss_decomposition_heatmap(tts, participants,
                                   save_path=os.path.join(out, "ss_heatmap.pdf"))
