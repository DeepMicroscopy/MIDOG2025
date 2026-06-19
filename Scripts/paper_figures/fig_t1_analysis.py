"""
Track 1 analysis figures.

Produces:
  f1_heatmap.pdf                       — median F1 heatmap (tumor × ROI) + drop columns
  precision-recall-area-type.pdf         — 3×3 figure: PR scatter + density vs. precision/recall
  pr_heatmap-by-team.pdf               — precision/recall per tumor × team heatmap
  mm_heatmap-by-team.pdf               — F1/FROC per tumor × team heatmap

Paper figures: fig:f1_heatmap, fig:3x3_pr, fig:mm_heatmap, fig:kappa
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from adjustText import adjust_text
import pandas as pd

from utils import (
    NAVY, MM_TO_INCH, TUMOR_TYPES, ROI_TYPES, ROI_COLORS, FONT_SERIF,
    load_track1_results, style_rcparams, EVALUATIONS_PATH, PRECOMPUTED_PATH,
)


# ── F1 heatmap ─────────────────────────────────────────────────────────────────

def plot_f1_heatmap(roitype_tumortype, save_path="f1_heatmap.pdf"):
    """
    Heatmap: median F1 per tumor × ROI type + two drop columns.
    Paper figure: fig:f1_heatmap
    """
    style_rcparams()
    plt.rcParams.update({"xtick.labelsize": 7, "ytick.labelsize": 7})

    n_tumors = len(TUMOR_TYPES)
    n_roi    = len(ROI_TYPES)

    f1_median = np.full((n_tumors, n_roi), np.nan)
    f1_iqr    = np.full((n_tumors, n_roi), np.nan)

    for i, tumor in enumerate(TUMOR_TYPES):
        for j, roi in enumerate(ROI_TYPES):
            vals = roitype_tumortype.get(f"{i}_{roi}_f1", [])
            if vals:
                f1_median[i, j] = np.median(vals)
                f1_iqr[i, j]    = np.percentile(vals, 75) - np.percentile(vals, 25)

    drop_hr = f1_median[:, 0] - f1_median[:, 1]   # hotspot − random
    drop_hc = f1_median[:, 0] - f1_median[:, 2]   # hotspot − challenging

    sort_idx        = np.argsort(drop_hr)[::-1]
    f1_sorted       = f1_median[sort_idx]
    iqr_sorted      = f1_iqr[sort_idx]
    drop_sorted     = drop_hr[sort_idx]
    drop_hc_sorted  = drop_hc[sort_idx]
    labels_sorted   = [TUMOR_TYPES[i] for i in sort_idx]

    fig = plt.figure(figsize=(80 * MM_TO_INCH, 60 * MM_TO_INCH), dpi=300)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 0.8, 0.8], wspace=0.08)
    ax_main  = fig.add_subplot(gs[0])
    ax_dropR = fig.add_subplot(gs[1])
    ax_dropC = fig.add_subplot(gs[2])

    im = ax_main.imshow(f1_sorted, vmin=0.2, vmax=0.9, cmap="RdYlGn", aspect="auto")
    ax_main.set_xticks(range(n_roi))
    ax_main.set_xticklabels([r.capitalize() for r in ROI_TYPES],
                             fontsize=7, fontweight="bold")
    ax_main.set_yticks(range(n_tumors))
    ax_main.set_yticklabels(labels_sorted, fontsize=7)
    ax_main.set_title("Median $F_1$ score", fontsize=8, fontweight="bold",
                      color=NAVY, pad=6)

    for ii in range(n_tumors):
        for jj in range(n_roi):
            val = f1_sorted[ii, jj]
            iqr = iqr_sorted[ii, jj]
            if not np.isnan(val):
                txt_color = "white" if (val < 0.4 or val > 0.78) else NAVY
                ax_main.text(jj, ii, f"{val:.3f}\n±{iqr:.3f}",
                             ha="center", va="center", fontsize=5,
                             color=txt_color, linespacing=1.3)

    max_drop = np.nanmax(np.abs(np.concatenate([drop_sorted, drop_hc_sorted])))
    norm = TwoSlopeNorm(vmin=-max_drop, vcenter=0, vmax=max_drop)

    for ax_drop, drop_vals, title in zip(
        [ax_dropR, ax_dropC],
        [drop_sorted, drop_hc_sorted],
        ["Δ hot−rand", "Δ hot−chall"]
    ):
        drop_2d = drop_vals.reshape(-1, 1)
        ax_drop.imshow(drop_2d, cmap="RdYlGn_r", norm=norm, aspect="auto")
        ax_drop.set_xticks([0])
        ax_drop.set_xticklabels([title], fontsize=6, fontweight="bold",
                                 rotation=30, ha="right")
        ax_drop.set_yticks([])
        for ii, val in enumerate(drop_vals):
            if not np.isnan(val):
                txt_color = "white" if abs(val) > 0.25 else NAVY
                ax_drop.text(0, ii, f"{val:+.2f}",
                             ha="center", va="center", fontsize=5.5, color=txt_color)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── 3×3 PR figure ──────────────────────────────────────────────────────────────

def _plot_pr_row(results, axes):
    for ax, roi in zip(axes, ROI_TYPES):
        ax.set_facecolor("#FAFAFA")
        color = ROI_COLORS[roi]

        p_grid = np.linspace(0.01, 1.0, 300)
        for f1_target in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            with np.errstate(divide="ignore", invalid="ignore"):
                r_grid = f1_target * p_grid / (2 * p_grid - f1_target)
            mask = (r_grid >= 0) & (r_grid <= 1.0)
            ax.plot(p_grid[mask], r_grid[mask],
                    color="#CCCCCC", linewidth=0.5, linestyle="--", zorder=1)
            valid = np.where(mask)[0]
            if len(valid):
                ax.text(p_grid[valid[-1]] + 0.01, r_grid[valid[-1]],
                        f"F1={f1_target:.1f}", fontsize=4.5, color="#AAAAAA", va="center")

        texts = []
        for i, tumor in enumerate(TUMOR_TYPES):
            prec = results.get(f"{i}_{roi}_precision", [])
            rec  = results.get(f"{i}_{roi}_recall", [])
            if not prec or not rec:
                continue
            prec, rec = np.array(prec), np.array(rec)
            med_p, med_r = np.median(prec), np.median(rec)
            q1_p, q3_p   = np.percentile(prec, 25), np.percentile(prec, 75)
            q1_r, q3_r   = np.percentile(rec,  25), np.percentile(rec,  75)

            ax.plot([q1_p, q3_p], [med_r, med_r], color=color, linewidth=0.8, alpha=0.5, zorder=3)
            ax.plot([med_p, med_p], [q1_r, q3_r], color=color, linewidth=0.8, alpha=0.5, zorder=3)
            ax.scatter(med_p, med_r, s=20, color=color, edgecolors="white", linewidths=0.3, zorder=5)
            texts.append(ax.text(med_p, med_r, tumor, fontsize=5, color=NAVY, clip_on=True))

        adjust_text(texts, ax=ax, expand=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                    lw=0.4, shrinkA=2, shrinkB=2))
        ax.set_xlim(0.0, 1.08)
        ax.set_ylim(0.0, 1.08)
        ax.set_xlabel("Precision", fontsize=7, labelpad=4)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[0].set_ylabel("Recall", fontsize=7, labelpad=4)


def _plot_density_vs_metric(results, metric, metric_label, axes, show_xlabel=True):
    for ax, roi in zip(axes, ROI_TYPES):
        ax.set_facecolor("#FAFAFA")
        color = ROI_COLORS[roi]
        counts, values, labels = [], [], []

        for i, tumor in enumerate(TUMOR_TYPES):
            pos = results.get(f"{i}_{roi}_positives", [])
            met = results.get(f"{i}_{roi}_{metric}", [])
            if not pos or not met:
                continue
            counts.append(np.median(pos))
            values.append(np.median(met))
            labels.append(tumor)

        counts = np.array(counts)
        values = np.array(values)
        rho, p_val = stats.spearmanr(counts, values) if len(counts) > 2 else (np.nan, np.nan)

        if len(counts) > 2:
            slope, intercept, *_ = stats.linregress(counts, values)
            x_fit = np.linspace(counts.min(), counts.max(), 300)
            ax.plot(x_fit, np.clip(slope * x_fit + intercept, 0, 1),
                    color="#888888", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)

        ax.scatter(counts, values, color=color, s=20, edgecolors="white",
                   linewidths=0.3, zorder=4)
        texts = []
        for x, y, name in zip(counts, values, labels):
            texts.append(ax.text(x, y, name, fontsize=5, color=NAVY, clip_on=True))
        adjust_text(texts, ax=ax, expand=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                    lw=0.4, shrinkA=2, shrinkB=2))

        p_str = r"$p$ < 0.001" if p_val < 0.001 else rf"$p$ = {p_val:.3f}"
        ax.text(0.97, 0.05,
                rf"$\rho_s$ = {rho:.2f}" + "\n" + p_str,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.5, color=NAVY, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=0.5, alpha=0.9))

        ax.set_ylim(-0.02, 1.08)
        ax.set_xlim(left=0)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        if show_xlabel:
            ax.set_xlabel("Median mitotic count per ROI", fontsize=7, labelpad=4)
    axes[0].set_ylabel(metric_label, fontsize=7, labelpad=4)


def plot_3x3_pr(roitype_tumortype, save_path="precision-recall-area-type.pdf"):
    """
    3×3 figure: PR scatter (row 1) + precision vs. density (row 2) + recall vs. density (row 3).
    Paper figure: fig:3x3_pr
    """
    style_rcparams()

    fig, axes = plt.subplots(
        3, 3,
        figsize=(180 * MM_TO_INCH, 60 * MM_TO_INCH * 3),
        dpi=300,
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )
    fig.patch.set_facecolor("white")

    _plot_pr_row(roitype_tumortype, axes[0])
    _plot_density_vs_metric(roitype_tumortype, "precision", "Median precision",
                            axes[1], show_xlabel=False)
    _plot_density_vs_metric(roitype_tumortype, "recall", "Median recall",
                            axes[2], show_xlabel=True)

    for ax, roi in zip(axes[0], ROI_TYPES):
        ax.set_title(f"{roi} ROIs", fontsize=8, fontweight="bold",
                     color=ROI_COLORS[roi], pad=6)

    for ax, label in zip(axes.flat, list("abcdefghi")):
        ax.text(-0.18, 1.06, label + ")",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                color=NAVY, va="top")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Precision/Recall by-team heatmap ──────────────────────────────────────────

def plot_pr_by_team_heatmap(tumor_type_scores, participants,
                            save_path="pr_heatmap-by-team.pdf"):
    """
    Heatmap: precision (left) + recall (right), rows = tumor types, cols = teams.
    """
    style_rcparams()
    n_tumors = len(TUMOR_TYPES)
    n_teams  = len(participants)

    prec_matrix = np.full((n_tumors, n_teams), np.nan)
    rec_matrix  = np.full((n_tumors, n_teams), np.nan)

    for i in range(n_tumors):
        prec = tumor_type_scores["precision"].get(i, [])
        rec  = tumor_type_scores["recall"].get(i, [])
        if prec:
            prec_matrix[i, :len(prec)] = prec
        if rec:
            rec_matrix[i, :len(rec)] = rec

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, matrix, title, cmap in zip(
        axes, [prec_matrix, rec_matrix], ["Precision", "Recall"], ["Blues", "Oranges"]
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


# ── F1/FROC by-team heatmap ───────────────────────────────────────────────────

def plot_mm_by_team_heatmap(tumor_type_scores, participants,
                             save_path="mm_heatmap-by-team.pdf"):
    """
    Heatmap: F1 (left) + FROC-AUC (right), rows = tumor types, cols = teams.
    """
    style_rcparams()
    n_tumors = len(TUMOR_TYPES)
    n_teams  = len(participants)

    f1_matrix   = np.full((n_tumors, n_teams), np.nan)
    froc_matrix = np.full((n_tumors, n_teams), np.nan)

    for i in range(n_tumors):
        f1_vals   = tumor_type_scores["F1"].get(i, [])
        froc_vals = tumor_type_scores["froc_auc"].get(i, [])
        if f1_vals:
            f1_matrix[i, :len(f1_vals)] = f1_vals
        if froc_vals:
            froc_matrix[i, :len(froc_vals)] = froc_vals

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, matrix, title, cmap, vmax in zip(
        axes, [f1_matrix, froc_matrix], ["$F_1$ score", "FROC AUC"],
        ["Greens", "Purples"], [1.0, 7.0]
    ):
        im = ax.imshow(matrix, vmin=0.2, vmax=vmax, cmap=cmap, aspect="auto")
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


# ── Kappa vs. performance ──────────────────────────────────────────────────────

def _scatter_kappa_vs_metric_t1(ax, kappa_vals, metric_vals, labels,
                                 color, metric_name, clip_val_max=1.0, show_ylabel=False):
    kappa_vals  = np.array(kappa_vals)
    metric_vals = np.array(metric_vals)
    rho, p_val  = stats.spearmanr(kappa_vals, metric_vals)

    slope, intercept, *_ = stats.linregress(kappa_vals, metric_vals)
    x_fit = np.linspace(kappa_vals.min(), kappa_vals.max(), 300)
    ax.plot(x_fit, np.clip(slope * x_fit + intercept, 0, clip_val_max),
            color="#888888", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)

    ax.scatter(kappa_vals, metric_vals, color=color, s=28,
               edgecolors="white", linewidths=0.4, zorder=4)

    texts = []
    for x, y, name in zip(kappa_vals, metric_vals, labels):
        texts.append(ax.text(x, y, name, fontsize=5.5, color=NAVY, clip_on=True))
    adjust_text(texts, ax=ax, expand=(1.2, 1.4),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                lw=0.4, shrinkA=2, shrinkB=2))

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
    ax.set_title(metric_name, fontsize=7.5, fontweight="bold", color=NAVY, pad=5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, clip_val_max * 1.02)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def plot_t1_kappa_vs_performance(tumor_type_scores, results_df,
                                  save_path="t1_kappa_vs_performance.pdf"):
    """
    2×2 figure: annotator kappa (R1 vs R2) vs. F1, FROC, recall, precision.
    Paper figure: fig:kappa_t1
    """
    style_rcparams()

    kappa_r1r2 = (
        results_df[results_df["Comparison"] == "R1 vs R2"]
        .set_index("Tumor_Type")["Cohen_kappa"]
    )
    kappa_vals = np.array([kappa_r1r2.get(t, np.nan) for t in TUMOR_TYPES])
    valid  = ~np.isnan(kappa_vals)
    labels = [TUMOR_TYPES[i] for i in range(len(TUMOR_TYPES)) if valid[i]]

    def domain_medians(key):
        return np.array([np.median(tumor_type_scores[key][k])
                         for k in range(len(TUMOR_TYPES))])

    f1_vals   = domain_medians("F1")
    froc_vals = domain_medians("froc_auc")
    rec_vals  = domain_medians("recall")
    prec_vals = domain_medians("precision")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(130 * MM_TO_INCH, 110 * MM_TO_INCH),
        dpi=300,
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )
    fig.patch.set_facecolor("white")

    COLOR_F1   = "#0072B2"
    COLOR_FROC = "#009E73"
    COLOR_REC  = "#D55E00"
    COLOR_PREC = "#CC79A7"

    panels = [
        (axes[0, 0], f1_vals,   COLOR_F1,   "$F_1$ score",  1.0, True),
        (axes[0, 1], froc_vals, COLOR_FROC, "FROC AUC",     8.0, False),
        (axes[1, 0], rec_vals,  COLOR_REC,  "Recall",       1.0, True),
        (axes[1, 1], prec_vals, COLOR_PREC, "Precision",    1.0, False),
    ]
    for ax, metric_vals, color, name, clip_max, show_ylabel in panels:
        _scatter_kappa_vs_metric_t1(ax, kappa_vals[valid], metric_vals[valid],
                                    labels, color=color, metric_name=name,
                                    clip_val_max=clip_max, show_ylabel=show_ylabel)

    axes[0, 1].set_ylim([2, 8])
    for ax, label in zip(axes.flat, ["a", "b", "c", "d"]):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=NAVY, va="top", clip_on=False)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from sklearn.metrics import cohen_kappa_score

    parser = argparse.ArgumentParser(description="Generate Track 1 analysis figures.")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()
    out = args.outdir

    print("Loading Track 1 results…")
    t1 = load_track1_results()
    rtt = t1["roitype_tumortype"]
    tts = t1["tumor_type_scores"]
    participants = list(t1["f1_score"].keys())

    # F1 heatmap
    plot_f1_heatmap(rtt, save_path=os.path.join(out, "f1_heatmap.pdf"))

    # 3×3 PR figure
    plot_3x3_pr(rtt, save_path=os.path.join(out, "precision-recall-area-type.pdf"))

    # By-team heatmaps
    plot_pr_by_team_heatmap(tts, participants,
                            save_path=os.path.join(out, "pr_heatmap-by-team.pdf"))
    plot_mm_by_team_heatmap(tts, participants,
                            save_path=os.path.join(out, "mm_heatmap-by-team.pdf"))

