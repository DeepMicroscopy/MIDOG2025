"""
Rank and score stability figure.

Produces:
  rank_score_stability.pdf  — 3-panel figure: rank bump chart, F1 slope graph,
                              and Pearson r bar chart for all pairwise ROI-type
                              comparisons.

Paper figure: fig:score_stability
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

from utils import (
    NAVY, MM_TO_INCH, ROI_COLORS, FONT_SERIF,
    CB_COLORS,
    load_track1_results, style_rcparams,
)

# Per-team color / line-style maps (Okabe-Ito, 8 colors, 2 line styles each)
_USERNAME_CMAP = {
    "wildsquirrel":      CB_COLORS[0],
    "RaphaelBourgade":   CB_COLORS[1],
    "ytopuz53":          CB_COLORS[2],
    "cacfek":            CB_COLORS[3],
    "navyasri.kelam":    CB_COLORS[4],
    "christian.marzahl": CB_COLORS[5],
    "tengyoux":          CB_COLORS[6],
    "masarno":           CB_COLORS[7],
    "piotrgiedziun":     CB_COLORS[0],
    "ammeling":          CB_COLORS[1],
    "SZTU-134":          CB_COLORS[2],
    "schoe":             CB_COLORS[3],
    "vidushiwalia":      CB_COLORS[4],
    "npic-ab":           CB_COLORS[5],
}
_USERNAME_LS = {
    "wildsquirrel":      "solid",
    "RaphaelBourgade":   "dotted",
    "ytopuz53":          "dashed",
    "cacfek":            "dashdot",
    "navyasri.kelam":    "solid",
    "christian.marzahl": "dotted",
    "tengyoux":          "dashed",
    "masarno":           "dashdot",
    "piotrgiedziun":     "solid",
    "ammeling":          "dotted",
    "SZTU-134":          "dashed",
    "schoe":             "dashdot",
    "vidushiwalia":      "solid",
    "npic-ab":           "dotted",
}

PAIR_DEFS = [
    ("hotspot",     "random",      "Hot. vs Rand."),
    ("hotspot",     "challenging", "Hot. vs Chall."),
    ("hotspot",     "overall",     "Hot. vs Overall"),
    ("random",      "challenging", "Rand. vs Chall."),
    ("random",      "overall",     "Rand. vs Overall"),
    ("challenging", "overall",     "Chall. vs Overall"),
]
PAIR_COLORS = [
    ROI_COLORS["random"],
    ROI_COLORS["challenging"],
    ROI_COLORS["overall"],
    ROI_COLORS["challenging"],
    ROI_COLORS["overall"],
    ROI_COLORS["overall"],
]


def _pearson_ci(r, n):
    z    = np.arctanh(r)
    se   = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def plot_rank_score_stability(team_scores, save_path="rank_score_stability.pdf"):
    """
    3-panel figure:
      a) Rank bump chart across hotspot / random / challenging / overall
      b) F1 slope graph across the same ROI types (per-team color + line style)
      c) Pearson r bar chart for all pairwise comparisons with 95% CI

    team_scores : dict  {team_name: {roi_type: f1_score}}
    Paper figure: fig:score_stability
    """
    style_rcparams()

    roi_types = ["hotspot", "random", "challenging", "overall"]
    teams     = list(team_scores.keys())
    n_teams   = len(teams)

    scores_arr = {roi: np.array([team_scores[t][roi] for t in teams]) for roi in roi_types}
    ranks = {
        roi: pd.Series(scores_arr[roi], index=teams)
        .rank(ascending=False, method="min").astype(int)
        for roi in roi_types
    }
    rank_df  = pd.DataFrame(ranks)
    score_df = pd.DataFrame(
        {roi: [team_scores[t][roi] for t in teams] for roi in roi_types},
        index=teams,
    )

    # Pairwise Pearson correlations
    results = []
    for a, b, label in PAIR_DEFS:
        r, p   = stats.pearsonr(score_df[a], score_df[b])
        lo, hi = _pearson_ci(r, n_teams)
        results.append({"label": label, "r": r, "p": p, "lo": lo, "hi": hi})
    res_df = pd.DataFrame(results)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(180 * MM_TO_INCH, 70 * MM_TO_INCH),
        dpi=300,
        gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 1.8]},
    )
    fig.patch.set_facecolor("white")

    x_positions = [0, 1, 2, 3]
    x_labels    = ["Hotspot", "Random", "Challenging", "Overall"]

    # ── Panel a: rank bump chart ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    for team in teams:
        team_ranks = [rank_df.loc[team, roi] for roi in roi_types]
        ax.plot(x_positions, team_ranks,
                color=_USERNAME_CMAP.get(team, "#888888"),
                linestyle=_USERNAME_LS.get(team, "solid"),
                linewidth=0.8, alpha=0.8, zorder=3)
        for xi, yi in zip(x_positions, team_ranks):
            ax.scatter(xi, yi, color=_USERNAME_CMAP.get(team, "#888888"),
                       s=10, edgecolors="white", linewidths=0.3, zorder=4)
    ax.yaxis.set_inverted(True)
    ax.set_ylabel("Rank (1 = best)", fontsize=7, labelpad=4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=6.5, fontweight="bold", rotation=90)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    # ── Panel b: F1 score slope graph ─────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#ffffff")
    for team in teams:
        row = [score_df.loc[team, roi] for roi in roi_types]
        ax.plot(x_positions, row,
                color=_USERNAME_CMAP.get(team, "#888888"),
                linestyle=_USERNAME_LS.get(team, "solid"),
                linewidth=1.0, alpha=0.9, zorder=3, label=team)
        for xi, yi in zip(x_positions, row):
            ax.scatter(xi, yi, color=_USERNAME_CMAP.get(team, "#888888"),
                       s=12, edgecolors="white", linewidths=0.3, zorder=4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=6.5, fontweight="bold", rotation=90)
    ax.set_ylabel("$F_1$ score", fontsize=7, labelpad=4)
    ax.set_ylim(score_df[roi_types].values.min() - 0.02,
                score_df[roi_types].values.max() + 0.02)
    ax.set_xlim(-0.25, 3.25)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    ax.legend(title="Participants", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False, fontsize=5)

    # ── Panel c: Pearson r bar chart ──────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#ffffff")
    y_pos = np.arange(len(PAIR_DEFS))

    for i, (row, color) in enumerate(zip(res_df.itertuples(), PAIR_COLORS)):
        yi = y_pos[i]
        ax.barh(yi, row.r, color=color, alpha=0.75, height=0.55, zorder=3)
        for x_cap in [row.lo, row.hi]:
            ax.plot([x_cap, x_cap], [yi - 0.18, yi + 0.18],
                    color=NAVY, linewidth=1.2, zorder=4)
        ax.plot([row.lo, row.hi], [yi, yi], color=NAVY, linewidth=1.2, zorder=4)

        x_label = row.hi + 0.03 if row.r >= 0 else row.lo - 0.03
        ha      = "left" if row.r >= 0 else "right"
        ax.text(x_label, yi, rf"$r$={row.r:.2f}",
                va="center", ha=ha, fontsize=5.5, color=NAVY)

        if row.p < 0.001:
            sig = "***"
        elif row.p < 0.01:
            sig = "**"
        elif row.p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        ax.text(1.12, yi, sig, va="center", ha="center", fontsize=6, color=NAVY,
                fontstyle="italic" if sig == "ns" else "normal")

    ax.axvline(0, color="#AAAAAA", linewidth=0.6, zorder=1)
    ax.axvline(1, color="#AAAAAA", linewidth=0.5, linestyle=":", zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df["label"], fontsize=6.5)
    ax.set_xlabel(r"Pearson $r$ (95% CI)", fontsize=7, labelpad=4)
    ax.set_xlim(-0.5, 1.25)
    ax.set_ylim(-0.5, len(PAIR_DEFS) - 0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Panel labels
    for ax, label in zip(axes, ["a", "b", "c"]):
        ax.text(-0.15, 1.06, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=NAVY, va="top", clip_on=False)

    # Print correlation summary
    print(f"\nPearson correlations (n={n_teams}):")
    for row in res_df.itertuples():
        p_str = "p < 0.001" if row.p < 0.001 else f"p = {row.p:.3f}"
        print(f"  {row.label:25s}: r = {row.r:+.3f} [{row.lo:+.3f}, {row.hi:+.3f}], {p_str}")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate rank/score stability figure.")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    print("Loading Track 1 results…")
    t1 = load_track1_results()
    plot_rank_score_stability(
        team_scores=t1["scores"]["participants"],
        save_path=os.path.join(args.outdir, "rank_score_stability.pdf"),
    )
