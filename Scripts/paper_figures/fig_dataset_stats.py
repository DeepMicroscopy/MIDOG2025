"""
Dataset statistics figures.

Produces:
  fig_mitotic_counts_scaled.pdf  — violin + strip plots of MF counts per tumor / ROI type (Track 1)
  fig_t2_class_distribution.pdf  — stacked bar chart of AMF/NMF proportions per tumor type (Track 2)
  tab_dataset_stats.csv          — summary statistics table for Track 1

Paper figures: fig:testset-stats-t1, fig:distrib-t2
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import cohen_kappa_score

from utils import (
    DATASET_PATH, EVALUATIONS_PATH, PRECOMPUTED_PATH,
    TUMOR_TYPES, ROI_TYPES, ROI_COLORS,
    NAVY, MM_TO_INCH, style_rcparams,
)


# ── Track 1 dataset ────────────────────────────────────────────────────────────

def build_track1_dataset(dataset_path=None, evaluations_path=None):
    """
    Merge the MIDOG 2022 final-test annotations with the MIDOG 2025 extension
    into one dataframe with columns: uid, slide, CB, RK, class, Tumor, roi_type.
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH
    if evaluations_path is None:
        evaluations_path = EVALUATIONS_PATH

    filenames_csv = os.path.join(dataset_path, "Filenames_finaltest_MIDOG25.csv")
    filenames     = pd.read_csv(filenames_csv, delimiter=";")
    origslide_to_slide    = dict(zip(filenames.OrigSlide, filenames.Slide))
    slide_to_tumor        = dict(zip(filenames.Slide,     filenames.Tumor))
    slide_to_roi_type     = dict(zip(filenames.Slide,     filenames.roi_type))

    # MIDOG 2022 portion (from sqlite)
    db_path = os.path.join(dataset_path, "MIDOG2022_finaltest.sqlite")
    db      = sqlite3.connect(db_path).cursor()
    classes = {1: "MF", 2: "NMF"}

    expert1 = {
        uid: (slide, cls)
        for slide, uid, cls in db.execute(
            "SELECT slide, Annotations.uid, class "
            "FROM Annotations_label "
            "LEFT JOIN Annotations ON Annotations_label.annoId = Annotations.uid "
            "WHERE Annotations_label.person = 1"
        ).fetchall()
    }
    expert2 = {
        uid: (slide, cls)
        for slide, uid, cls in db.execute(
            "SELECT slide, Annotations.uid, class "
            "FROM Annotations_label "
            "LEFT JOIN Annotations ON Annotations_label.annoId = Annotations.uid "
            "WHERE Annotations_label.person = 3"
        ).fetchall()
    }
    agreed = {
        uid: (slide, cls)
        for slide, uid, cls in db.execute(
            "SELECT slide, Annotations.uid, agreedClass "
            "FROM Annotations_label "
            "LEFT JOIN Annotations ON Annotations_label.annoId = Annotations.uid "
            "WHERE Annotations_label.person = 1"
        ).fetchall()
    }

    slide_fmt = lambda uid: "%03d.tiff" % expert1[uid][0]
    midog2022 = pd.DataFrame({
        "uid":   list(expert1.keys()),
        "slide": [slide_fmt(u)                     for u in expert1],
        "CB":    [classes[expert1[u][1]]            for u in expert1],
        "RK":    [classes[expert2[u][1]]            for u in expert1],
        "class": [classes[agreed[u][1]]             for u in expert1],
        "Tumor": [slide_to_tumor[slide_fmt(u)]      for u in expert1],
    })

    # MIDOG 2025 extension portion
    ext_csv = os.path.join(dataset_path, "T1_midog2025_consensus_raw.csv")
    ext     = pd.read_csv(ext_csv)
    valid   = ~ext["1"].isna()
    ext_df  = pd.DataFrame({
        "uid":   ext.id[valid].values,
        "slide": [origslide_to_slide[x] for x in ext.filename[valid]],
        "CB":    [classes[x] for x in ext["0"][valid]],
        "RK":    [classes[x] for x in ext["1"][valid]],
        "class": [classes[x] for x in ext["consensus"][valid]],
        "Tumor": [slide_to_tumor[origslide_to_slide[x]] for x in ext.filename[valid]],
    })

    all_decisions = pd.concat([midog2022, ext_df], ignore_index=True)

    # Attach roi_type
    roi_types = [slide_to_roi_type.get(s, "unknown") for s in all_decisions.slide]
    all_decisions["roi_type"] = roi_types
    return all_decisions


def _compute_per_roi_counts(df):
    return (
        df.groupby(["slide", "Tumor", "roi_type"])
        .apply(lambda g: pd.Series({
            "mf_count":    (g["class"] == "MF").sum(),
            "total_count": len(g),
        }), include_groups=False)
        .reset_index()
    )


def load_t1_roi_counts(precomputed_path=None):
    """
    Load pre-computed per-ROI mitotic-figure counts.
    Falls back to building from raw data if the precomputed file is absent.
    """
    if precomputed_path is None:
        precomputed_path = PRECOMPUTED_PATH
    csv_path = os.path.join(precomputed_path, "t1_roi_counts.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Fallback: build from raw annotations (requires MIDOG22 SQLite + T1 CSV)
    df = build_track1_dataset()
    return _compute_per_roi_counts(df)


def load_t2_class_distribution(precomputed_path=None):
    """
    Load pre-computed per-tumor AMF/NMF counts.
    Falls back to building from raw data if the precomputed file is absent.
    """
    if precomputed_path is None:
        precomputed_path = PRECOMPUTED_PATH
    csv_path = os.path.join(precomputed_path, "t2_class_distribution.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    df = build_track2_dataset()
    counts = (
        df.groupby(["Tumor_Type", "final_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["AMF", "NMF"], fill_value=0)
        .reset_index()
    )
    return counts


def plot_mitotic_count_distributions(df_or_counts, save_path="fig_mitotic_counts_scaled.pdf"):
    """
    Violin + strip plots of per-ROI mitotic figure counts, one panel per ROI type,
    tumor types sorted by descending median hotspot count.
    Paper figure: fig:testset-stats-t1
    """
    style_rcparams()
    # Accept either a raw annotation DataFrame or a pre-computed counts DataFrame
    if "mf_count" in df_or_counts.columns:
        counts = df_or_counts
    else:
        counts = _compute_per_roi_counts(df_or_counts)

    hotspot_medians = (
        counts[counts["roi_type"] == "hotspot"]
        .groupby("Tumor")["mf_count"]
        .median()
    )
    sorted_tumors = (
        hotspot_medians.reindex(TUMOR_TYPES)
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(180 * MM_TO_INCH, 65 * MM_TO_INCH),
        dpi=300,
        gridspec_kw={"wspace": 0.35},
    )
    fig.patch.set_facecolor("white")

    for ax, roi in zip(axes, ROI_TYPES):
        ax.set_facecolor("#FFFFFF")
        color    = ROI_COLORS[roi]
        roi_data = counts[counts["roi_type"] == roi]

        data_per_tumor, positions = [], []
        for j, tumor in enumerate(sorted_tumors):
            vals = roi_data[roi_data["Tumor"] == tumor]["mf_count"].values
            if len(vals):
                data_per_tumor.append(vals)
                positions.append(j)

        if not data_per_tumor:
            continue

        parts = ax.violinplot(
            data_per_tumor, positions=positions,
            widths=0.7, showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.5)

        rng = np.random.default_rng(42)
        for j, vals in zip(positions, data_per_tumor):
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(j + jitter, vals, color=color, s=8, alpha=0.55,
                       edgecolors="none", zorder=3)

        ax.set_xticks(range(len(sorted_tumors)))
        ax.set_xticklabels(sorted_tumors, rotation=45, ha="right",
                           fontsize=6, fontstyle="italic")
        ax.set_title(f"{roi} ROIs", fontsize=8, fontweight="bold", color=color, pad=5)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.set_xlim(-0.6, len(sorted_tumors) - 0.4)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Mitotic figures per ROI", fontsize=7, labelpad=4)

    for ax, label in zip(axes, ["a)", "b)", "c)"]):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=NAVY, va="top", clip_on=False)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


def export_dataset_stats_table(df, save_path="tab_dataset_stats.csv"):
    """Export per-tumor × ROI-type summary statistics for Table 1."""
    counts = _compute_per_roi_counts(df)
    rows = []
    for tumor in TUMOR_TYPES:
        row = {"Domain": tumor}
        for roi in ROI_TYPES:
            sub = counts[(counts["Tumor"] == tumor) & (counts["roi_type"] == roi)]
            row[f"{roi}_n_rois"]    = len(sub)
            row[f"{roi}_mf_total"]  = int(sub["mf_count"].sum())
            row[f"{roi}_mf_median"] = round(sub["mf_count"].median(), 1)
            row[f"{roi}_mf_iqr"]    = (
                f"{int(sub['mf_count'].quantile(0.25))}"
                f"–{int(sub['mf_count'].quantile(0.75))}"
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(save_path, index=False)
    print(f"Saved → {save_path}")
    return table


# ── Track 2 dataset ────────────────────────────────────────────────────────────

def build_track2_dataset(dataset_path=None):
    """
    Load the Track 2 test-set annotations and attach tumor type labels.
    Returns a DataFrame with columns including Tumor_Type, final_label, VW, CB.
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH

    df        = pd.read_csv(os.path.join(dataset_path, "MIDOG25_T2_Test_Set_Final.csv"))
    df_images = pd.read_csv(os.path.join(dataset_path, "MIDOG25_Test_Images_To_Annotate.csv"))
    filenames = pd.read_csv(os.path.join(dataset_path, "Filenames_finaltest_MIDOG25.csv"),
                            delimiter=";")

    ttype_LUT = dict(zip(filenames.Slide, filenames.Tumor))
    df = df.join(df_images)
    df["Tumor_Type"] = [ttype_LUT[x] for x in df.original_filename]
    return df


def plot_t2_class_distribution(df_or_counts, save_path="fig_t2_class_distribution.pdf"):
    """
    Horizontal stacked bar chart: AMF vs. NMF proportions per tumor type.
    Accepts either a raw annotator DataFrame or a pre-aggregated counts DataFrame
    (with columns Tumor_Type, AMF, NMF).
    Paper figure: fig:distrib-t2
    """
    style_rcparams()

    if "AMF" in df_or_counts.columns and "Tumor_Type" in df_or_counts.columns:
        # Pre-computed counts DataFrame
        counts = df_or_counts.set_index("Tumor_Type")[["AMF", "NMF"]].copy()
    else:
        counts = (
            df_or_counts.groupby(["Tumor_Type", "final_label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["AMF", "NMF"], fill_value=0)
        )
    counts["total"]    = counts["AMF"] + counts["NMF"]
    counts["amf_prop"] = counts["AMF"] / counts["total"]
    counts = counts.sort_values("amf_prop", ascending=True)
    tumors = counts.index.tolist()

    COLOR_AMF = "#D55E00"
    COLOR_NMF = "#0072B2"
    n = len(tumors)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(90 * MM_TO_INCH, 80 * MM_TO_INCH), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    ax.barh(y, counts["amf_prop"] * 100, color=COLOR_AMF, alpha=0.80,
            height=0.62, label="Atypical MF")
    ax.barh(y, (1 - counts["amf_prop"]) * 100, left=counts["amf_prop"] * 100,
            color=COLOR_NMF, alpha=0.80, height=0.62, label="Normal MF")

    for i, prop in enumerate(counts["amf_prop"]):
        pct = prop * 100
        if pct > 12:
            ax.text(pct / 2, i - 0.05, f"{pct:.1f}%",
                    va="center", ha="center", fontsize=7, color="white", fontweight="500")
        else:
            ax.text(pct + 1, i - 0.05, f"{pct:.1f}%",
                    va="center", ha="left", fontsize=7, color="white")

    for i, total in enumerate(counts["total"]):
        ax.text(102, i - 0.05, f"n={total}", va="center", ha="left", fontsize=7, color=NAVY)

    ax.set_yticks(y)
    ax.set_yticklabels(tumors, fontsize=6.5, fontstyle="italic")
    ax.set_xlabel("Proportion (%)", fontsize=7, labelpad=4)
    ax.set_xlim(0, 120)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(fontsize=7, framealpha=0.9, edgecolor="#CCCCCC",
              handlelength=1.0, handleheight=0.8, bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {save_path}")


# ── Annotator agreement (Track 2) ─────────────────────────────────────────────

def compute_t2_kappa_table(df):
    """
    Cohen's kappa between R1, R2, and the consensus label, per tumor type.
    Returns a pivot table (rows = tumor types, columns = R1/R2, R1/final, R2/final).
    """
    results = []
    for tumor_type, group in df.groupby("Tumor_Type"):
        for col_a, col_b, label in [("VW", "CB", "R1 vs R2"),
                                     ("VW", "final_label", "R1 vs final"),
                                     ("CB", "final_label", "R2 vs final")]:
            pair = group[[col_a, col_b]].dropna()
            kappa = cohen_kappa_score(pair[col_a], pair[col_b])
            results.append({"Tumor_Type": tumor_type, "Comparison": label,
                             "n": len(pair), "Cohen_kappa": round(kappa, 4)})

    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index="Tumor_Type", columns="Comparison",
                              values="Cohen_kappa")
    return pivot[["R1 vs R2", "R1 vs final", "R2 vs final"]], results_df


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dataset statistics figures.")
    parser.add_argument("--outdir", default=".", help="Output directory for figures")
    args = parser.parse_args()

    out = args.outdir

    print("Loading T1 ROI counts…")
    t1_counts = load_t1_roi_counts()
    plot_mitotic_count_distributions(t1_counts,
        save_path=os.path.join(out, "fig_mitotic_counts_scaled.pdf"))

    print("Loading T2 class distribution…")
    t2_dist = load_t2_class_distribution()
    plot_t2_class_distribution(t2_dist,
        save_path=os.path.join(out, "fig_t2_class_distribution.pdf"))
