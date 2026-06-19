"""
One-time data collection script: copies the minimal set of files needed to
reproduce all paper figures into Data/ within this repository.

What it does
------------
1. Copies metrics.json (aggregates only), inference_stats.json (peak_mem only) for
   every participant (T1 + T2).
2. Pre-aggregates per-sample time.csv files into a single duration.json per participant.
3. Copies ablation metrics.json files (aggregates only).
4. Copies metadata files (Methods Excel, Filenames CSV) — strips Methods Excel to only
   the columns used by the figure scripts (see T1_METHODS_KEEP / T2_METHODS_KEEP below).
5. Anonymises the peer-review Excel (keeps: paper title, track, per-criterion scores,
   sum score, decision — removes all reviewer-identifying information).
6. Pre-computes summary statistics that require the test-set annotation files
   (which are NOT copied to the repo):
     - t1_roi_counts.csv      : per-ROI mitotic-figure counts per slide/tumor
     - t1_kappa.csv           : Cohen's κ (R1 vs R2) per tumor type
     - t2_class_distribution.csv : AMF/NMF proportion per tumor type
     - t2_kappa.csv           : Cohen's κ per tumor type (3 comparisons)

Run once from the repository root (needs access to the full paper workspace):
    python Scripts/paper_figures/collect_data.py

The source paths are taken from utils.py environment variables or defaults.
"""

import os, json, shutil
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ── Resolve source paths ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_HERE, "..", "..")

# Source paths (same defaults as utils.py)
_PAPER_ROOT = os.path.join(_REPO_ROOT, "..", "MIDOG 2025 paper")
SRC_T1    = os.environ.get("MIDOG25_T1_OUTPUT",    os.path.join(_PAPER_ROOT, "track1", "outputs"))
SRC_T2    = os.environ.get("MIDOG25_T2_OUTPUT",    os.path.join(_PAPER_ROOT, "track2", "outputs"))
SRC_DS    = os.environ.get("MIDOG25_DATASETS",     os.path.join(_PAPER_ROOT, "datasets"))
SRC_EVAL  = os.environ.get("MIDOG25_EVALUATIONS",  os.path.join(_PAPER_ROOT, "evaluations"))

# Destination
DATA_ROOT = os.path.join(_REPO_ROOT, "Data")

T1_PARTICIPANTS = [
    "wildsquirrel", "RaphaelBourgade", "ytopuz53", "cacfek", "navyasri.kelam",
    "christian.marzahl", "tengyoux", "masarno", "piotrgiedziun", "ammeling",
    "SZTU-134", "schoe", "vidushiwalia", "npic-ab",
]
T2_PARTICIPANTS = [
    "guillaume.balezo", "nasires", "yohsuke.yamagishi", "masarno", "zerostarcraft",
    "krausara", "lrc9859", "be_yuan", "piotrgiedziun", "cacfek", "schoe",
    "saipradeepvg", "navyasri.kelam", "Leire", "Kaustubh_Atey", "mirazzak",
    "mlafarge", "baseline", "chillice",
]

TUMOR_TYPES = [
    "hMel", "hAC", "hBlC", "cMC", "ccMCT",
    "hMen", "hCoC", "cHAS", "fSTS", "fLym", "hGBM", "hLUAD",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _mkdir(path):
    os.makedirs(path, exist_ok=True)


def _copy(src, dst):
    if os.path.exists(src):
        _mkdir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    print(f"  MISSING: {src}")
    return False


def _copy_metrics(src, dst):
    """Copy metrics.json keeping only the 'aggregates' key (strips per-image predictions)."""
    if not os.path.exists(src):
        print(f"  MISSING: {src}")
        return False
    with open(src) as f:
        data = json.load(f)
    stripped = {"aggregates": data["aggregates"]} if "aggregates" in data else data
    _mkdir(os.path.dirname(dst))
    with open(dst, "w") as f:
        json.dump(stripped, f)
    return True


def _copy_inference_stats(src, dst):
    """Copy inference_stats.json keeping only peak_mem per sample (strips duration)."""
    if not os.path.exists(src):
        print(f"  MISSING: {src}")
        return False
    with open(src) as f:
        data = json.load(f)
    stripped = {k: {"peak_mem": v["peak_mem"]} for k, v in data.items()}
    _mkdir(os.path.dirname(dst))
    with open(dst, "w") as f:
        json.dump(stripped, f)
    return True


# ── 1 & 2: inference outputs ──────────────────────────────────────────────────

def collect_inference_outputs(participants, src_base, dst_base):
    for p in participants:
        src_p = os.path.join(src_base, p)
        dst_p = os.path.join(dst_base, p)
        _mkdir(dst_p)

        _copy_metrics(os.path.join(src_p, "metrics.json"), os.path.join(dst_p, "metrics.json"))
        _copy_inference_stats(os.path.join(src_p, "inference_stats.json"), os.path.join(dst_p, "inference_stats.json"))

        # Pre-aggregate time.csv → duration.json
        times = []
        for k in range(1, 366):
            t_path = os.path.join(src_p, str(k), "time.csv")
            if os.path.exists(t_path):
                with open(t_path) as f:
                    try:
                        times.append(float(f.readline().strip()))
                    except ValueError:
                        pass
        if times:
            with open(os.path.join(dst_p, "duration.json"), "w") as f:
                json.dump({"mean_s": float(np.mean(times)),
                           "std_s":  float(np.std(times)),
                           "n":      len(times)}, f, indent=2)

        # Ablation metrics
        abl_src = os.path.join(src_p, "ablations")
        if os.path.isdir(abl_src):
            for abl in os.listdir(abl_src):
                m_src = os.path.join(abl_src, abl, "metrics.json")
                m_dst = os.path.join(dst_p, "ablations", abl, "metrics.json")
                _copy_metrics(m_src, m_dst)

        print(f"  {p}: done ({len(times)} timing samples)")


# ── 3: metadata files ─────────────────────────────────────────────────────────

# Only the columns actually read by tables.py are kept.
_T1_METHODS_KEEP = [
    "Username", "Authors", "Approach", "Approach.1", "Base architecture",
    "Single model size (parameters)",
    "MIDOG++ in training", "CMC in training", "CCMCT in training", "Additional datasets",
    "TTA used", "Ensembling",
]
_T2_METHODS_KEEP = [
    "Username", "Authors", "Approach", "Base architecture",
    "Single model size (parameters)",
    "MIDOG25 in training", "Ami-Br in training", "OMG-Octo in training", "Additional datasets",
    "TTA used", "Ensembling",
]


def collect_metadata(src_eval, dst_meta):
    _mkdir(dst_meta)

    for fname, keep in [
        ("MIDOG 2025 - Track 1 - Methods.xlsx", _T1_METHODS_KEEP),
        ("MIDOG 2025 - Track 2 - Methods-3.xlsx", _T2_METHODS_KEEP),
    ]:
        src = os.path.join(src_eval, fname)
        dst = os.path.join(dst_meta, fname)
        if not os.path.exists(src):
            print(f"  MISSING: {src}")
            continue
        df = pd.read_excel(src)
        present = [c for c in keep if c in df.columns]
        df = df[present]
        df.to_excel(dst, index=False)
        print(f"  Saved ({len(present)} cols): {dst}")

    # Filenames CSV (slide metadata only, no annotations)
    for src_dir in [SRC_DS, src_eval]:
        fpath = os.path.join(src_dir, "Filenames_finaltest_MIDOG25.csv")
        if os.path.exists(fpath):
            _copy(fpath, os.path.join(dst_meta, "Filenames_finaltest_MIDOG25.csv"))
            break


# ── 5: anonymise peer reviews ─────────────────────────────────────────────────

_KEEP_REVIEW_COLS = [
    "Title of the paper I'm reviewing",
    "Challenge track",
    "Technical Clarity: The approach and the training are clearly described.",
    "Innovation: The approach is novel or adds an innovative twist to an existing method.",
    "Formal: The paper is written well, clearly structured and easy to understand.",
    "Were external (i.e., not provided by us) datasets used?",
    "Were the external datasets available publicly before July 19th?",
    "Recommendation",
    "Sum Score",
    "Average Score",
    "Decision",
]

def collect_reviews(src_eval, dst_meta):
    src = os.path.join(src_eval, "MIDOG 2025 Reviews (Responses)-2.xlsx")
    dst = os.path.join(dst_meta, "reviews_anonymized.csv")
    if not os.path.exists(src):
        print(f"  MISSING: {src}")
        return
    df = pd.read_excel(src)
    present = [c for c in _KEEP_REVIEW_COLS if c in df.columns]
    df[present].to_csv(dst, index=False)
    print(f"  Saved (anonymised, {len(df)} reviews → {dst})")


# ── 6: pre-compute annotation summaries ──────────────────────────────────────

def precompute_t1_summaries(src_eval, src_ds, dst_pre):
    """
    Uses t1_dataset_with_tumor_types.csv + Filenames CSV to produce:
      - t1_roi_counts.csv  (per-ROI mitotic figure counts)
      - t1_kappa.csv       (Cohen's κ per tumor type)
    """
    _mkdir(dst_pre)

    t1_csv = os.path.join(src_eval, "t1_dataset_with_tumor_types.csv")
    fn_csv = None
    for d in [src_ds, src_eval]:
        p = os.path.join(d, "Filenames_finaltest_MIDOG25.csv")
        if os.path.exists(p):
            fn_csv = p
            break

    if not os.path.exists(t1_csv):
        print(f"  MISSING (skip T1 summaries): {t1_csv}")
        return
    if fn_csv is None:
        print("  MISSING: Filenames_finaltest_MIDOG25.csv (skip T1 summaries)")
        return

    df = pd.read_csv(t1_csv)
    fn = pd.read_csv(fn_csv, delimiter=";")
    roi_map = dict(zip(fn["Slide"], fn["roi_type"]))
    df["roi_type"] = df["slide"].map(roi_map)

    # Per-ROI counts
    counts = (
        df.groupby(["slide", "Tumor", "roi_type"])
        .apply(lambda g: pd.Series({
            "mf_count":    (g["class"] == "MF").sum(),
            "total_count": len(g),
        }), include_groups=False)
        .reset_index()
    )
    counts.to_csv(os.path.join(dst_pre, "t1_roi_counts.csv"), index=False)
    print(f"  t1_roi_counts.csv: {len(counts)} rows")

    # Kappa per tumor type
    kappa_rows = []
    for tumor, group in df.groupby("Tumor"):
        pair = group[["CB", "RK"]].dropna()
        if len(pair) < 2:
            continue
        kappa = cohen_kappa_score(pair["CB"], pair["RK"])
        kappa_rows.append({"Tumor_Type": tumor, "Comparison": "R1 vs R2",
                           "n": len(pair), "Cohen_kappa": round(kappa, 4)})
    pd.DataFrame(kappa_rows).to_csv(os.path.join(dst_pre, "t1_kappa.csv"), index=False)
    print(f"  t1_kappa.csv: {len(kappa_rows)} tumor types")


def precompute_t2_summaries(src_eval, dst_pre):
    """
    Uses t2_dataset_with_tumor_types.csv to produce:
      - t2_class_distribution.csv
      - t2_kappa.csv
    """
    _mkdir(dst_pre)

    t2_csv = os.path.join(src_eval, "t2_dataset_with_tumor_types.csv")
    if not os.path.exists(t2_csv):
        print(f"  MISSING (skip T2 summaries): {t2_csv}")
        return

    df = pd.read_csv(t2_csv)

    # Class distribution per tumor type
    counts = (
        df.groupby(["Tumor_Type", "final_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["AMF", "NMF"], fill_value=0)
        .reset_index()
    )
    counts.to_csv(os.path.join(dst_pre, "t2_class_distribution.csv"), index=False)
    print(f"  t2_class_distribution.csv: {len(counts)} tumor types")

    # Kappa per tumor type (3 comparisons)
    kappa_rows = []
    for tumor, group in df.groupby("Tumor_Type"):
        for col_a, col_b, label in [
            ("VW", "CB",          "R1 vs R2"),
            ("VW", "final_label", "R1 vs final"),
            ("CB", "final_label", "R2 vs final"),
        ]:
            pair = group[[col_a, col_b]].dropna()
            if len(pair) < 2:
                continue
            kappa = cohen_kappa_score(pair[col_a], pair[col_b])
            kappa_rows.append({"Tumor_Type": tumor, "Comparison": label,
                               "n": len(pair), "Cohen_kappa": round(kappa, 4)})
    pd.DataFrame(kappa_rows).to_csv(os.path.join(dst_pre, "t2_kappa.csv"), index=False)
    print(f"  t2_kappa.csv: {len(kappa_rows)} rows")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Data root: {DATA_ROOT}\n")

    print("=== Track 1 inference outputs ===")
    collect_inference_outputs(T1_PARTICIPANTS, SRC_T1, os.path.join(DATA_ROOT, "track1"))

    print("\n=== Track 2 inference outputs ===")
    collect_inference_outputs(T2_PARTICIPANTS, SRC_T2, os.path.join(DATA_ROOT, "track2"))

    print("\n=== Metadata ===")
    collect_metadata(SRC_EVAL, os.path.join(DATA_ROOT, "metadata"))

    print("\n=== Peer reviews (anonymise) ===")
    collect_reviews(SRC_EVAL, os.path.join(DATA_ROOT, "metadata"))

    print("\n=== Pre-computed T1 summaries ===")
    precompute_t1_summaries(SRC_EVAL, SRC_DS, os.path.join(DATA_ROOT, "precomputed"))

    print("\n=== Pre-computed T2 summaries ===")
    precompute_t2_summaries(SRC_EVAL, os.path.join(DATA_ROOT, "precomputed"))

    print("\nDone. Review Data/ before committing.")
