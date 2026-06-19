"""Shared constants, style helpers, and data-loading utilities for MIDOG 2025 paper figures."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ── Data paths ─────────────────────────────────────────────────────────────────
# Defaults point to the sibling repository layout. Override via environment
# variables or by editing these strings directly.
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_HERE, "..", "..")
_DATA_ROOT = os.path.join(_REPO_ROOT, "Data")

TRACK1_OUTPUT_PATH = os.environ.get(
    "MIDOG25_T1_OUTPUT",
    os.path.join(_DATA_ROOT, "track1"),
)
TRACK2_OUTPUT_PATH = os.environ.get(
    "MIDOG25_T2_OUTPUT",
    os.path.join(_DATA_ROOT, "track2"),
)
DATASET_PATH = os.environ.get(
    "MIDOG25_DATASETS",
    os.path.join(_DATA_ROOT, "metadata"),
)
EVALUATIONS_PATH = os.environ.get(
    "MIDOG25_EVALUATIONS",
    os.path.join(_DATA_ROOT, "metadata"),
)
PRECOMPUTED_PATH = os.environ.get(
    "MIDOG25_PRECOMPUTED",
    os.path.join(_DATA_ROOT, "precomputed"),
)

# ── Participant lists ──────────────────────────────────────────────────────────
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

# ── Domain labels ──────────────────────────────────────────────────────────────
TUMOR_TYPES = [
    "hMel", "hAC", "hBlC", "cMC", "ccMCT",
    "hMen", "hCoC", "cHAS", "fSTS", "fLym", "hGBM", "hLUAD",
]
ROI_TYPES = ["hotspot", "random", "challenging"]

# ── Colors ─────────────────────────────────────────────────────────────────────
# Okabe-Ito accessible palette
ROI_COLORS = {
    "hotspot":     "#009E73",
    "random":      "#0072B2",
    "challenging": "#D55E00",
    "overall":     "#7B2D8B",
}
CB_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]

NAVY       = "#1B2A4A"
FONT_SERIF = ["Palatino Linotype", "Palatino", "Georgia", "DejaVu Sans"]
MM_TO_INCH = 1 / 25.4


# ── Style ──────────────────────────────────────────────────────────────────────

def style_rcparams():
    """Apply consistent publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family":         "serif",
        "font.serif":          FONT_SERIF,
        "mathtext.fontset":    "cm",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      0.6,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.size":    3,
        "ytick.major.size":    3,
        "xtick.minor.size":    1.5,
        "ytick.minor.size":    1.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.labelsize":     6,
        "ytick.labelsize":     6,
    })


# ── Track 1 data loading ───────────────────────────────────────────────────────

def load_track1_results(participants=None, basepath=None):
    """
    Load Track 1 inference metrics for all participants.

    Returns a dict with keys:
        f1_score, peak_mem, duration, precisions, recalls,
        scores, tumor_type_scores, roitype_tumortype, ablations
    """
    if participants is None:
        participants = T1_PARTICIPANTS
    if basepath is None:
        basepath = TRACK1_OUTPUT_PATH

    peak_mem = {}
    duration = {}
    f1_score = {}
    precisions = {}
    recalls = {}
    scores = {"hotspot": {}, "random": {}, "challenging": {}, "participants": {}}
    ablations = {}
    roitype_tumortype = {}
    tumor_type_scores = {
        "F1":        {k: [] for k in range(12)},
        "froc_auc":  {k: [] for k in range(12)},
        "recall":    {k: [] for k in range(12)},
        "precision": {k: [] for k in range(12)},
    }

    for p in participants:
        stats_path = os.path.join(basepath, p, "inference_stats.json")
        if not os.path.exists(stats_path):
            print(f"WARNING: No inference_stats.json for {p}")
            continue

        j = json.load(open(stats_path))
        peak_mem[p] = np.median([j[x]["peak_mem"] for x in j])

        dur_path = os.path.join(basepath, p, "duration.json")
        if os.path.exists(dur_path):
            duration[p] = json.load(open(dur_path))["mean_s"]
        else:
            times = []
            for k in range(365):
                t_path = os.path.join(basepath, p, str(k + 1), "time.csv")
                if os.path.exists(t_path):
                    with open(t_path) as f:
                        times.append(float(f.readlines()[0]))
            duration[p] = np.mean(times) if times else np.nan

        j   = json.load(open(os.path.join(basepath, p, "metrics.json")))
        agg = j["aggregates"]

        f1_score[p]               = agg["f1_score"]
        scores["hotspot"][p]      = agg["roi_type_hotspot_f1"]
        scores["random"][p]       = agg["roi_type_random_f1"]
        scores["challenging"][p]  = agg["roi_type_challenging_f1"]
        scores["participants"][p] = {
            "hotspot":     agg["roi_type_hotspot_f1"],
            "random":      agg["roi_type_random_f1"],
            "challenging": agg["roi_type_challenging_f1"],
            "overall":     agg["f1_score"],
        }
        precisions[p] = {
            "hotspot":     agg["roi_type_hotspot_precision"],
            "random":      agg["roi_type_random_precision"],
            "challenging": agg["roi_type_challenging_precision"],
            "overall":     agg["precision"],
        }
        recalls[p] = {
            "hotspot":     agg["roi_type_hotspot_recall"],
            "random":      agg["roi_type_random_recall"],
            "challenging": agg["roi_type_challenging_recall"],
            "overall":     agg["recall"],
        }

        for k in range(12):
            tumor_type_scores["F1"][k]       .append(agg[f"tumor_{k}_f1"])
            tumor_type_scores["recall"][k]   .append(agg[f"tumor_{k}_recall"])
            tumor_type_scores["precision"][k].append(agg[f"tumor_{k}_precision"])
            tumor_type_scores["froc_auc"][k] .append(agg[f"tumor_{k}_froc_auc"])

        for key, val in agg["tumor_roitype"].items():
            roitype_tumortype.setdefault(key, []).append(val)

        ablation_dir = os.path.join(basepath, p, "ablations")
        ablations[p] = os.listdir(ablation_dir) if os.path.isdir(ablation_dir) else []

    return {
        "f1_score":          f1_score,
        "peak_mem":          peak_mem,
        "duration":          duration,
        "precisions":        precisions,
        "recalls":           recalls,
        "scores":            scores,
        "tumor_type_scores": tumor_type_scores,
        "roitype_tumortype": roitype_tumortype,
        "ablations":         ablations,
    }


# ── Track 2 data loading ───────────────────────────────────────────────────────

def load_track2_results(participants=None, basepath=None):
    """
    Load Track 2 inference metrics for all participants.

    Returns a dict with keys:
        balanced_accuracy, peak_mem, duration, tumor_type_scores, ablations
    """
    if participants is None:
        participants = T2_PARTICIPANTS
    if basepath is None:
        basepath = TRACK2_OUTPUT_PATH

    peak_mem = {}
    duration = {}
    balanced_accuracy = {}
    ablations_available = {}
    tumor_type_scores = {
        "BA":          {k: [] for k in range(12)},
        "AUROC":       {k: [] for k in range(12)},
        "sensitivity": {k: [] for k in range(12)},
        "specificity": {k: [] for k in range(12)},
    }

    for p in participants:
        stats_path = os.path.join(basepath, p, "inference_stats.json")
        if not os.path.exists(stats_path):
            print(f"WARNING: No inference_stats.json for {p}")
            continue

        j = json.load(open(stats_path))
        peak_mem[p] = np.median([j[x]["peak_mem"] for x in j])

        dur_path = os.path.join(basepath, p, "duration.json")
        if os.path.exists(dur_path):
            duration[p] = json.load(open(dur_path))["mean_s"]
        else:
            times = []
            for k in range(365):
                t_path = os.path.join(basepath, p, str(k + 1), "time.csv")
                if os.path.exists(t_path):
                    with open(t_path) as f:
                        times.append(float(f.readlines()[0]))
            duration[p] = np.mean(times) if times else np.nan

        j   = json.load(open(os.path.join(basepath, p, "metrics.json")))
        agg = j["aggregates"]
        balanced_accuracy[p] = agg["overall_balanced_accuracy"]

        for k in range(12):
            tumor_type_scores["BA"][k]         .append(agg[f"domain_{k}"]["balanced_accuracy"])
            tumor_type_scores["AUROC"][k]      .append(agg[f"domain_{k}"]["roc_auc"])
            tumor_type_scores["sensitivity"][k].append(agg[f"domain_{k}"]["sensitivity"])
            tumor_type_scores["specificity"][k].append(agg[f"domain_{k}"]["specificity"])

        ablation_dir = os.path.join(basepath, p, "ablations")
        ablations_available[p] = os.listdir(ablation_dir) if os.path.isdir(ablation_dir) else []

    return {
        "balanced_accuracy": balanced_accuracy,
        "peak_mem":          peak_mem,
        "duration":          duration,
        "tumor_type_scores": tumor_type_scores,
        "ablations":         ablations_available,
    }


# ── Ablation result loading ────────────────────────────────────────────────────

def load_ablation_results(skey, participants, basepath):
    """
    For a given metric key, load full and ablated results for all participants.

    Returns
    -------
    final_results, abl_tta, abl_ensembling, abl_tta_ensembling,
    abl_ensembling_all, abl_tta_ensembling_all
    """
    final_results        = {}
    abl_tta              = {}
    abl_ensembling       = {}
    abl_tta_ensembling   = {}
    abl_ensembling_all   = {}
    abl_tta_ensembling_all = {}

    for p in participants:
        metrics_path = os.path.join(basepath, p, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        j = json.load(open(metrics_path))
        final_results[p] = j["aggregates"][skey]

        ensembling     = []
        ensembling_TTA = []

        ablation_dir = os.path.join(basepath, p, "ablations")
        if not os.path.isdir(ablation_dir):
            continue

        for abl in os.listdir(ablation_dir):
            abln      = abl.replace("_src", "")
            json_path = os.path.join(ablation_dir, abl, "metrics.json")
            if not os.path.exists(json_path):
                print(f"  Missing: {json_path}")
                continue
            met = json.load(open(json_path))["aggregates"][skey]
            if abln == "TTA":
                abl_tta[p] = met
            elif "ensembling_TTA" in abln:
                ensembling_TTA.append(met)
            elif "ensembling" in abln:
                ensembling.append(met)

        if ensembling:
            abl_ensembling[p]     = np.median(ensembling)
            abl_ensembling_all[p] = ensembling
        if ensembling_TTA:
            abl_tta_ensembling[p]     = np.median(ensembling_TTA)
            abl_tta_ensembling_all[p] = ensembling_TTA

    return (
        final_results,
        abl_tta,
        abl_ensembling,
        abl_tta_ensembling,
        abl_ensembling_all,
        abl_tta_ensembling_all,
    )


# ── Per-tumor/ROI metric helpers ───────────────────────────────────────────────

def get_roitype_metric(results, tumor_idx, roi_type, metric):
    """Extract per-algorithm score list for a given tumor/ROI/metric combination."""
    return results.get(f"{tumor_idx}_{roi_type}_{metric}", [])


def get_tumortype_metric(results, tumor_idx, metric):
    """Extract per-algorithm score list for a given tumor/metric combination (T2)."""
    return results.get(metric, {}).get(tumor_idx, [])
