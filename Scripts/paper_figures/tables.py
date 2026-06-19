"""
LaTeX leaderboard tables for the MIDOG 2025 paper.

Produces:
  tab_t1_full.tex  — Track 1 sidewaystable: rank, author, approach, arch, params,
                     training data, TTA, ensembling, inference time, F1, FROC-AUC, AP,
                     review score, decision
  tab_t2_full.tex  — Track 2 sidewaystable: rank, author, approach, arch, params,
                     training data, TTA, ensembling, inference time, bAcc, ROC AUC,
                     review score, decision

Inputs (from EVALUATIONS_PATH):
  MIDOG 2025 - Track 1 - Methods.xlsx  (columns: Username, Authors, Approach, Approach.1,
                                         Base architecture, Single model size, training flags,
                                         TTA used, Ensembling)
  MIDOG 2025 - Track 2 - Methods-3.xlsx (same structure, T2 training flags)
  reviews_anonymized.csv               (paper title, scores, decision — no reviewer identity)

Paper: Table~\ref{tab:t1_full}, Table~\ref{tab:t2_full}
"""

import os
import json
import numpy as np
import pandas as pd

from utils import (
    T1_PARTICIPANTS, T2_PARTICIPANTS,
    EVALUATIONS_PATH,
    load_track1_results, load_track2_results,
)

# ── Review-score aggregation ───────────────────────────────────────────────────

def _load_reviews(evaluations_path):
    """Return the anonymized review DataFrame (title, Sum Score, Decision, …)."""
    csv_path = os.path.join(evaluations_path, "reviews_anonymized.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Fallback: raw Excel (not committed to the repo)
    return pd.read_excel(
        os.path.join(evaluations_path, "MIDOG 2025 Reviews (Responses)-2.xlsx")
    )


def _load_review_scores(evaluations_path):
    """Return a dict mapping paper title → mean review score."""
    df        = _load_reviews(evaluations_path)
    col_title = "Title of the paper I'm reviewing"
    col_score = "Sum Score"

    scores = {}
    for title, group in df.groupby(col_title):
        vals = pd.to_numeric(group[col_score], errors="coerce").dropna()
        if len(vals):
            scores[title.strip()] = round(float(vals.mean()), 1)
    return scores


def _load_review_decisions(evaluations_path):
    """Return a dict mapping paper title → majority decision string."""
    df           = _load_reviews(evaluations_path)
    col_title    = "Title of the paper I'm reviewing"
    col_decision = "Decision"
    col_rec      = "Recommendation"

    decisions = {}
    for title, group in df.groupby(col_title):
        vals = group[col_decision].dropna()
        if len(vals) == 0:
            vals = group[col_rec].dropna()
        if len(vals):
            decisions[title.strip()] = vals.mode().iloc[0]
    return decisions


# ── Training-data abbreviations ───────────────────────────────────────────────

def _abbrev_training_t1(row):
    parts = []
    if str(row.get("MIDOG++ in training", "0")) in ("1", "1.0"):
        parts.append("M++")
    if str(row.get("CMC in training", "0")) in ("1", "1.0"):
        parts.append("CMC")
    if str(row.get("CCMCT in training", "0")) in ("1", "1.0"):
        parts.append("CCMCT")
    ext = str(row.get("Additional datasets", ""))
    if ext not in ("nan", "", "None", "0"):
        parts.append("Ext.")
    return ",".join(parts) if parts else "—"


def _abbrev_training_t2(row):
    parts = []
    if str(row.get("MIDOG25 in training", "0")) in ("1", "1.0"):
        parts.append("M25")
    if str(row.get("Ami-Br in training", "0")) in ("1", "1.0"):
        parts.append("AMi-Br")
    if str(row.get("OMG-Octo in training", "0")) in ("1", "1.0"):
        parts.append("OMG")
    ext = str(row.get("Additional datasets", ""))
    if ext not in ("nan", "", "None", "0"):
        parts.append("Ext.")
    return ",".join(parts) if parts else "—"


def _fmt(val, digits=3):
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _trunc(s, n=20):
    s = str(s)
    return s[:n] + ".." if len(s) > n else s


def _bool_cell(val):
    s = str(val)
    if s in ("0", "0.0", "nan", "None", ""):
        return "No"
    if s in ("1", "1.0"):
        return "Yes"
    return "Yes" if pd.notna(val) and val else "No"


# ── Track 1 table ─────────────────────────────────────────────────────────────

def build_t1_table(evaluations_path=None, t1_results=None, save_path="tab_t1_full.tex"):
    if evaluations_path is None:
        evaluations_path = EVALUATIONS_PATH
    if t1_results is None:
        t1_results = load_track1_results()

    methods = pd.read_excel(
        os.path.join(evaluations_path, "MIDOG 2025 - Track 1 - Methods.xlsx")
    )

    review_scores    = _load_review_scores(evaluations_path)
    review_decisions = _load_review_decisions(evaluations_path)

    f1_scores   = t1_results["f1_score"]
    froc_scores = {}
    ap_scores   = {}

    for p in T1_PARTICIPANTS:
        import os as _os, json as _json
        from utils import TRACK1_OUTPUT_PATH
        mpath = _os.path.join(TRACK1_OUTPUT_PATH, p, "metrics.json")
        if _os.path.exists(mpath):
            j = _json.load(open(mpath))
            agg = j["aggregates"]
            froc_scores[p] = agg.get("froc_score", agg.get("froc_auc", np.nan))
            ap_scores[p]   = agg.get("AP", agg.get("average_precision", np.nan))

    participants_sorted = sorted(f1_scores.keys(),
                                 key=lambda p: f1_scores[p], reverse=True)
    top3 = set(participants_sorted[:3])

    durations = t1_results["duration"]

    lines = []
    for rank, uname in enumerate(participants_sorted):
        row_m = methods[methods["Username"].str.strip() == uname]
        if row_m.empty:
            row_m = methods[methods["Username"].str.strip().str.lower() == uname.lower()]
        m = row_m.iloc[0] if not row_m.empty else pd.Series(dtype=object)

        author    = _trunc(m.get("Authors", "—"), 18)
        approach  = _trunc(m.get("Approach.1", m.get("Approach", "—")), 18)
        arch      = _trunc(m.get("Base architecture", "—"), 22)
        params_raw = m.get("Single model size (parameters)", np.nan)
        try:
            params = f"{int(float(params_raw)) // 1_000_000}M"
        except (TypeError, ValueError):
            params = "—"
        train_data = _abbrev_training_t1(m) if not m.empty else "—"
        tta        = _bool_cell(m.get("TTA used", 0))
        ens        = "Yes" if str(m.get("Ensembling", "0")) not in ("0", "nan", "No", "") else "No"
        dur        = f"{durations.get(uname, np.nan):.1f}" if not np.isnan(durations.get(uname, np.nan)) else "—"

        f1_v  = f1_scores.get(uname, np.nan)
        froc_v = froc_scores.get(uname, np.nan)
        ap_v  = ap_scores.get(uname, np.nan)

        def bold(s, cond):
            return rf"\textbf{{{s}}}" if cond else s

        f1_str   = bold(_fmt(f1_v),  uname in top3)
        froc_str = bold(_fmt(froc_v), uname in top3)
        ap_str   = bold(_fmt(ap_v),  uname in top3)

        paper_url = str(m.get("Approach", ""))
        review_key = None
        for k in review_scores:
            pass  # match by author name heuristic
        rev_score = "—"
        rev_dec   = "—"
        for k in review_scores:
            author_lower = str(author).split()[0].lower().rstrip(".")
            if author_lower in k.lower():
                rev_score = str(review_scores[k])
                rev_dec   = review_decisions.get(k, "—")
                break

        lines.append(
            f"{rank} & {author} & {approach} & {arch} & {params} & "
            f"{train_data} & {tta} & {ens} & {dur} & "
            f"{f1_str} & {froc_str} & {ap_str} & {rev_score} & {rev_dec} \\\\"
        )

    body = "\n".join(lines)
    tex = rf"""\begin{{sidewaystable}}
\centering
\caption{{Track~1 (mitotic figure detection) final leaderboard with algorithmic details. $F_1$, FROC-AUC (up to 8\,FP/image) and average precision (AP) are reported on the final test set. OD = object detection; Seg.\ = segmentation.}}
\label{{tab:t1_full}}
\small\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{clllrllllccccl}}
\toprule
Rank & First author & Approach & Architecture & Params & Training data & TTA & Ens. & Time (s) & $F_1$ & FROC-AUC & AP & Review & Decision \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\footnotesize Training data: M++=MIDOG++, CMC=WSI\_CMC, CCMCT=WSI\_CCMCT, Ext.=additional external dataset(s). Params: single model parameter count. Time: mean inference time per ROI on RTX~4090. Review: mean peer review score (max.~15). Top-3 results in bold.
\end{{sidewaystable}}
"""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(tex)
    print(f"Saved → {save_path}")
    return tex


# ── Track 2 table ─────────────────────────────────────────────────────────────

def build_t2_table(evaluations_path=None, t2_results=None, save_path="tab_t2_full.tex"):
    if evaluations_path is None:
        evaluations_path = EVALUATIONS_PATH
    if t2_results is None:
        t2_results = load_track2_results()

    methods = pd.read_excel(
        os.path.join(evaluations_path, "MIDOG 2025 - Track 2 - Methods-3.xlsx")
    )

    review_scores    = _load_review_scores(evaluations_path)
    review_decisions = _load_review_decisions(evaluations_path)

    ba_scores   = t2_results["balanced_accuracy"]
    auc_scores  = {}

    for p in T2_PARTICIPANTS:
        import os as _os, json as _json
        from utils import TRACK2_OUTPUT_PATH
        mpath = _os.path.join(TRACK2_OUTPUT_PATH, p, "metrics.json")
        if _os.path.exists(mpath):
            j = _json.load(open(mpath))
            agg = j["aggregates"]
            auc_scores[p] = agg.get("overall_roc_auc", np.nan)

    participants_sorted = sorted(ba_scores.keys(),
                                 key=lambda p: ba_scores[p], reverse=True)
    top3 = set(participants_sorted[:3])
    durations = t2_results["duration"]

    lines = []
    for rank, uname in enumerate(participants_sorted):
        row_m = methods[methods["Username"].str.strip() == uname]
        if row_m.empty:
            row_m = methods[methods["Username"].str.strip().str.lower() == uname.lower()]
        m = row_m.iloc[0] if not row_m.empty else pd.Series(dtype=object)

        author    = _trunc(m.get("Authors", "—"), 18)
        approach  = _trunc(m.get("Approach", "—"), 18)
        arch      = _trunc(m.get("Base architecture", "—"), 24)
        params_raw = m.get("Single model size (parameters)", np.nan)
        try:
            val = float(params_raw)
            params = f"{int(val) // 1_000_000}M" if val > 1e6 else f"{int(val / 1000)}K"
        except (TypeError, ValueError):
            params = "—"
        train_data = _abbrev_training_t2(m) if not m.empty else "—"
        tta = _bool_cell(m.get("TTA used", 0))
        ens = "Yes" if str(m.get("Ensembling", "0")) not in ("0", "nan", "No", "") else "No"
        dur = f"{durations.get(uname, np.nan):.1f}" \
              if not np.isnan(durations.get(uname, np.nan)) else "—"

        ba_v  = ba_scores.get(uname, np.nan)
        auc_v = auc_scores.get(uname, np.nan)

        def bold(s, cond):
            return rf"\textbf{{{s}}}" if cond else s

        ba_str  = bold(_fmt(ba_v),  uname in top3)
        auc_str = bold(_fmt(auc_v), uname in top3)

        rev_score = "—"
        rev_dec   = "—"
        for k in review_scores:
            author_lower = str(author).split()[0].lower().rstrip(".")
            if author_lower in k.lower():
                rev_score = str(review_scores[k])
                rev_dec   = review_decisions.get(k, "—")
                break

        lines.append(
            f"{rank} & {author} & {approach} & {arch} & {params} & "
            f"{train_data} & {tta} & {ens} & {dur} & "
            f"{ba_str} & {auc_str} & {rev_score} & {rev_dec} \\\\"
        )

    body = "\n".join(lines)
    tex = rf"""\begin{{sidewaystable}}
\centering
\caption{{Track~2 (atypical mitotic figure classification) final leaderboard with algorithmic details. Balanced accuracy and ROC AUC are reported on the final test set.}}
\label{{tab:t2_full}}
\small\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{clllrllllccl l}}
\toprule
Rank & First author & Approach & Architecture & Params & Training data & TTA & Ens. & Time (s) & Bal.\ Acc. & ROC AUC & Review & Decision \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\footnotesize Training data: M25=MIDOG~2025 atypical set, AMi-Br=AMi-Br atypical dataset, OMG=OMG-Octo atypical dataset, Ext.=additional external dataset(s). Params: single model parameter count. Time: mean inference time per ROI on RTX~4090. Review: mean peer review score (max.~15). Top-3 results in bold.
\end{{sidewaystable}}
"""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(tex)
    print(f"Saved → {save_path}")
    return tex


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LaTeX leaderboard tables.")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()
    out = args.outdir

    print("Loading Track 1 results…")
    t1 = load_track1_results()
    build_t1_table(t1_results=t1, save_path=os.path.join(out, "tab_t1_full.tex"))

    print("Loading Track 2 results…")
    t2 = load_track2_results()
    build_t2_table(t2_results=t2, save_path=os.path.join(out, "tab_t2_full.tex"))
