# MIDOG 2025

This repository contains analysis scripts and ablation study results for the **MIDOG 2025 challenge paper** — the Mitosis Domain Generalization Challenge 2025.

MIDOG 2025 ran two tracks on the [grand-challenge.org](https://grand-challenge.org) platform:

- **Track 1 — Mitotic Figure Detection:** Detect mitotic figures in whole-slide images across multiple tumor types and scanner domains.
- **Track 2 — Atypical Mitosis Classification:** Classify detected mitotic figures as typical or atypical.

## Repository Structure

```
Scripts/
  Track1/                    # Inference and metric-computation scripts for Track 1
  Track2/                    # Inference and metric-computation scripts for Track 2
  paper_figures/             # Publication-quality figure scripts (see below)
Ablations/
  Track1/                    # Per-team ablation HTML reports for Track 1
  Track2/                    # Per-team ablation HTML reports for Track 2
  generate_report_html.py    # Generates the ablation index page
index.html                   # Ablation study transparency portal
logo/                        # Challenge branding assets
paper/                       # Challenge paper draft
```

The `Scripts/` directory contains Python scripts that load participant Docker images (via Podman), run inference on the test set, and compute evaluation metrics. The `Ablations/` directory stores per-team HTML reports generated from those runs, rendered on the [ablation transparency portal](index.html).

## Paper Figures (`Scripts/paper_figures/`)

All figures and tables in the challenge report are generated from scripts in `Scripts/paper_figures/`. Each script is self-contained and can be run from the command line:

```bash
cd Scripts/paper_figures/
python fig_dataset_stats.py --outdir /path/to/output
python fig_rank_score_stability.py --outdir /path/to/output
python fig_inference_bubbles.py --outdir /path/to/output
python fig_ablation_studies.py --outdir /path/to/output
python fig_t1_analysis.py --outdir /path/to/output
python fig_t2_analysis.py --outdir /path/to/output
python tables.py --outdir /path/to/output
```

### Data (`Data/`)

The `Data/` directory is committed to this repository and contains everything needed to reproduce the figures — no access to the full paper workspace required.

```
Data/
  track1/{participant}/
    metrics.json          # aggregated evaluation metrics (predictions stripped)
    inference_stats.json  # peak VRAM per sample
    duration.json         # mean/std inference time (pre-aggregated from time.csv)
    ablations/{name}/
      metrics.json        # ablation aggregated metrics
  track2/{participant}/   # same structure
  metadata/
    Filenames_finaltest_MIDOG25.csv       # slide → tumor type / ROI type mapping
    MIDOG 2025 - Track 1 - Methods.xlsx  # algorithmic details per team
    MIDOG 2025 - Track 2 - Methods-3.xlsx
    MIDOG_2025_submissions_t1.json       # submission metadata (e-mails redacted)
    MIDOG_2025_submissions_t2.json
    reviews_anonymized.csv               # peer review scores (reviewer identity removed)
  precomputed/
    t1_roi_counts.csv          # per-ROI mitotic figure counts (no raw annotations)
    t1_kappa.csv               # inter-annotator κ per tumor type (Track 1)
    t2_class_distribution.csv  # AMF / NMF counts per tumor type (Track 2)
    t2_kappa.csv               # inter-annotator κ per tumor type (Track 2)
```

**What is excluded** (never committed):
- Per-image inference predictions (`mitotic-figures.json`, `multiple-mitotic-figure-classification.json`)
- Raw test-set annotations (MIDOG 2022 SQLite, consensus CSVs) — these belong to the dataset repositories
- Full reviewer identities — `reviews_anonymized.csv` keeps only scores and decisions

**Regenerating `Data/`** (requires access to the full paper workspace):
```bash
python Scripts/paper_figures/collect_data.py
```

### Environment variable overrides

If your data lives in a different location, override the defaults:

| Variable | Default | Contents |
|---|---|---|
| `MIDOG25_T1_OUTPUT` | `Data/track1` | Track 1 metrics / timing / ablations |
| `MIDOG25_T2_OUTPUT` | `Data/track2` | Track 2 metrics / timing / ablations |
| `MIDOG25_DATASETS` | `Data/metadata` | Filenames CSV, Methods Excel |
| `MIDOG25_EVALUATIONS` | `Data/metadata` | Submissions JSON, reviews CSV |
| `MIDOG25_PRECOMPUTED` | `Data/precomputed` | Pre-computed summary statistics |

### Scripts Overview

| Script | Paper figure(s) | Outputs |
|---|---|---|
| `utils.py` | — | Shared constants, style helpers, data loaders |
| `fig_dataset_stats.py` | Dataset statistics | `fig_mitotic_counts_scaled.pdf`, `fig_t2_class_distribution.pdf`, `tab_dataset_stats.csv` |
| `fig_rank_score_stability.py` | Rank/score stability | `rank_score_stability.pdf` |
| `fig_inference_bubbles.py` | Inference time | `inference_bubbles_updated.pdf`, per-track SVGs |
| `fig_ablation_studies.py` | Ablation studies | `ablation_studies.pdf`, per-metric SVGs |
| `fig_t1_analysis.py` | Track 1 analysis | `f1_heatmap.pdf`, `precision_recall_per_tumor_type_3x3.pdf`, `pr_heatmap-by-team.pdf`, `mm_heatmap-by-team.pdf`, `t1_kappa_vs_performance.pdf` |
| `fig_t2_analysis.py` | Track 2 analysis | `ba_roc_auc_tumor_type_t2.pdf`, `t2_ss_by_domain.pdf`, `ss_heatmap.pdf`, `mm_track2_heatmap.pdf`, `t2_figure_kappa_vs_sens_spec.pdf` |
| `tables.py` | Leaderboard tables | `tab_t1_full.tex`, `tab_t2_full.tex` |

## Related Repositories (DeepMicroscopy)

| Repository | Description |
|---|---|
| [MIDOG_2025_Guide](https://github.com/DeepMicroscopy/MIDOG_2025_Guide) | Getting-started guide: dataset access, environment setup, example workflows |
| [MIDOG25_T1_reference_docker](https://github.com/DeepMicroscopy/MIDOG25_T1_reference_docker) | Reference algorithm for Track 1 (Mitosis Detection) |
| [MIDOG25_T2_reference_docker](https://github.com/DeepMicroscopy/MIDOG25_T2_reference_docker) | Reference algorithm for Track 2 (Atypical Mitosis Classification) |
| [MIDOG25_T1_evaluation_docker](https://github.com/DeepMicroscopy/MIDOG25_T1_evaluation_docker) | Evaluation container for Track 1 submissions on grand-challenge |
| [MIDOG25_T2_evaluation_docker](https://github.com/DeepMicroscopy/MIDOG25_T2_evaluation_docker) | Evaluation container for Track 2 submissions on grand-challenge |
| [MIDOGpp](https://github.com/DeepMicroscopy/MIDOGpp) | The underlying multi-domain mitotic figure dataset (MIDOG++) |
| [MIDOG_reference_docker](https://github.com/DeepMicroscopy/MIDOG_reference_docker) | Domain-adversarial RetinaNet reference implementation (MIDOG 2021) |
