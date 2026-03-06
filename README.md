# Mouse EKG AF Detection

Detects Atrial Fibrillation (AF) in mouse EKG recordings using HRV feature extraction and a Balanced Random Forest classifier validated against expert manual annotations.

## Setup

```bash
pip install -r requirements.txt
```

Place raw data in the project root before running:

```
AF Analysis/
├── EKG Recordings/              ← Raw iWorx exports (*_Export.xls or *_Export.txt)
└── EKG_Annotations (1).xlsx    ← Master annotation sheet with episode timestamps
```

## Project Structure

```
AF Analysis/
├── scripts/
│   ├── extract_features.py          # HRV extraction + sliding window pipeline
│   ├── train.py                     # Balanced RF classifier (LOFO-CV)
│   ├── benchmark.py                 # LazyPredict comparison across 26 classifiers
│   ├── summarize.py                 # Dataset statistics
│   ├── label_episodes.py            # Automatic paroxysmal episode boundary detection
│   ├── run_pipeline.py              # Top-level entry point
│   ├── visualize_trace.py           # EKG trace + peak visualization
│   └── visualize_hrv.py             # HRV box plots (AF vs No-AF)
│
├── Experiments/                     # Experiment logs (000–004)
├── Results/                         # Outputs: CSVs, plots
├── EKG_Annotations (1).xlsx
└── README.md
```

## Workflow

### Step 1: Dataset Summary
```bash
python scripts/summarize.py
```
Outputs: `Results/dataset_summary.csv`, `Results/per_file_stats.csv`

### Step 2: Feature Extraction
```bash
python scripts/run_pipeline.py
```
Processes each recording through a 3Hz high-pass filter, sliding window peak detection (10s window, 2s step), and HRV feature calculation. Also appends temporal context features (`prev_*` / `next_*`) so the classifier can see whether neighboring windows are similarly irregular.

Outputs: `Results/all_results_summary.csv`, `Results/af_vs_nonaf_pvalues.csv`

### Step 3: Train Classifier
```bash
python scripts/train.py
```
Trains a Balanced Random Forest using Leave-One-File-Out cross-validation. Evaluates BalancedRF, XGBoost, and a soft-voting ensemble. Picks the model with the highest PR-AUC and applies a decision threshold optimized for Recall ≥ 0.80.

For paroxysmal AF files, a window is only labeled AF if ≥ 5 seconds of the window overlaps with a known AF episode (prevents mislabeling boundary windows).

Outputs: `Results/rf_pr_curve.png`, `Results/rf_feature_importance.csv/png`

### Step 4: Benchmark Models
```bash
python scripts/benchmark.py
```
Runs all standard sklearn classifiers via LazyPredict on an 80/20 file-stratified split for a quick comparison baseline.

Output: `Results/lazypredict_benchmark.csv`

## Features

| Domain | Metrics |
|---|---|
| **Time domain** | `RR_Mean`, `SDNN`, `RMSSD`, `pNN20/50/100`, `HR_Mean`, `HR_Std` |
| **Frequency domain** | `VLF`, `LF`, `HF`, `LF/HF` (Welch) |
| **Nonlinear** | `SD1`, `SD2`, `CSI`, `CVI` (Poincaré) |
| **Temporal context** | `prev_*`, `next_*` variants of all features above |

## Notes

- Peak detection is tuned for mouse physiology (300–900 BPM).
- Annotation matching is fuzzy — filenames are normalized before lookup to handle `.xls` vs `.iwxdata` differences.
