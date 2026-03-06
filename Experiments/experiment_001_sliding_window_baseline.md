# Experiment 001 — Sliding Window Baseline

**Date:** 2026-02-19 → Run: 2026-02-20  
**Author:** Kushaan Sharma  
**Status:** Complete — Full pipeline run with 117 annotated files  

---

## Goal

Replace the fixed 10-second block analysis with a true overlapping sliding window to multiply the number of observations available for classifier training. 

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window length | 10 s | Chosen as a balance: short enough to capture paroxysmal AF, long enough for stable RR interval estimates |
| Step size | 2 s | 80% overlap between consecutive windows |
| Sampling rate (target) | 1000 Hz | Data is downsampled from 5000 Hz if needed |
| Pre-processing | Butterworth 2nd order HPF @ 3 Hz + NeuroKit biosppy cleaning
| Signal trim | 2 s from start |
| pRR threshold for rule-based AF flag | 3.25% | % RR changes ≥ 3.25%; threshold derived empirically |
| AF classification threshold (pRR value) | 75.32% | A window is flagged AF if pRR_3.25 > 75.32 |
| RF: n_estimators | 100 | |
| RF: class_weight | balanced | Compensates for AF/non-AF class imbalance |
| Validation strategy | Leave-One-File-Out (LOGO) | Prevents data leakage; each recording is held out once |
| Imputation | Median | For windows with failed HRV calculations |

---

## Key Design Decisions

1. **Sliding window vs. fixed blocks:** Using `step_size_sec = 2.0` with a 10 s window gives ~4–5× more observations per recording compared to non-overlapping 10 s blocks. This is critical because the current dataset has only a few dozen recordings.

2. **Rule-based `is_af` flag vs. `manual_label`:** `is_af` is computed automatically using the pRR threshold. `manual_label` is loaded from `EKG_Annotations.xlsx` and used as the ground truth for training the RF classifier. This allows us to compare the rule-based detection with the trained model.

3. **Window-level labeling limitation (current):** Each window within a recording inherits the file-level label from `EKG_Annotations.xlsx`. This means a mixed-rhythm AF recording will have *all* its windows labeled AF — even quiet intervals. This is acknowledged as a source of label noise. The next step (Experiment 002) will implement timestamp-based window labeling when episode start/end times become available.

4. **Peak detection ensemble:** Three detectors are tried per recording (NeuroKit2, prominence scipy, adaptive threshold). The detector with the lowest RR coefficient of variation (CV) in the physiologically plausible HR range (300–900 bpm for mice) is selected.

---

## Features Extracted Per Window

| Category | Features |
|----------|---------|
| pRR | `pRR_3.25` |
| Time domain | `RR_Mean`, `RR_Min`, `RR_Max`, `HR_Mean`, `HR_Min`, `HR_Max`, `HR_Std`, `SDNN`, `RMSSD`, `SDSD` |
| NNx | `NN20`, `NN50`, `NN100`, `pNN20`, `pNN50`, `pNN100` |
| Frequency domain | `VLF`, `LF`, `HF`, `LFHF` |
| Nonlinear (Poincaré) | `SD1`, `SD2`, `SD1SD2`, `CSI`, `CVI`, `Modified_CSI`, `Area` |

---

## Scripts Involved

| Script | Role |
|--------|------|
| `Analysis_Scripts/analyze_ekg.py` | EKG loading, preprocessing, sliding window, feature extraction |
| `Analysis_Scripts/train_classifier.py` | RF model training, LOGO-CV, feature importance |
| `Analysis_Scripts/summarize_dataset.py` | Dataset overview stats (mouse count, duration, file counts) |
| `Analysis_Scripts/run_roc_analysis.py` | ROC curve + pRR threshold selection |

---

## Results

### Dataset Summary (`summarize_dataset.py`)

| Metric | Value |
|--------|-------|
| Total recordings | 118 |
| Unique mice | 59 |
| Total recording duration | 321.6 min (5.4 hr) |
| Mean duration per recording | 2.7 min |
| AF recordings | 25 (sustained: 22, paroxysmal: 3) |
| No-AF recordings | 93 |
| AF recording duration | 59.9 min |
| No-AF recording duration | 261.8 min |

Outputs: `Results/dataset_summary.csv`, `Results/per_file_stats.csv`

---

### Feature Extraction (`analyze_ekg.py`)

| Metric | Value |
|--------|-------|
| Files processed | 117 |
| Total windows | 1,170 |
| Window length | 10 s |
| Extraction Range | 10 seconds to 40 seconds |
| Step size | 2 s (80% overlap) |
| Rule-based AF windows | 344 / 1,170 (29.4%) |

### Classifier Training (train_classifier.py)

**Dataset used:** `EKG_Annotations (1).xlsx` — 127 records (24 sustained + 4 paroxysmal + 99 No-AF)  
**Matched to pipeline windows:** 1,170 windows from 117 files

Label breakdown after window assignment:

| AF Type | Windows |
|---------|---------|
| Sustained AF | 210 |
| Paroxysmal AF | 30 |
| No-AF | 930 |

**Leave-One-File-Out Cross-Validation (117 folds):**

| Metric | Mean | ± Std |
|--------|------|-------|
| Accuracy | **87.4%** | ±27.4% |
| Precision | 15.4% | ±39.7% |
| Recall | 11.7% | ±29.9% |
| F1 Score | 12.6% | ±32.6% |

**Top 10 RF Feature Importances:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | pNN20 | 0.136 |
| 2 | NN20 | 0.100 |
| 3 | pRR_3.25 | 0.092 |
| 4 | HR_Max | 0.055 |
| 5 | RR_Mean | 0.050 |
| 6 | RR_Min | 0.048 |
| 7 | RR_Max | 0.042 |
| 8 | HR_Mean | 0.042 |
| 9 | HR_Min | 0.042 |
| 10 | RMSSD | 0.039 |

Outputs saved to `Results/`:
- `all_results_summary.csv` — 1,170 rows of per-window HRV features (117 files)
- `rf_feature_importance.csv` + `rf_feature_importance.png`

---

## Interpretation

**Consistent accuracy (87.4%) but extremely low precision/recall (≈12-15%)** — Restricting the analysis to strictly the annotated 10s-40s segments significantly reduced the number of observations (from 9,003 to 1,170 windows), but the precision and recall scores remained poor. 

- This definitively rules out "out-of-bounds" windows in mixed files as the primary source of label noise. The 10-40s segments *themselves* likely contain a mix of normal sinus rhythm and paroxysmal AF bursts.
- Even when strictly analyzing the 10-40s annotated range, assigning an "AF" label to every single 10-second window within that 30-second block means capturing quiet intervals between bursts, inflating false positives.
- **Initial window-level validation:** Providing explicit `Episode Start` and `Episode End` bounds for paroxysmal files successfully filters these normal sinus rhythm gaps out. In an early dummy test on just 4 paroxysmal files, applying a 15s-25s burst bound successfully reverted 9 non-fibrillating sliding windows back to the No-AF class, directly reducing false label noise.
- The 4× class imbalance (930 No-AF vs 240 AF) continues to suppress recall even with `class_weight='balanced'`.

**Most predictive features are RR-interval variability metrics:** pNN20, NN20, pRR_3.25, HR_Max — all biologically coherent (AF = irregular, fast RR intervals in mice).
*(Per Sunny's note, this aligns with medical knowledge: AF is characterized by higher RR variation, represented strongly by NN20 and pNN20 differences between AF and non-AF.)*

**pRR_3.25 ranks only 9th** even though it's the most statistically different (p → 0). This is because in a RF, correlated features (NN20, pNN20, pRR) split importance among themselves.

---

## Known Issues / Limitations

1. **File-level label noise:** All windows in a mixed AF recording are labeled AF — including non-AF intervals. This inflates recall at the cost of precision.
2. **Short windows + VLF estimates:** Windows < ~30 s produce unreliable VLF power estimates (NeuroKit2 will warn). Suppressed via `warnings.filterwarnings`.
3. **Minimum peaks per window:** Windows with < 3 detected peaks return `None` for HRV metrics and are filled with `is_af = False, pRR_3.25 = 0`. These windows are still included in the CSV with a `note = 'Insufficient peaks'` flag.
4. **No frequency-domain normalization:** LF/HF power values are in absolute (ms²) units. These may be unstable for very short windows.

---

## Next Steps → Experiment 002

1. **Obtain AF episode timestamps** from Dr. Wang/Dr. Li → enables window-level labels instead of file-level labels (directly addresses the precision/recall problem)
2. **Tune window size:** Try 5 s and 15 s windows — smaller windows may better isolate short paroxysmal bursts
3. **Separate sustained vs. paroxysmal models:** With enough data, train two classifiers or a 3-class model
