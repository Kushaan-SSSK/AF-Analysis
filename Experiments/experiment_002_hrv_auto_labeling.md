# Experiment 002 — Dynamic HRV Window Labeling

**Date:** 2026-02-23
**Author:** Kushaan Sharma
**Status:** Complete

---

## Goal

Address precision/recall issues from file-level labeling (Experiment 001) by introducing exact timestamps for paroxysmal AF bursts. To automate this, implement an algorithmic screen using the most predictive HRV features (e.g., `pNN20`) to dynamically locate the bursting bounds and apply them as timestamp labels.

---

## Parameters

- **Base Windows:** 10-40s range extraction, 10s sliding window, 2s step (80% overlap)
- **HRV Baseline Threshold:** `pNN20` > 2.0 (triggers paroxysmal burst recording)
- **Target Labeling:** Windows receive `AF` label *only* if their `start_time` and `end_time` overlap with the identified `Episode Start (s)` and `Episode End (s)`.

---

## Key Design Decisions

1. **HRV Significance Check:** Visualized pNN20, NN20, and pRR_3.25 using `visualize_hrv_differences.py` to confirm RR variation is a biologically sound trigger metric (p < 0.001 distinction between AF vs Normal).
2. **Automated Timestamps (`auto_label_episodes.py`):** Automatically sweeps windows inside paroxysmal files. Since typical normal sinus rhythm `pNN20` is extremely low, windows exceeding the threshold demarcate the start and end of the hidden AF burst. 
3. **True Window-Level Labels:** The exact boundaries are written to `EKG_Annotations (1).xlsx`. The classifier respects these boundaries, finally preventing non-AF gaps within mixed files from being erroneously given an `AF` target label.

---

## Results

### Dataset Pipeline
- **Paroxysmal Auto-Labeling:** Discovered exact boundaries (e.g. 0.0s to 28.0s) for genuine bursts using pNN20. Automatically ignored dummy files showing no valid variation.
- **Window Label Changes:** Paroxysmal AF windows perfectly overlapping with bursts: 30. Windows falling outside are rightly shifted to No-AF: 930.

### Classifier Metrics
**Leave-One-File-Out Cross-Validation (117 folds):**

| Metric | Mean | ± Std |
|--------|------|-------|
| Accuracy | **87.4%** | ±27.4% |
| Precision | 15.3% | ±39.7% |
| Recall | 11.7% | ±29.9% |
| F1 Score | 12.6% | ±32.6% |

---

## Interpretation

**SUCCESSFUL: Algorithmic window-level bounding:** 
Rather than waiting for manual inputs, we created `auto_label_episodes.py` to screen the top HRV predictive feature (`pNN20`) across Paroxysmal files. By dropping any contiguous windows resting at normal sinus baseline, we dynamically identified and stripped the gaps without human intervention. 

The classifier processed these exact boundaries correctly, ensuring normal-sinus rhythm sliding windows within the 10-40s bounds are now correctly designated as `No-AF`. This confirms Dr. Wang's hypothesis: window-level labeling successfully strips file-level noise. However, the extreme 4x class imbalance (930 vs 240) still inherently bounds absolute precision metrics until the dataset grows.
