# Experiment 000 — File-Level Labeling Baseline (Retrospective)

**Date:** ~2026-01 through 2026-02-18  
**Author:** Kushaan Sharma  
**Status:** Superseded — see Experiment 001  

## Goal

Establish a baseline pipeline for AF classification in mouse EKG recordings. The approach applied a single AF label to each entire recording file and trained a Random Forest on recording-level HRV metrics.

## Approach

### Data
- Source directory: `GroundtruthverifiedEKGData/`
- Labels: File-level from `EKG_Annotations.xlsx` (AF_Label: Yes/No per recording)
- Each recording treated as one observation

### Processing
- Recordings loaded from LabChart tab-separated exports (`*_Export.xls`, `*_Export.txt`)
- Signal preprocessed: 3 Hz Butterworth HPF + NeuroKit2 biosppy cleaning
- R-peaks detected via multi-detector ensemble (NeuroKit2, prominence, adaptive threshold)
- HRV features extracted over the **full recording**
- Each recording = one row in the training dataset

### Model
- Random Forest Classifier (`n_estimators=100`, `class_weight='balanced'`)
- Leave-One-File-Out cross-validation

## Parameters

| Parameter | Value |
|-----------|-------|
| Analysis unit | Entire recording (no segmentation) |
| Label granularity | File-level (1 label per recording) |
| RF: n_estimators | 100 |
| RF: class_weight | balanced |
| Validation | Leave-One-File-Out |

## Problems Identified

> These issues were noted in the Feb 18, 2026 meeting with Dr. Wang and Dr. Pu.

1. **Too few observations:** With ~tens of recordings total, the classifier had very limited training data. Each recording contributed only one sample.

2. **Label noise for mixed rhythms:** Recordings containing both normal sinus rhythm and AF episodes received a single AF label. This propagated incorrect labels to non-AF intervals, adding noise to the feature space.

3. **No temporal localization:** The model could classify a file as AF vs. non-AF but could not identify *when* within a recording AF occurred. The clinical goal is to locate AF episodes, not just flag files.

4. **HRV instability over full recording:** Very long recordings may have non-stationary statistics. Aggregating HRV over an entire recording obscures short-burst paroxysmal AF episodes.

## Outcome

Approach was replaced by the sliding window strategy (Experiment 001) per Dr. Wang's recommendation to:
- Increase observations by ~4–5× via overlapping windows
- Enable temporal localization of AF episodes

## Reference

Udawat & Singh (2022) — Paroxysmal AF detection using sliding window + RR interval feature extraction + window-level classification.
