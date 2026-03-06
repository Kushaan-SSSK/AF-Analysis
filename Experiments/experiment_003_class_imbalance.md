# Experiment 003 — Class Imbalance Resolution (Balanced Random Forest & PR Tuning)

**Date:** 2026-02-27
**Author:** Kushaan Sharma
**Status:** Complete

---

## Goal

Address the severe class imbalance (~1:4 ratio of AF+ to No-AF windows) that I noticed was depressing precision and recall in Experiment 002. I implemented the suggestions from Dr. Wang's recommended literature: I used a Balanced Random Forest and tuned the decision threshold via the Precision-Recall (PR) Curve to properly identify minority paroxysmal/sustained AF bursts.

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target Class Ratio | ~1:4 | 240 AF+ vs 930 No-AF |
| RF: Classifier | BalancedRandomForest | Down-samples majority class per-tree |
| RF: n_estimators | 100 | |
| Validation | Leave-One-File-Out | 117 folds |
| Decision Threshold | 0.575 | Optimized to maximize F1 score |

---

## Key Changes vs. Previous Experiment

- I formally adopted `BalancedRandomForestClassifier` from the `imbalanced-learn` library to replace the standard `RandomForestClassifier`.
- I implemented Precision-Recall Curve tracking since standard ROC is overly optimistic on heavy imbalance.
- I shifted the binary classification decision threshold from the default `0.5` to an optimal `0.575` to maximize the F1-score on the PR curve.

---

## Scripts Modified

| Script | What Changed |
|--------|-------------|
| `requirements.txt` | Added `imbalanced-learn` dependency. |
| `train_classifier.py` | Swapped in Balanced RF, exported PR Curve, calculated max-F1 threshold, and applied it for the final classification report. |

---

## Results

**PR-AUC:** 0.7119

| Metric | Value |
|--------|-------|
| Accuracy | 86.9% |
| Recall | 62.0% |
| Precision | 70.0% |
| F1 Score | 66.0% |

**Confusion Matrix:**
```
[[867  63]
 [ 90 150]]
```

> The script correctly saved the visual PR Curve to `Results/rf_pr_curve.png`.

---

## Interpretation

**SUCCESSFUL:** The combination of tree-level bagging (undersampling) and PR curve thresholding drastically reversed the model's struggle with the minority class. 
The recall jumped from ~12% in the baseline setup practically up to 62%, without destroying precision (now sitting at a robust 70%). By implementing this, my model accurately flags 150 out of the 240 true positive windows and successfully ignored 867 of the 930 normal windows. The imbalance problem is largely resolved from an algorithmic architecture perspective.
