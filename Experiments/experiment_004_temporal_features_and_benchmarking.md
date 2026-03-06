# Experiment 004 — Temporal Continuity Features and Benchmarking

**Date:** 2026-03-05
**Author:** Kushaan Sharma
**Status:** Complete

---

## Goal

Improve AF detection model recall to > 0.80 by (1) incorporating temporal continuity features from neighboring windows, (2) tuning the decision threshold to jointly satisfy Accuracy > 0.90 and Recall > 0.80, and (3) benchmarking a range of classifiers using `lazypredict`.

---

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target Class Ratio | ~1:4 | 240 AF+ vs 930 No-AF |
| RF: Classifier | BalancedRandomForest | Down-samples majority class per-tree |
| n_estimators | 500 | Increased from 100 |
| Temporal Features | `prev_*`, `next_*` | One-step lag and lead for all 24 metrics |
| Decision Threshold | 0.3767 | Best accuracy while enforcing Recall >= 0.80 |

---

## Key Changes vs. Previous Experiment

- Added **temporal continuity features**: `prev_*` and `next_*` variants of all HRV metrics to capture information from neighboring windows. These became the top predictors in the final model.
- Expanded the threshold selection strategy to enforce a **dual constraint**: Recall >= 0.80 AND Accuracy >= 0.90 simultaneously. Falls back to maximizing accuracy among all recall-satisfying thresholds if joint constraint is unachievable.
- Evaluated three models via LOFO CV: **BalancedRF (n=500)**, **XGBoost (scale_pos_weight)**, and a **soft-voting ensemble**. BalancedRF won.
- Benchmarked 26 classifiers using `lazypredict` on an 80/20 file-grouped split.

---

 [ 48 192]]
```

**Top features:** `prev_pNN20`, `next_pNN20`, `pNN20`, `next_pRR_3.25`, `pRR_3.25`

**Benchmark (top 3 models, lazypredict, untuned):**
| Model | Accuracy |
|-------|----------|
| XGBClassifier | 79.0% |
| PassiveAggressiveClassifier | 78.0% |
| BaggingClassifier | 78.0% |

> Full benchmark: `Results/lazypredict_benchmark.csv`

---

## Interpretation

**Recall target achieved (80%)**, reducing the false negative rate from 38% (Exp 003) to **20%** — only 48 AF windows missed out of 240. This is a direct improvement in clinical utility.

**Accuracy target (> 90%) was not achievable simultaneously.** Enforcing Recall >= 0.80 lowers the decision threshold, which unavoidably increases false positives (202 No-AF windows flagged). With the current dataset (PR-AUC ≈ 0.74), no threshold satisfies both Accuracy > 0.90 and Recall > 0.80 at the same time. This is a known dataset-level limitation: with ~1.2K samples and a 1:4 class imbalance, the model's discriminative power is capped. A PR-AUC of ~0.90+ (needed to satisfy both constraints) would require either more training data or richer signal features (e.g., P-wave morphology, QRS duration, raw waveform features). This is documented as a future direction.

**Temporal context is highly informative.** The top 5 features are all window-local or lagged/led variants of pNN20 and pRR_3.25, confirming that AF episodes manifest consistently across consecutive windows — validating the clinical rationale for incorporating temporal continuity.
