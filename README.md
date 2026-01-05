# Mouse EKG AF Analysis & Machine Learning Tool

This tool analyzes mouse EKG data to detect Atrial Fibrillation (AF). It implements comprehensive HRV feature extraction (Time, Frequency, Nonlinear) and uses a **Random Forest classifier verified against manual annotations** to identify the most predictive parameters.

## 1. Setup

Ensure you have Python installed. Then, install the required libraries:

```bash
pip install pandas neurokit2 scipy numpy matplotlib scikit-learn openpyxl
```

## 2. Directory Structure

The project is organized as follows:

```
AF Analysis/
├── Analysis_Scripts/           # MAIN PIPELINE
│   ├── analyze_ekg.py          # Step 1: Extracts features & computes statistics
│   ├── visualize_ekg.py        # Step 2: Generates full-trace plots for verification
│   ├── train_classifier.py     # Step 3: Trains ML model using manual annotations
│   └── summarize_analysis.py   # Optional: Summarizes results per file
│
├── GroundtruthverifiedEKGData/ # INPUT DATA (Verified Recordings & Annotations)
│   ├── EKG_Annotations.xlsx    # Master Excel file with manual "Yes/No" labels
│   └── *.xls / *.iwxdata       # Raw EKG files
│
├── Verification_Scripts/       # UTILITIES
│   ├── verify_p_values.py      # Sanity check for statistics
│   └── run_roc_analysis.py     # Generates ROC curves
│
├── EKGAFAnnotations/           # Storage for additional annotation files
├── Logs/                       # Processing logs and debug info
├── Results/                    # OUTPUT folder (CSVs, Feature Importance, P-values)
└── Verification_Plots/         # OUTPUT folder (Trace images)
```

## 3. Workflow

To run the tools, open a terminal in the main folder and navigate to `Analysis_Scripts` (or run from root as shown below):

### Step 1: Run Analysis (`analyze_ekg.py`)
This script processes the files in `GroundtruthverifiedEKGData`. It:
1.  Applies a **3Hz High-Pass Filter** to remove wander.
2.  Detects peaks using an adaptive threshold.
3.  Calculates Time, Frequency, and Nonlinear HRV metrics.
4.  Generates a master results file (`Results/all_results_summary.csv`).
5.  Calculates **P-Values** comparing AF vs Non-AF segments (based on rule-based metric pRR3.25 > 75% as a preliminary screen).

```bash
python Analysis_Scripts/analyze_ekg.py
```

**Outputs (`Results/`):**
*   `all_results_summary.csv`: Feature values for every 10s window.
*   `af_vs_nonaf_pvalues.csv`: Ranked statistical significance of features.

### Step 2: Verify Traces (`visualize_ekg.py`)
Generates plots showing the **entire EKG trace** for each file to ensure signals are clean and peaks are detected correctly. This is critical for visual confirmation.

```bash
python Analysis_Scripts/visualize_ekg.py
```

**Outputs (`Verification_Plots/`):**
*   `trace_*.png`: Full EKG signal with detected peaks (red dots).

### Step 3: Train Machine Learning Classifier (`train_classifier.py`)
Trains a Random Forest classifier.
*   **Input**: `Results/all_results_summary.csv` (features) + `GroundtruthverifiedEKGData/EKG_Annotations.xlsx` (labels).
*   **Method**: Uses **Leave-One-Group-Out Cross-Validation**. It trains on N-1 mice and tests on the remaining mouse to ensure the model generalizes to new subjects.
*   **Target**: Predicts the **Manual Label** (Human Diagnosis) using only the calculated metrics.

```bash
python Analysis_Scripts/train_classifier.py
```

**Outputs (`Results/`):**
*   **Console Report**: Accuracy, Sensitivity, Precision, F1-Score.
*   `rf_feature_importance.png`: Bar chart of the top 20 features that predict the human diagnosis.
*   `rf_feature_importance.csv`: Exact importance scores.

### Step 4: ROC Analysis (`Analysis_Scripts/run_roc_analysis.py`)
Generates ROC curves to test the performance of specific metrics (like `pRR_3.25`) against the manual ground truth.
*   **Input**: `GroundtruthverifiedEKGData/*.xls` (Raw Signals) + `EKG_Annotations.xlsx` (Labels).

```bash
python Analysis_Scripts/run_roc_analysis.py
```

**Outputs (`Results_Verification/`):**
*   `mouse_af_roc_curve.png`: ROC Curve showing Area Under Curve (AUC).
*   `optimal_mouse_thresholds.csv`: The best cut-off value for the metric to maximize Sensitivity + Specificity.

## 4. Calculated Parameters
The tool calculates the following metrics for every window:
*   **Time-Domain**: `RR_Mean`, `SDNN`, `RMSSD`, `SDSD`, `NNx` (20, 50, 100), `pNNx`.
*   **Frequency-Domain**: `VLF`, `LF`, `HF`, `LF/HF` (using Welch's method via NeuroKit2).
*   **Nonlinear**: `SD1`, `SD2`, `Area` (Ellipse), `CSI`, `CVI`, `Modified_CSI` (Poincaré plot metrics).

## 5. Notes for Users
*   **Ground Truth**: The classifier strictly requires `EKG_Annotations.xlsx` to be present in the data folder. Filenames in the Excel sheet must loosely match the data filenames.
*   **Circular Logic Avoided**: The classifier is trained on *manual* labels, not the rule-based thresholds calculated in Step 1.
