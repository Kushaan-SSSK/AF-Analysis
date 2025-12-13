# Mouse EKG AF Analysis Tool - Usage Guide

This tool analyzes mouse EKG data for Atrial Fibrillation (AF) using an adaptive thresholding algorithm and Poincaré plot metrics.

## 1. Setup

Ensure you have Python installed. Then, install the required libraries:

```bash
pip install pandas neurokit2 scipy numpy matplotlib
```

## 2. Directory Structure

Ensure your folder looks like this:

```
AF Analysis/
├── analyze_ekg.py       # The main analysis script
├── visualize_ekg.py     # Script to generate verification plots
├── EKG data/            # Folder containing your .xls files
│   ├── Mouse1_Export.xls
│   └── ...
└── Results/             # (Created automatically) Output folder
```

**Input File Requirements:**
*   **Filename:** Must end in `_Export.xls` (e.g., `12345_Export.xls`).
*   **Format:** Tab-separated text file (standard LabChart export).
*   **Columns:** Must contain at least `Time` and `Raw Ch 1`.

## 3. Running the Analysis

Open a terminal (PowerShell or Command Prompt), navigate to the folder, and run:

```bash
python analyze_ekg.py
```

## 4. Understanding Results

The script will create a `Results` folder containing:

1.  **`all_results_summary.csv`**: A master file with metrics for *every* 10-second window across all files.
2.  **Individual CSVs**: Detailed results for each processed file.

## 5. Visualizing Verification

To generate trace plots and AF metric timelines for verification:

```bash
python visualize_ekg.py
```

This will create a `Verification_Plots` folder containing:
*   **Trace Plots**: Shows the EKG signal with red dots on detected peaks. Use this to confirm it's not missing beats.
*   **Metric Plots**: Shows the AF burden (pRR3.25%) over time for each recording.

## 6. Summarizing Results (Optional)

To generate a high-level summary (Per-File AF Burden, Mean HR):

```bash
python summarize_analysis.py
```


## 7. Validation (ROC Analysis)

To generate Receiver Operating Characteristic (ROC) curves and calculate sensitivity/specificity (requires ground truth labels):

1.  Calculates `pRR3.25%` scores for all files.
2.  **Note:** You must manually annotate the output CSV with a `label` column (1 for AF, 0 for SR) to generate the final plots.

```bash
python run_roc_analysis.py
```


