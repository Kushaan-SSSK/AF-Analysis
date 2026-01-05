import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from analyze_ekg import analyze_directory

def run_roc_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    ground_truth_dir = os.path.join(base_dir, "GroundtruthverifiedEKGData")
    output_dir = os.path.join(base_dir, "Results_Verification") # Previously deleted, recreates it
    os.makedirs(output_dir, exist_ok=True)
    
    annot_path = os.path.join(ground_truth_dir, "EKG_Annotations.xlsx")
    
    # 1. Load Annotations
    if not os.path.exists(annot_path):
        print(f"Error: {annot_path} not found.")
        return
        
    print("Loading Ground Truth Annotations...")
    df_annot = pd.read_excel(annot_path)
    # Use iloc to be safe against header variations
    # Col 0: Filename, Col 1: AF Label
    df_annot = df_annot.iloc[:, [0, 1]] 
    df_annot.columns = ['Filename', 'AF_Label']
    
    # Clean labels
    df_annot = df_annot.dropna(subset=['AF_Label'])
    df_annot['AF_Label'] = df_annot['AF_Label'].astype(str).str.strip().str.lower()
    df_annot['Ground_Truth'] = df_annot['AF_Label'].map({'yes': 1, 'no': 0})
    df_annot = df_annot.dropna(subset=['Ground_Truth'])
    
    gt_map = {}
    print("Sample Annotation Keys:")
    for idx, row in df_annot.iterrows():
        # Clean filename to get core identifier
        full_str = str(row['Filename']).replace('\\', '/')
        basename = os.path.basename(full_str)
        # Remove extension
        basename = basename.replace('.iwxdata', '')
        gt_map[basename] = row['Ground_Truth']
        if idx < 3: print(f"  Key: {basename} -> Label: {row['Ground_Truth']}")
        
    print(f"Loaded {len(gt_map)} ground truth labels.")
    
    # 2. Run Analysis on Files
    print("\nRunning Analysis on Ground Truth Files (with 3Hz Filter)...")
    # This uses the updated analyze_directory from analyze_ekg.py
    # It filters raw data and extracts metrics.
    results_df = analyze_directory(ground_truth_dir, output_dir, file_pattern="*_Export.xls*")
    
    if results_df.empty:
        print("No results generated.")
        return

    # 3. Match Results to Ground Truth
    # Focus on windows 1, 2, 3 (10s-40s) as per annotation instructions
    target_windows = [1, 2, 3]
    df_windowed = results_df[results_df['window_idx'].isin(target_windows)].copy()
    
    # Create a "File-Level" metric aggregation
    # We take the MAXIMUM metric value across the 3 windows for each file.
    # Logic: If any window is highly irregular, the file is likely AF (or at least that segment is).
    # Since the label is for the whole 10-40s block ("AF: Yes/No"), max aggregation makes sense for pRR.
    
    files_processed = df_windowed['file'].unique()
    matched_data = []
    
    # Pre-process GT map keys with normalization
    def normalize_name(name):
        base = os.path.basename(name)
        base = os.path.splitext(base)[0]
        base = base.replace('_Export', '').replace('.iwxdata', '')
        return base.lower().strip()

    norm_gt_map = {normalize_name(k): v for k, v in gt_map.items()}

    matched_data = []
    
    for f in files_processed:
        norm_f = normalize_name(f)
        
        # Try match in normalized map
        label = None
        if norm_f in norm_gt_map:
            label = norm_gt_map[norm_f]
        else:
             # Fuzzy match fallback
             for k, v in norm_gt_map.items():
                 if norm_f in k or k in norm_f:
                     label = v
                     break
        
        if label is None:
            # print(f"Warning: No label found for {core_name}")
            continue
            
        # Aggregate metrics for this file
        file_subset = df_windowed[df_windowed['file'] == f]
        
        # Metrics to analyze
        metrics = {
            'pRR_3.25': file_subset['pRR_3.25'].max(), # Max pRR seen
            'RMSSD': file_subset['RMSSD'].mean(),      # Avg RMSSD
            'CV': file_subset['CV'].mean() if 'CV' in file_subset else 0, # Coeff of Variation
            'Entropy': file_subset['ApEn'].mean() if 'ApEn' in file_subset else 0 # Approximate Entropy if available
        }
        
        # Add basic pRR
        metrics['Label'] = label
        metrics['File'] = f
        matched_data.append(metrics)
        
    df_analysis = pd.DataFrame(matched_data)
    print(f"\nMatched {len(df_analysis)} files to annotations.")
    print("Label Distribution:\n", df_analysis['Label'].value_counts())
    print("Metric Summary:\n", df_analysis.describe())
    print("First 5 rows:\n", df_analysis.head())
    
    if len(df_analysis) < 5:
        print("Not enough matched data for ROC analysis.")
        return

    # 4. Generate ROC Curves
    plt.figure(figsize=(10, 8))
    
    optimal_thresholds = []
    
    # We analyze pRR_3.25 primarily
    metric_cols = ['pRR_3.25']
    # Add others if they show variance
    
    for metric in metric_cols:
        y_true = df_analysis['Label']
        y_scores = df_analysis[metric]
        
        if y_scores.std() == 0:
            print(f"Skipping {metric} (no variance)")
            continue
            
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{metric} (area = {roc_auc:.2f})')
        
        # Calculate optimal threshold (Youden's Index)
        # J = Sensitivity + Specificity - 1 = tpr - fpr
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        
        optimal_thresholds.append({
            'Metric': metric,
            'Optimal_Threshold': best_thresh,
            'AUC': roc_auc,
            'Sensitivity': tpr[ix],
            'Specificity': 1 - fpr[ix]
        })
        
        # Plot optimal point
        plt.plot(fpr[ix], tpr[ix], 'ko', label=f'Best {metric} Cutoff: {best_thresh:.2f}')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve: Detection of AF in Mice (10s-40s)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(output_dir, "mouse_af_roc_curve.png")
    plt.savefig(roc_path)
    print(f"\nROC Plot saved to {roc_path}")
    
    # Save Thresholds
    thresh_df = pd.DataFrame(optimal_thresholds)
    thresh_path = os.path.join(output_dir, "optimal_mouse_thresholds.csv")
    thresh_df.to_csv(thresh_path, index=False)
    print(f"Optimal Thresholds:\n{thresh_df}")

if __name__ == "__main__":
    run_roc_analysis()
