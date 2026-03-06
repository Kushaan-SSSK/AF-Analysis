import pandas as pd
import numpy as np
import os

def auto_label_paroxysmal_episodes():
    print("Auto-Labeling Paroxysmal AF episodes using HRV Metrics...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, "Results")
    
    # 1. Load the summary results (contains the 10-40s sliding windows)
    summary_path = os.path.join(results_dir, "all_results_summary.csv")
    if not os.path.exists(summary_path):
         print("Missing all_results_summary.csv")
         return
    df_results = pd.read_csv(summary_path)
    
    # 2. Load Ground Truth Excel
    annot_path = os.path.join(base_dir, "EKG_Annotations (1).xlsx")
    if not os.path.exists(annot_path):
         print("Missing EKG_Annotations (1).xlsx")
         return
    
    df_annot = pd.read_excel(annot_path)
    df_annot.columns = [c.strip() for c in df_annot.columns]
    
    label_col = next((c for c in df_annot.columns if c.strip() == 'AF' or 'AF' in c), None)
    file_col = df_annot.columns[0]
    
    if 'Episode Start (s)' not in df_annot.columns:
        df_annot['Episode Start (s)'] = None
    if 'Episode End (s)' not in df_annot.columns:
        df_annot['Episode End (s)'] = None
        
    def normalize_name(name):
        return os.path.splitext(os.path.basename(name))[0].replace('_Export', '').replace('.iwxdata', '').lower().strip()

    # Create mapping of file normalized names to their row index in df_annot for writing
    row_map = {}
    for idx, row in df_annot.iterrows():
         row_map[normalize_name(str(row[file_col]))] = idx
         
    # 3. Define the Paroxysmal Threshold Screen based on our Feature Importance
    # Mean Normal Sinus pNN20 is usually < 1.0 (mean variation is tiny).
    # Bursting AF easily leaps to 5-20. We will set a conservative screen threshold.
    PNN20_THRESHOLD = 2.0 
    
    updated_files = 0
    
    # Isolate uniquely 'paroxysmal' rows
    for idx, row in df_annot.iterrows():
        label_str = str(row[label_col]).lower().strip()
        if 'paroxysmal' not in label_str:
            continue
            
        base_name = normalize_name(str(row[file_col]))
        
        # Give us all the 10-s blocks belonging to this paroxysmal file
        file_windows = df_results[df_results['file'].apply(lambda x: normalize_name(x) == base_name or normalize_name(x) in base_name)].copy()
        
        if file_windows.empty:
            continue
            
        file_windows = file_windows.sort_values(by='start_time')
        
        # Find which windows trip the AF Screen Threshold for our top indicator (pNN20)
        file_windows['is_burst'] = file_windows['pNN20'] > PNN20_THRESHOLD
        
        burst_windows = file_windows[file_windows['is_burst']]
        
        if burst_windows.empty:
            print(f"Skipping {base_name}: No windows breached the pNN20 threshold.")
            continue
            
        # Define the contiguous boundary (lowest start of the burst -> highest end of the burst)
        ep_start = burst_windows['start_time'].min()
        ep_end = burst_windows['end_time'].max()
        
        # Write back to Excel logic
        df_annot.at[idx, 'Episode Start (s)'] = ep_start
        df_annot.at[idx, 'Episode End (s)'] = ep_end
        
        print(f"Matched Paroxysmal Burst precisely bounds for {base_name}: {ep_start}s to {ep_end}s")
        updated_files += 1

    if updated_files > 0:
        print(f"\nSuccessfully auto-labeled {updated_files} paroxysmal records based on HRV dynamic thresholds.")
        print(f"Overwriting {annot_path} with automated exact timestamps.")
        df_annot.to_excel(annot_path, index=False)
    else:
        print("\nNo new paroxysmal bounds needed to be updated.")

if __name__ == "__main__":
    auto_label_paroxysmal_episodes()
