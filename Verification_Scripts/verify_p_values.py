import pandas as pd
import numpy as np
import os
from scipy import stats

def verify_data():
    base_dir = r"c:\Users\kusha\Downloads\AF Analysis\Results"
    file_path = os.path.join(base_dir, "all_results_summary.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run analyze_ekg.py first.")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Check for NaNs/Infs
    print("\n--- Data Integrity Check ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        n_nans = df[col].isna().sum()
        n_infs = np.isinf(df[col]).sum()
        if n_nans > 0 or n_infs > 0:
            print(f"[{col}] NaNs: {n_nans}, Infs: {n_infs}")
    
    # 2. Biological Plausibility
    print("\n--- Biological Plausibility ---")
    if 'HR_Mean' in df.columns:
        print(f"HR Range: {df['HR_Mean'].min():.1f} - {df['HR_Mean'].max():.1f} BPM")
        if df['HR_Mean'].max() > 600 or df['HR_Mean'].min() < 20:
             print("WARNING: Extreme Heart Rates detected!")
    
    # 3. P-Value Verification (Manual Check)
    print("\n--- P-Value Verification (Sample) ---")
    if 'is_af' in df.columns and 'pRR_3.25' in df.columns:
        af_group = df[df['is_af'] == True]['pRR_3.25'].dropna()
        non_af_group = df[df['is_af'] == False]['pRR_3.25'].dropna()
        
        t_stat, p_val = stats.ttest_ind(af_group, non_af_group, equal_var=False)
        print(f"Manual t-test for pRR_3.25: p={p_val:.5e}")
        
        # Check against saved file
        pval_path = os.path.join(base_dir, "af_vs_nonaf_pvalues.csv")
        if os.path.exists(pval_path):
            pdf = pd.read_csv(pval_path)
            row = pdf[pdf['Metric'] == 'pRR_3.25']
            if not row.empty:
                saved_p = row['P_Value'].values[0]
                print(f"Saved p-value:          p={saved_p:.5e}")
                if np.isclose(p_val, saved_p):
                    print("VERIFIED: Calculation matches.")
                else:
                    print("WARNING: Calculation mismatch!")

if __name__ == "__main__":
    verify_data()
