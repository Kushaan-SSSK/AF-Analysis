"""
Run the full AF analysis pipeline on EKG data exports.
Overrides the default data directory in analyze_ekg.py to point at 'EKG data/'
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Analysis_Scripts'))
import analyze_ekg

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "EKG Recordings")
output_dir = os.path.join(base_dir, "Results")
os.makedirs(output_dir, exist_ok=True)

print(f"Running analysis on: {data_dir}")
print(f"Output dir: {output_dir}\n")

final_df = analyze_ekg.analyze_directory(data_dir, output_dir)

if not final_df.empty:
    from scipy import stats
    import numpy as np, pandas as pd
    print(f"\nTotal windows: {len(final_df)}")
    if 'is_af' in final_df.columns:
        n_af = final_df['is_af'].sum()
        n_total = len(final_df)
        print(f"Rule-based AF windows: {n_af}/{n_total} ({n_af/n_total*100:.1f}%)")

    # P-value analysis
    print("\nCalculating P-values (AF vs Non-AF)...")
    af_group = final_df[final_df['is_af'] == True]
    non_af_group = final_df[final_df['is_af'] == False]

    exclude_cols = ['file', 'window_idx', 'start_time', 'end_time', 'is_af', 'note', 'peaks']
    metric_cols = [c for c in final_df.columns if c not in exclude_cols and
                   pd.api.types.is_numeric_dtype(final_df[c])]

    p_val_results = []
    for col in metric_cols:
        af_vals = af_group[col].dropna()
        non_af_vals = non_af_group[col].dropna()
        if len(af_vals) > 1 and len(non_af_vals) > 1:
            try:
                t_stat, p_val = stats.ttest_ind(af_vals, non_af_vals, equal_var=False)
                p_val_results.append({
                    'Metric': col, 'AF_Mean': np.mean(af_vals), 'AF_Std': np.std(af_vals, ddof=1),
                    'NonAF_Mean': np.mean(non_af_vals), 'NonAF_Std': np.std(non_af_vals, ddof=1),
                    'P_Value': p_val, 'Significant': p_val < 0.05
                })
            except Exception:
                pass

    if p_val_results:
        pval_df = pd.DataFrame(p_val_results).sort_values('P_Value')
        pval_path = os.path.join(output_dir, "af_vs_nonaf_pvalues.csv")
        pval_df.to_csv(pval_path, index=False)
        print(f"\nTop significant features:")
        print(pval_df[pval_df['Significant']].head(10)[['Metric','AF_Mean','NonAF_Mean','P_Value']].to_string(index=False))
        print(f"\nP-values saved to: {pval_path}")
else:
    print("No results generated.")
