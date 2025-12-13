import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(base_dir, "Results", "all_results_summary.csv")
    
    if not os.path.exists(results_file):
        print("Results file not found.")
        return

    df = pd.read_csv(results_file)

    summary = df.groupby('file').agg(
        total_windows=('window_idx', 'count'),
        valid_windows=('peak_count', lambda x: (x > 5).sum()), 
        total_peaks=('peak_count', 'sum'),
        mean_hr_bpm=('peak_count', lambda x: x.mean() * 6), 
        af_burden=('is_af', 'mean'),
        mean_pRR3_25=('pRR_3.25', 'mean')
    ).reset_index()

    summary['af_burden'] = summary['af_burden'] * 100
    
    summary_out_path = os.path.join(base_dir, "Results", "final_summary_report.csv")
    summary.to_csv(summary_out_path, index=False)
    print(f"Summary report saved to: {summary_out_path}")
    print(summary.to_string())

if __name__ == "__main__":
    main()
