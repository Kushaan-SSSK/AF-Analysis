import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import os
from analyze_ekg import load_data, preprocess_ecg, detect_peaks

def visualize_file(file_path, output_dir):
    print(f"Visualizing {os.path.basename(file_path)}...")
    df = load_data(file_path)
    if df is None: return

    t0 = df['Time'].iloc[0]
    t1 = df['Time'].iloc[1]
    fs = int(round(1 / (t1 - t0)))
    
    signal = df['Raw Ch 1'].values
    clean_signal = preprocess_ecg(signal, sampling_rate=fs)
    
    # Trim first 2 seconds to remove filter artifacts/transients before detection
    trim_sec = 2
    trim_idx = int(trim_sec * fs)
    if len(clean_signal) > trim_idx:
        clean_signal = clean_signal[trim_idx:]
    else:
        print("Signal too short after trim.")
        return

    peaks = detect_peaks(clean_signal, sampling_rate=fs)
    
    if len(peaks) < 10:
        print("Not enough peaks to visualize.")
        return
        
    # Calculate BPM for title
    rr_intervals = np.diff(peaks) / fs
    bpm = 60 / np.mean(rr_intervals)

    plt.figure(figsize=(15, 5))
    
    # Plot entire trimmed signal
    t_axis = np.arange(len(clean_signal)) / fs
    plt.plot(t_axis, clean_signal, label='Cleaned EKG (Trimmed)', linewidth=0.5)
    
    plt.plot(peaks/fs, clean_signal[peaks], 'r.', markersize=3, label=f'Peaks (BPM: {bpm:.1f})')
    
    # Robust scaling on the already trimmed signal
    y_min, y_max = np.percentile(clean_signal, [0.5, 99.5])
    y_range = y_max - y_min
    if y_range == 0: y_range = 1
    plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    plt.title(f"EKG Trace: {os.path.basename(file_path)}")
    plt.xlabel("Time (s) from trim point")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    out_trace = os.path.join(output_dir, f"trace_{os.path.basename(file_path)}.png")
    plt.savefig(out_trace)
    plt.close()
    
    results_file = os.path.join(output_dir, os.path.basename(file_path).replace('.xls', '_results.csv'))
    if os.path.exists(results_file):
        res_df = pd.read_csv(results_file)
        
        plt.figure(figsize=(15, 5))
        plt.plot(res_df['start_time'], res_df['pRR_3.25'], label='pRR 3.25%', marker='o')
        plt.axhline(y=75.32, color='r', linestyle='--', label='AF Threshold (75.32%)')
        
        af_windows = res_df[res_df['is_af']]
        plt.scatter(af_windows['start_time'], af_windows['pRR_3.25'], color='red', s=50, zorder=5, label='Detected AF')
        
        plt.xlabel("Time (s)")
        plt.ylabel("pRR 3.25% (%)")
        plt.title(f"AF Severity over Time: {os.path.basename(file_path)}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        
        out_metric = os.path.join(output_dir, f"metric_{os.path.basename(file_path)}.png")
        plt.savefig(out_metric)
        plt.close()
    else:
        print(f"No results file found for {os.path.basename(file_path)}, skipping metric plot.")
    
    print(f"Saved plots to {output_dir}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "EKG data")
    output_dir = os.path.join(base_dir, "Verification_Plots")
    os.makedirs(output_dir, exist_ok=True)
    
    import glob
    files = glob.glob(os.path.join(data_dir, "*_Export.xls"))
    
    for f in files:
        visualize_file(f, output_dir)

if __name__ == "__main__":
    main()
