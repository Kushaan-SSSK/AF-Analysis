import pandas as pd
import neurokit2 as nk
import numpy as np
import os
from analyze_ekg import load_data, preprocess_ecg

def inspect_file(file_name):
    # Construct full path (assuming standard location from previous context)
    base_dir = r"c:\Users\kusha\Downloads\AF Analysis\EKG data"
    file_path = os.path.join(base_dir, file_name)
    
    print(f"Inspecting {file_name}...")
    df = load_data(file_path)
    if df is None: return

    t0 = df['Time'].iloc[0]
    t1 = df['Time'].iloc[1]
    fs = int(round(1 / (t1 - t0)))
    print(f"Sampling Rate: {fs} Hz")
    
    signal = df['Raw Ch 1'].values
    clean_signal = preprocess_ecg(signal, sampling_rate=fs)
    
    print(f"Signal Length: {len(clean_signal)} samples ({len(clean_signal)/fs:.2f} s)")
    print(f"Min: {np.min(clean_signal):.4f}")
    print(f"Max: {np.max(clean_signal):.4f}")
    print(f"Mean: {np.mean(clean_signal):.4f}")
    
    # Percentiles
    p1, p5, p95, p99 = np.percentile(clean_signal, [1, 5, 95, 99])
    print(f"1st Percentile: {p1:.4f}")
    print(f"5th Percentile: {p5:.4f}")
    print(f"95th Percentile: {p95:.4f}")
    print(f"99th Percentile: {p99:.4f}")
    
    # Check first 5 seconds vs rest
    cutoff = int(5 * fs)
    if len(clean_signal) > cutoff:
        first_part = clean_signal[:cutoff]
        rest = clean_signal[cutoff:]
        print(f"First 5s Range: {np.min(first_part):.4f} to {np.max(first_part):.4f}")
        print(f"Rest Range: {np.min(rest):.4f} to {np.max(rest):.4f}")

inspect_file("132282-3-1in4-a_Export.xls")
inspect_file("132283-4-1in4-a-s_Export.xls")
