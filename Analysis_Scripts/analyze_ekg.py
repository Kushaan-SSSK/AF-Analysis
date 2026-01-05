import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats, signal as scipy_signal
import warnings
import os
import glob

def load_data(file_path):
    try:
        # Load with low_memory=False to suppress DtypeWarning for mixed types (e.g. units row)
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        
        if 'Raw Ch 1' not in df.columns:
            print(f"Warning: 'Raw Ch 1' not found in {file_path}. Columns: {df.columns}")
            return None
            
        # Ensure Time and Signal are numeric, coercing errors (like units 's', 'mV') to NaN
        if 'Time' in df.columns:
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        
        for col in ['Raw Ch 1', 'Raw Ch 2']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values in time or strictly required Ch 1
        df = df.dropna(subset=['Time', 'Raw Ch 1'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        if df.empty or len(df) < 2:
             print(f"Warning: Insufficient valid numeric data in {file_path}")
             return None

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def downsample_data(df, target_fs=1000):
    if df is None or len(df) < 2: return df
    
    t0 = df['Time'].iloc[0]
    t1 = df['Time'].iloc[1]
    current_fs = 1 / (t1 - t0)
    
    if current_fs > (target_fs * 1.5): # Only downsample if significantly higher
        factor = int(round(current_fs / target_fs))
        if factor > 1:
            # Simple decimation (pick every Nth sample)
            # This is fast and generally okay for EKG oversampling
            # Ideally we might lowpass first to avoid aliasing, but bio-signals are usually bandlimited anyway
            df = df.iloc[::factor].reset_index(drop=True)
            # print(f"    Downsampled from {int(current_fs)}Hz to {int(current_fs/factor)}Hz (Factor: {factor})")
            
    return df

def preprocess_ecg(ecg_signal, sampling_rate=1000):
    # 1. 3Hz High-Pass Filter (LabScribe equivalent)
    # Using a 2nd order Butterworth filter
    b, a = scipy_signal.butter(2, 3, 'highpass', fs=sampling_rate)
    filtered_ecg = scipy_signal.filtfilt(b, a, ecg_signal)
    
    # 2. NeuroKit Cleaning (Biosppy method)
    cleaned_ecg = nk.ecg_clean(filtered_ecg, sampling_rate=sampling_rate, method="biosppy")
    
    return cleaned_ecg

def adaptive_threshold_peaks(signal, sampling_rate=1000, window_sec=0.1, upshift_pct=0.035):
    window_size = int(window_sec * sampling_rate)
    
    mavg = pd.Series(signal).rolling(window=window_size, center=True).mean().fillna(0).values
    
    signal_range = np.max(signal) - np.min(signal)
    threshold = mavg + (upshift_pct * signal_range)
    
    above_threshold = signal > threshold
    
    peaks = []
    
    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(signal))
        
    if len(starts) > len(ends):
        starts = starts[:len(ends)]
    elif len(ends) > len(starts):
        ends = ends[:len(starts)]
        
    for start, end in zip(starts, ends):
        segment = signal[start:end]
        if len(segment) == 0: continue
        peak_idx = start + np.argmax(segment)
        peaks.append(peak_idx)
        
    return np.array(peaks)

def detect_peaks(cleaned_ecg, sampling_rate=1000):
    candidates = []
    
    for inverted in [False, True]:
        sig = cleaned_ecg * -1 if inverted else cleaned_ecg
        
        try:
            _, info = nk.ecg_peaks(sig, sampling_rate=sampling_rate, method="neurokit")
            peaks = info["ECG_R_Peaks"]
            if len(peaks) > 10:
                candidates.append({'peaks': peaks, 'method': 'neurokit', 'inverted': inverted})
        except:
            pass
            
        try:
            p5, p95 = np.percentile(sig, [5, 95])
            amp = p95 - p5
            min_prominence = amp * 0.15
            
            peaks, _ = find_peaks(sig, distance=int(sampling_rate*0.05), prominence=min_prominence)
            if len(peaks) > 10:
                candidates.append({'peaks': peaks, 'method': 'prominence', 'inverted': inverted})
        except:
            pass
        
        try:
            peaks = adaptive_threshold_peaks(sig, sampling_rate=sampling_rate)
            if len(peaks) > 10:
                candidates.append({'peaks': peaks, 'method': 'adaptive', 'inverted': inverted})
        except:
            pass

    best_peaks = np.array([])
    best_score = float('inf')
    best_meta = None
    
    for cand in candidates:
        peaks = cand['peaks']
        if len(peaks) < 2: continue
        
        rr_sec = np.diff(peaks) / sampling_rate
        bpm = 60 / np.mean(rr_sec)
        
        if 300 < bpm < 900:
            cv = np.std(rr_sec) / np.mean(rr_sec)
            
            if cv < best_score:
                best_score = cv
                best_peaks = peaks
                best_meta = cand
                
    if len(best_peaks) > 0:
        print(f"    Selected: {best_meta['method']} (Inv: {best_meta['inverted']}), BPM: {60/np.mean(np.diff(best_peaks)/sampling_rate):.1f}, CV: {best_score:.3f}")
        return best_peaks
    else:
        print("    No valid peaks found.")
        return np.array([])

def calculate_full_metrics(r_peaks, sampling_rate=1000, threshold_pct=3.25):
    if len(r_peaks) < 3:
        return None
        
    metrics = {}
    
    # 1. pRR Metrics (Existing)
    rr_intervals = np.diff(r_peaks) # in samples
    rr_ms = rr_intervals / sampling_rate * 1000
    
    rr_diffs = np.abs(np.diff(rr_intervals))
    rr_prev = rr_intervals[:-1]
    
    # Avoid division by zero
    valid_rr = rr_prev > 0
    if np.sum(valid_rr) == 0:
        return None
        
    relative_diffs = (rr_diffs[valid_rr] / rr_prev[valid_rr]) * 100
    
    count_above = np.sum(relative_diffs >= threshold_pct)
    total_intervals = len(relative_diffs)
    
    metrics[f'pRR_{threshold_pct}'] = (count_above / total_intervals) * 100 if total_intervals > 0 else 0
    
    # 2. Basic Time Domain (Paper)
    metrics['RR_Mean'] = np.mean(rr_ms)
    metrics['RR_Min'] = np.min(rr_ms)
    metrics['RR_Max'] = np.max(rr_ms)
    
    hr_values = 60000 / rr_ms
    metrics['HR_Mean'] = np.mean(hr_values)
    metrics['HR_Min'] = np.min(hr_values)
    metrics['HR_Max'] = np.max(hr_values)
    metrics['HR_Std'] = np.std(hr_values, ddof=1) # SDHR
    
    metrics['SDNN'] = np.std(rr_ms, ddof=1)
    
    diff_rr_ms = np.abs(np.diff(rr_ms))
    metrics['RMSSD'] = np.sqrt(np.mean(diff_rr_ms**2))
    metrics['SDSD'] = np.std(np.diff(rr_ms), ddof=1)
    
    # NNx and pNNx
    for x in [20, 50, 100]:
        nn_x = np.sum(diff_rr_ms > x)
        pnn_x = (nn_x / len(diff_rr_ms)) * 100 if len(diff_rr_ms) > 0 else 0
        metrics[f'NN{x}'] = nn_x
        metrics[f'pNN{x}'] = pnn_x

    # 3. Comprehensive HRV using NeuroKit2
    # We supress warnings because short windows often cause VLF warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # nk.hrv expects peaks indices
            hrv_indices = nk.hrv(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Extract relevant columns
            # Frequency (Welch is default in nk.hrv)
            for band in ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']:
                if band in hrv_indices.columns:
                    metrics[band.replace('HRV_', '')] = hrv_indices[band].values[0]
            
            # Nonlinear
            for nl in ['HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_CSI', 'HRV_CVI', 'HRV_Modified_CSI']:
                if nl in hrv_indices.columns:
                    metrics[nl.replace('HRV_', '')] = hrv_indices[nl].values[0]
            
            if 'SD1' in metrics and 'SD2' in metrics:
                metrics['Area'] = np.pi * metrics['SD1'] * metrics['SD2']
                    
        except Exception as e:
            # print(f"HRV calculation failed: {e}")
            pass
            
    return metrics

def analyze_file(file_path, output_dir, window_length_sec=10):
    print(f"Processing {os.path.basename(file_path)}...")
    df = load_data(file_path)
    if df is None: return
    
    # Downsample if needed (optimization for 5000Hz data)
    df = downsample_data(df, target_fs=1000)
    
    t0 = df['Time'].iloc[0]
    t1 = df['Time'].iloc[1]
    fs = int(round(1 / (t1 - t0)))
    print(f"  FS: {fs} Hz")
    
    signal = df['Raw Ch 1'].values
    
    clean_signal = preprocess_ecg(signal, sampling_rate=fs)
    
    # Trim first 2 seconds to remove filter artifacts/transients
    trim_sec = 2
    trim_idx = int(trim_sec * fs)
    if len(clean_signal) > trim_idx:
        clean_signal = clean_signal[trim_idx:]
        signal = signal[trim_idx:] # Keep raw signal aligned if needed later
        
    peaks = detect_peaks(clean_signal, sampling_rate=fs)
    print(f"  Peaks: {len(peaks)}")
    
    total_duration_sec = len(signal) / fs
    num_windows = int(total_duration_sec // window_length_sec)
    
    results = []
    af_threshold = 75.32 
    
    for i in range(num_windows):
        start_time = i * window_length_sec
        end_time = (i + 1) * window_length_sec
        
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        window_peaks = peaks[(peaks >= start_sample) & (peaks < end_sample)]
        
        metrics = calculate_full_metrics(window_peaks, sampling_rate=fs, threshold_pct=3.25)
        
        row = {
            'file': os.path.basename(file_path),
            'window_idx': i,
            'start_time': start_time,
            'end_time': end_time,
            'peak_count': len(window_peaks)
        }
        
        if metrics:
            row.update(metrics)
            row['is_af'] = metrics['pRR_3.25'] > af_threshold
        else:
            row['is_af'] = False
            row['pRR_3.25'] = 0
            row['note'] = 'Insufficient peaks'
            
        results.append(row)
        
    res_df = pd.DataFrame(results)
    out_name = os.path.basename(file_path).replace('.xls', '_results.csv')
    res_df.to_csv(os.path.join(output_dir, out_name), index=False)
    
    af_windows = res_df['is_af'].sum()
    print(f"  AF Windows: {af_windows}/{num_windows} ({af_windows/num_windows*100:.1f}%)")
    return res_df

def analyze_directory(data_dir, output_dir, file_pattern="*_Export.xls*"):
    # Supports .xls and .txt matching via wildcards like *.xls* or specific patterns
    files = glob.glob(os.path.join(data_dir, file_pattern))
    # Also look for .txt if pattern implies generic export
    if file_pattern == "*_Export.xls*":
         files += glob.glob(os.path.join(data_dir, "*_Export.txt"))
    
    files = sorted(list(set(files))) # Remove duplicates if any
    print(f"Found {len(files)} files in {data_dir}")
    
    all_results = []
    
    for f in files:
        try:
            res = analyze_file(f, output_dir)
            if res is not None:
                all_results.append(res)
        except Exception as e:
            print(f"Failed to process {f}: {e}")
            
    if all_results:
        final_df = pd.concat(all_results)
        summary_path = os.path.join(output_dir, "all_results_summary.csv")
        final_df.to_csv(summary_path, index=False)
        return final_df
    return pd.DataFrame()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    # data_dir = os.path.join(base_dir, "EKG data") 
    # Use verified data for classifier training
    data_dir = os.path.join(base_dir, "GroundtruthverifiedEKGData")
    output_dir = os.path.join(base_dir, "Results")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    final_df = analyze_directory(data_dir, output_dir)
    
    if not final_df.empty:
        # --- P-Value Calculation ---
        print("\nCalculating P-values (AF vs Non-AF)...")
        if 'is_af' in final_df.columns:
            af_group = final_df[final_df['is_af'] == True]
            non_af_group = final_df[final_df['is_af'] == False]
            
            p_val_results = []
            
            # Identify metric columns (numeric, excluding metadata)
            exclude_cols = ['file', 'window_idx', 'start_time', 'end_time', 'is_af', 'note', 'peaks']
            metric_cols = [c for c in final_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(final_df[c])]
            
            for col in metric_cols:
                af_vals = af_group[col].dropna()
                non_af_vals = non_af_group[col].dropna()
                
                if len(af_vals) > 1 and len(non_af_vals) > 1:
                    try:
                        t_stat, p_val = stats.ttest_ind(af_vals, non_af_vals, equal_var=False)
                        p_val_results.append({
                            'Metric': col,
                            'AF_Mean': np.mean(af_vals),
                            'AF_Std': np.std(af_vals, ddof=1),
                            'NonAF_Mean': np.mean(non_af_vals),
                            'NonAF_Std': np.std(non_af_vals, ddof=1),
                            'P_Value': p_val,
                            'Significant': p_val < 0.05
                        })
                    except Exception as e:
                        pass
                        
            if p_val_results:
                pval_df = pd.DataFrame(p_val_results)
                # Sort by P-value
                pval_df = pval_df.sort_values(by='P_Value')
                pval_path = os.path.join(output_dir, "af_vs_nonaf_pvalues.csv")
                pval_df.to_csv(pval_path, index=False)
                print(f"P-value analysis saved to:\n{pval_path}")
            else:
                print("Not enough data for P-value calculation.")
        else:
             print("Column 'is_af' not found in results.")

if __name__ == "__main__":
    main()
