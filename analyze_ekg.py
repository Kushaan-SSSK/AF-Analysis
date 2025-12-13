import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        if 'Raw Ch 1' not in df.columns:
            print(f"Warning: 'Raw Ch 1' not found in {file_path}. Columns: {df.columns}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_ecg(ecg_signal, sampling_rate=1000):
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="biosppy")
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

def calculate_poincare_metrics(r_peaks, sampling_rate=1000):
    if len(r_peaks) < 3:
        return None
        
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000 
    
    try:
        hrv_nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate)
        sd1 = hrv_nonlinear['HRV_SD1'].values[0]
        sd2 = hrv_nonlinear['HRV_SD2'].values[0]
    except:
        sd1, sd2 = 0, 0
        
    return {'sd1': sd1, 'sd2': sd2, 'rr_mean': np.mean(rr_intervals), 'rr_cv': np.std(rr_intervals)/np.mean(rr_intervals)}

def calculate_prr_metrics(r_peaks, sampling_rate=1000, threshold_pct=3.25):
    if len(r_peaks) < 3:
        return None
        
    rr_intervals = np.diff(r_peaks)
    
    rr_diffs = np.abs(np.diff(rr_intervals))
    rr_prev = rr_intervals[:-1]
    
    relative_diffs = (rr_diffs / rr_prev) * 100
    
    count_above = np.sum(relative_diffs >= threshold_pct)
    total_intervals = len(relative_diffs)
    
    prr_val = (count_above / total_intervals) * 100 if total_intervals > 0 else 0
    
    return {
        f'pRR_{threshold_pct}': prr_val,
        'rr_mean_ms': np.mean(rr_intervals) / sampling_rate * 1000
    }

def analyze_file(file_path, output_dir, window_length_sec=10):
    print(f"Processing {os.path.basename(file_path)}...")
    df = load_data(file_path)
    if df is None: return
    
    t0 = df['Time'].iloc[0]
    t1 = df['Time'].iloc[1]
    fs = int(round(1 / (t1 - t0)))
    print(f"  FS: {fs} Hz")
    
    signal = df['Raw Ch 1'].values
    
    clean_signal = preprocess_ecg(signal, sampling_rate=fs)
    
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
        
        metrics = calculate_prr_metrics(window_peaks, sampling_rate=fs, threshold_pct=3.25)
        
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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "EKG data")
    output_dir = os.path.join(base_dir, "Results")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(data_dir, "*_Export.xls"))
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
        print(f"\nBatch processing complete. Summary saved to:\n{summary_path}")

if __name__ == "__main__":
    main()
