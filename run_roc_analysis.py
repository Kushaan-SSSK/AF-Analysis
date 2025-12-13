import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import glob
from analyze_ekg import load_data, preprocess_ecg, detect_peaks, calculate_prr_metrics

def analyze_dataset_for_roc(data_dir, output_dir):
    files = glob.glob(os.path.join(data_dir, "*_Export.xls"))
    results = []
    
    print(f"Analyzing {len(files)} files for ROC...")
    
    for f in files:
        try:
            df = load_data(f)
            if df is None: continue
            
            t0 = df['Time'].iloc[0]
            t1 = df['Time'].iloc[1]
            fs = int(round(1 / (t1 - t0)))
            signal = df['Raw Ch 1'].values
            clean_signal = preprocess_ecg(signal, sampling_rate=fs)
            peaks = detect_peaks(clean_signal, sampling_rate=fs)
            
            window_length_sec = 10
            num_windows = int((len(signal)/fs) // window_length_sec)
            
            for i in range(num_windows):
                start = int(i * window_length_sec * fs)
                end = int((i+1) * window_length_sec * fs)
                window_peaks = peaks[(peaks >= start) & (peaks < end)]
                
                metrics = calculate_prr_metrics(window_peaks, sampling_rate=fs, threshold_pct=3.25)
                if metrics:
                    results.append({
                        'file': os.path.basename(f),
                        'score': metrics['pRR_3.25'],
                    })
                    
        except Exception as e:
            print(f"Error {f}: {e}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(output_dir, "roc_data.csv"), index=False)
    print("Saved ROC data. Once ground truth labels are added, use this to plot ROC.")

def plot_roc(csv_path):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        print("Error: 'label' column missing. Please annotate the CSV with ground truth.")
        return

    fpr, tpr, thresholds = roc_curve(df['label'], df['score'])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(csv_path.replace('.csv', '_roc.png'))
    print(f"Saved ROC plot to {csv_path.replace('.csv', '_roc.png')}")

if __name__ == "__main__":
    pass
