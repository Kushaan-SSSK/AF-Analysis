import pandas as pd
import os
import glob
from sklearn.metrics import confusion_matrix, classification_report
try:
    from analyze_ekg import analyze_file
except ImportError:
    # Handle direct execution vs module import
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from analyze_ekg import analyze_file

def verify_ground_truth():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "GroundtruthverifiedEKGData")
    results_dir = os.path.join(base_dir, "Results_Verification")
    os.makedirs(results_dir, exist_ok=True)
    
    annot_path = os.path.join(data_dir, "EKG_Annotations.xlsx")
    if not os.path.exists(annot_path):
        print(f"Error: Annotation file not found at {annot_path}")
        return

    print(f"Loading annotations from {annot_path}...")
    df_annot = pd.read_excel(annot_path)
    # Rename columns for consistency
    df_annot.columns = ['Filename', 'AF_Label', 'RR_irregular', 'P_wave_loss', 'Other']
    
    # Filter out NaNs in AF_Label
    df_annot = df_annot.dropna(subset=['AF_Label'])
    # Normalize labels: Yes -> 1, No -> 0
    df_annot['AF_Label'] = df_annot['AF_Label'].astype(str).str.strip().str.lower()
    df_annot['Ground_Truth'] = df_annot['AF_Label'].map({'yes': 1, 'no': 0})
    
    # Drop rows where label couldn't be mapped (e.g. 'unsure' if any)
    df_annot = df_annot.dropna(subset=['Ground_Truth'])
    
    y_true = []
    y_pred = []
    results_detail = []
    
    print(f"Processing {len(df_annot)} annotated records...")
    
    for idx, row in df_annot.iterrows():
        # Parse basename from "Folder/Basename.iwxdata"
        full_str = str(row['Filename'])
        basename = os.path.basename(full_str).replace('.iwxdata', '')
        
        # Look for matching _Export.xls (or .txt) in data_dir
        # Try exact match first, then wildcards
        candidates = glob.glob(os.path.join(data_dir, f"{basename}*_Export.xls"))
        if not candidates:
             candidates = glob.glob(os.path.join(data_dir, f"{basename}*_Export.txt"))
             
        if not candidates:
            print(f"Warning: No file found for {basename}")
            continue
            
        file_path = candidates[0] # Take first match
        
        # Run analysis
        try:
            # We assume analyze_file returns a DataFrame of results
            res_df = analyze_file(file_path, results_dir)
            if res_df is None or res_df.empty:
                print(f"  Analysis failed/empty for {basename}")
                continue
                
            # Check for AF in seconds 10-40
            # analyze_ekg uses 10s windows starting at 0.
            # Window 0: 0-10 -> Exclude
            # Window 1: 10-20 \
            # Window 2: 20-30  >  These are the target
            # Window 3: 30-40 /
            
            target_windows = [1, 2, 3]
            # Filter results for these windows
            relevant_res = res_df[res_df['window_idx'].isin(target_windows)]
            
            # If ANY window is AF, we classify as AF (High sensitivity approach)
            # Or we could say Majority vote. Let's stick to "Any AF detected" for now as AF is paroxysmal.
            is_af_detected = relevant_res['is_af'].any()
            pred_label = 1 if is_af_detected else 0
            
            y_true.append(row['Ground_Truth'])
            y_pred.append(pred_label)
            
            results_detail.append({
                'Filename': basename,
                'Ground_Truth': 'Yes' if row['Ground_Truth'] == 1 else 'No',
                'Predicted': 'Yes' if pred_label == 1 else 'No',
                'Correct': row['Ground_Truth'] == pred_label,
                'Windows_AF_Found': relevant_res[relevant_res['is_af']]['window_idx'].tolist()
            })
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            
    # Generate Report
    if not y_true:
        print("No valid comparisons made.")
        return

    print("\n" + "="*40)
    print("VERIFICATION RESULTS (10s-40s Window)")
    print("="*40)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Total Reviewed: {len(y_true)}")
    print(f"True Positives (AF correctly ID'd): {tp}")
    print(f"True Negatives (Normal correctly ID'd): {tn}")
    print(f"False Positives (Normal flagged as AF): {fp}")
    print(f"False Negatives (AF missed): {fn}")
    
    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nAccuracy:    {accuracy:.2%}")
    print(f"Sensitivity: {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")
    
    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'AF']))
    
    # Save detailed CSV
    detail_df = pd.DataFrame(results_detail)
    detail_path = os.path.join(results_dir, "verification_detail_10_40s.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"\nDetailed report saved to: {detail_path}")

if __name__ == "__main__":
    verify_ground_truth()
