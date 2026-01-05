import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_ground_truth(data_dir):
    """
    Attempts to load ground truth annotations from Excel files in the data directory.
    Returns a dictionary mapping {filename: is_af (bool)} or None.
    """
    # Look for likely annotation files
    potential_files = glob.glob(os.path.join(data_dir, "*Annotation*.xlsx")) + \
                      glob.glob(os.path.join(data_dir, "*ground_truth*.xlsx"))
    
    if not potential_files:
        # Fallback to checking parent directories or specific known paths
        parent_dir = os.path.dirname(data_dir)
        potential_files = glob.glob(os.path.join(parent_dir, "GroundtruthverifiedEKGData", "*Annotation*.xlsx"))

    if not potential_files:
        print("    No annotation file found.")
        return None
        
    annot_path = potential_files[0]
    print(f"    Loading annotations from: {annot_path}")
    
    try:
        df = pd.read_excel(annot_path)
        # Normalize columns - assume standard format or try to guess
        # Expected: Filename, AF_Label (Yes/No)
        
        # rename columns to standard if they match patterns
        df.columns = [c.strip() for c in df.columns]
        
        # Find column containing "AF" or "Label"
        label_col = next((c for c in df.columns if 'AF' in c and 'Label' in c), None)
        if not label_col:
            label_col = next((c for c in df.columns if 'AF' in c), None)
            
        file_col = next((c for c in df.columns if 'File' in c), None)
        
        if not label_col or not file_col:
            print(f"    Could not identify Filename/Label columns in {annot_path}")
            return None
            
        # Create mapping
        gt_map = {}
        for idx, row in df.iterrows():
            fname = str(row[file_col]).strip()
            # Handle potential extensions in annotation vs results
            # We'll store the basename without extension for matching
            base_fname = os.path.splitext(os.path.basename(fname))[0]
            
            label_str = str(row[label_col]).lower().strip()
            is_af = 'yes' in label_str or '1' in label_str or 'true' in label_str
            
            gt_map[base_fname] = is_af
            
        print(f"    Loaded {len(gt_map)} annotations.")
        return gt_map
        
    except Exception as e:
        print(f"    Error reading annotations: {e}")
        return None

def train_rf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, "Results")
    file_path = os.path.join(results_dir, "all_results_summary.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print("Loading analysis results...")
    df = pd.read_csv(file_path)
    
    print("Loading Ground Truth Annotations...")
    # Try looking in specific known directory first
    gt_dir = os.path.join(base_dir, "GroundtruthverifiedEKGData")
    gt_map = load_ground_truth(gt_dir)
    
    if gt_map is None:
        print("\nCRITICAL WARNING: No Ground Truth Annotations found!")
        print("Cannot train classifier without valid labels.")
        print("Please ensure 'EKG_Annotations.xlsx' exists in 'GroundtruthverifiedEKGData' or similar.")
        return

    # Map Ground Truth to Results
    # Match based on partial filename match since extensions might differ (.xls vs .iwxdata)
    def normalize_name(name):
        base = os.path.basename(name)
        # Remove extension
        base = os.path.splitext(base)[0]
        # Remove common suffixes from results/exports
        base = base.replace('_Export', '').replace('.iwxdata', '')
        return base.lower().strip()

    # Pre-process GT map keys
    # The dictionary keys are already basenames from load_ground_truth, but let's double check
    # load_ground_truth strips headers but maybe not extensions perfectly if they vary
    norm_gt_map = {normalize_name(k): v for k, v in gt_map.items()}

    def get_label(row_file):
        norm_row = normalize_name(row_file)
        
        # Try exact match in normalized map
        if norm_row in norm_gt_map:
            return norm_gt_map[norm_row]
            
        # Try checking if one contains the other (fuzzy match)
        # This handles cases where one might have a prefix path
        for k, v in norm_gt_map.items():
            if norm_row in k or k in norm_row:
                 return v
        return None

    df['manual_label'] = df['file'].apply(get_label)
    
    # Debug information for user
    matched_count = df['manual_label'].notna().sum()
    print(f"    Matched {matched_count} files to annotations.")
    
    if matched_count == 0:
        print("\nDEBUG: Match failure diagnosis:")
        print("First 3 Result Files (Normalized):")
        for f in df['file'].head(3).tolist():
             print(f"  '{f}' -> '{normalize_name(f)}'")
        print("First 3 Annotation Keys (Normalized):")
        for k in list(norm_gt_map.keys())[:3]:
             print(f"  '{k}'")
    
    # Filter out data without annotations
    df_clean = df.dropna(subset=['manual_label']).copy()
    print(f"\nEntries with annotations: {len(df_clean)} / {len(df)}")
    
    if len(df_clean) < 10:
        print("Not enough annotated data to train (need at least 10 samples).")
        return

    # Define Target and Features
    target_col = 'manual_label'
    
    # Exclude non-feature columns
    # We ALSO exclude the rule-based metrics to avoid circularity if desired, 
    # but the user might want to see if pRR is still the best predictor.
    # For now, we exclude metadata.
    exclude_cols = ['file', 'window_idx', 'start_time', 'end_time', 'is_af', \
                    'note', 'peaks', 'manual_label']
    
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_clean[c])]
    
    # Filter out pure noise columns
    feature_cols = [c for c in feature_cols if df_clean[c].notna().sum() > 0]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(int) 
    groups = df_clean['file']
    
    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_cols)
    
    # Initializing Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Leave-One-Group-Out CV
    print(f"\nRunning Leave-One-Group-Out Cross-Validation (Groups: {df_clean['file'].nunique()})...")
    cv = LeaveOneGroupOut()
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    try:
        scores = cross_validate(rf, X_scaled, y, groups=groups, cv=cv, scoring=scoring)
        
        print("\n--- Validation Results (Leave-One-File-Out) ---")
        print(f"Accuracy:  {scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.4f}")
        print(f"Sensitivity (Recall): {scores['test_recall'].mean():.4f}")
        print(f"Precision: {scores['test_precision'].mean():.4f}")
        print(f"F1 Score:  {scores['test_f1'].mean():.4f}")
    except ValueError as ve:
        print(f"Validation failed (likely inconsistent classes in splits): {ve}")
        # Calculate training error instead
        rf.fit(X_scaled, y)
        y_pred = rf.predict(X_scaled)
        print("\n--- Training Set Performance (Overfit Warning!) ---")
        print(classification_report(y, y_pred))

    
    # Final Fit for Feature Importance
    rf.fit(X_scaled, y)
    
    importances = rf.feature_importances_
    feature_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Predictors for Manual AF Labels:")
    print(feature_imp.head(10))
    
    imp_path = os.path.join(results_dir, "rf_feature_importance.csv")
    feature_imp.to_csv(imp_path, index=False)
    
    plt.figure(figsize=(10, 8))
    top_features = feature_imp.head(20).sort_values(by='Importance', ascending=True)
    plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
    plt.xlabel('Importance')
    plt.title('Top 20 AF Predictors (vs Manual Annotation)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rf_feature_importance.png"))
    plt.close()
    
    print(f"\nFeature importance saved to {imp_path}")

if __name__ == "__main__":
    train_rf()
