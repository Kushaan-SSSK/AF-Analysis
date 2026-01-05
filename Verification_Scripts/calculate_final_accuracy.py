import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def calc():
    base_dir = r"c:\Users\kusha\Downloads\AF Analysis\Results"
    file_path = os.path.join(base_dir, "all_results_summary.csv")
    
    if not os.path.exists(file_path):
        print("Data not found")
        return

    df = pd.read_csv(file_path)
    exclude_cols = ['file', 'window_idx', 'start_time', 'end_time', 'is_af', 'note', 'peaks']
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]
    
    X = df[feature_cols]
    y = df['is_af'].astype(int)
    groups = df['file']
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv = LeaveOneGroupOut()
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(rf, X, y, groups=groups, cv=cv, scoring=scoring)
    
    print(f"ACCURACY:{scores['test_accuracy'].mean():.4f}")
    print(f"SENSITIVITY:{scores['test_recall'].mean():.4f}")
    print(f"PRECISION:{scores['test_precision'].mean():.4f}")
    print(f"F1:{scores['test_f1'].mean():.4f}")

if __name__ == "__main__":
    calc()
