import pandas as pd
import numpy as np
import os
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from train import load_ground_truth





def run_benchmark():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, "Results")
    file_path   = os.path.join(results_dir, "all_results_summary.csv")



    if not os.path.exists(file_path):
        print(f"Missing {file_path} - run extract_features.py first.")
        return



    df = pd.read_csv(file_path)
    gt_dir = os.path.join(base_dir, "GroundtruthverifiedEKGData")
    gt_map = load_ground_truth(gt_dir)
    if gt_map is None:

        return



    def normalize(name):
        base = os.path.splitext(os.path.basename(name))[0]
        return base.replace('_Export', '').replace('.iwxdata', '').lower().strip()


    norm_gt_map = {normalize(k): v for k, v in gt_map.items()}


    def get_label(row):
        norm_row = normalize(row['file'])
        ann = norm_gt_map.get(norm_row)

        if ann is None:
            for k, v in norm_gt_map.items():
                if norm_row in k or k in norm_row:
                    ann = v
                    break

        if not ann:
            return None

        w_start, w_end = row['start_time'], row['end_time']
        is_af, af_type, episodes = ann['is_af'], ann['af_type'], ann.get('episodes', [])
        if not is_af: return False
        if af_type == 'sustained': return True
        if af_type == 'paroxysmal':

            if not episodes: return True
            for ep_s, ep_e in episodes:
                if (w_start < ep_e) and (w_end > ep_s):
                    return True

            return False
        return None



    df['label'] = df.apply(get_label, axis=1)
    df_clean = df.dropna(subset=['label']).copy()



    if len(df_clean) < 10:
        print("Not enough annotated data.")
        return



    exclude_cols = ['file', 'window_idx', 'start_time', 'end_time', 'is_af',

                    'note', 'peaks', 'label', 'af_type', 'annotation', 'manual_label']

    feature_cols = [c for c in df_clean.columns

                    if c not in exclude_cols

                    and pd.api.types.is_numeric_dtype(df_clean[c])

                    and df_clean[c].notna().sum() > 0]



    X      = df_clean[feature_cols]
    y      = df_clean['label'].astype(int)
    groups = df_clean['file']



    imputer  = SimpleImputer(strategy='median')
    X_imp    = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_cols)



    # GroupShuffleSplit keeps all windows from a file on the same side of the split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_scaled, y, groups))



    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Train: {len(X_train)} (AF+: {y_train.sum()})  "
          f"Test: {len(X_test)} (AF+: {y_test.sum()})")



    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, _ = clf.fit(X_train, X_test, y_train, y_test)



    out_path = os.path.join(results_dir, "lazypredict_benchmark.csv")
    models.to_csv(out_path)
    print("\n", models)
    print(f"\nBenchmark saved -> {out_path}")





if __name__ == "__main__":
    run_benchmark()

