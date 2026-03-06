import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_curve, auc)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

# Minimum overlap (seconds) between a window and a paroxysmal AF episode
# for that window to be labeled AF. Prevents boundary windows from being
# mislabeled when only a sliver of the episode falls within the window.
MIN_AF_OVERLAP_SEC = 5.0


def load_ground_truth(data_dir):
    parent_dir = os.path.dirname(data_dir)
    potential_files = (
        glob.glob(os.path.join(parent_dir, "EKG_Annotations.xlsx")) +
        glob.glob(os.path.join(parent_dir, "EKG_Annotations*.xlsx")) +
        glob.glob(os.path.join(data_dir, "*Annotation*.xlsx")) +
        glob.glob(os.path.join(parent_dir, "GroundtruthverifiedEKGData", "*Annotation*.xlsx"))
    )

    if not potential_files:
        print("    No annotation file found.")
        return None

    annot_path = potential_files[0]
    print(f"    Loading annotations from: {annot_path}")

    try:
        df = pd.read_excel(annot_path)
        df.columns = [c.strip() for c in df.columns]

        label_col = next((c for c in df.columns if 'AF' in c and 'Label' in c), None)
        if not label_col:
            label_col = next((c for c in df.columns if c.strip() == 'AF' or 'AF' in c), None)
        file_col = df.columns[0]

        if not label_col:
            print(f"    Could not identify Label column in {annot_path}")
            return None

        ep_start_col = next((c for c in df.columns if 'Episode Start' in c), None)
        ep_end_col   = next((c for c in df.columns if 'Episode End' in c), None)

        valid_labels = {'yes', 'no', 'sustained', 'paroxysmal', '0', '1', 'true', 'false'}
        gt_map = {}
        for _, row in df.iterrows():
            fname     = str(row[file_col]).strip()
            label_str = str(row[label_col]).lower().strip()

            if not any(v in label_str for v in valid_labels):
                continue

            base_fname = os.path.splitext(os.path.basename(fname))[0].strip()

            start_t = end_t = None
            if ep_start_col and ep_end_col:
                try:
                    start_t = float(row[ep_start_col])
                    end_t   = float(row[ep_end_col])
                except (ValueError, TypeError):
                    pass

            if 'sustained' in label_str:
                is_af, af_type = True, 'sustained'
            elif 'paroxysmal' in label_str:
                is_af, af_type = True, 'paroxysmal'
            elif 'yes' in label_str or '1' == label_str or 'true' in label_str:
                is_af, af_type = True, 'af'
            else:
                is_af, af_type = False, 'no_af'

            if base_fname not in gt_map:
                gt_map[base_fname] = {'is_af': is_af, 'af_type': af_type, 'episodes': []}

            if start_t is not None and end_t is not None:
                gt_map[base_fname]['episodes'].append((start_t, end_t))

        n_af       = sum(1 for v in gt_map.values() if v['is_af'])
        n_sustained = sum(1 for v in gt_map.values() if v['af_type'] == 'sustained')
        n_parox    = sum(1 for v in gt_map.values() if v['af_type'] == 'paroxysmal')
        print(f"    Loaded {len(gt_map)} annotations: {n_af} AF "
              f"({n_sustained} sustained, {n_parox} paroxysmal), {len(gt_map)-n_af} No-AF")
        return gt_map

    except Exception as e:
        print(f"    Error reading annotations: {e}")
        return None


def train_rf():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    base_dir    = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, "Results")
    file_path   = os.path.join(results_dir, "all_results_summary.csv")

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print("Loading analysis results...")
    df = pd.read_csv(file_path)

    print("Loading Ground Truth Annotations...")
    gt_dir = os.path.join(base_dir, "GroundtruthverifiedEKGData")
    gt_map = load_ground_truth(gt_dir)

    if gt_map is None:
        print("No Ground Truth Annotations found — cannot train.")
        return

    def normalize_name(name):
        base = os.path.splitext(os.path.basename(name))[0]
        return base.replace('_Export', '').replace('.iwxdata', '').lower().strip()

    norm_gt_map = {normalize_name(k): v for k, v in gt_map.items()}

    def get_window_label(row):
        norm_row = normalize_name(row['file'])
        file_ann = norm_gt_map.get(norm_row)
        if file_ann is None:
            for k, v in norm_gt_map.items():
                if norm_row in k or k in norm_row:
                    file_ann = v
                    break
        if not file_ann:
            return None, None

        w_start  = row['start_time']
        w_end    = row['end_time']
        is_af    = file_ann['is_af']
        af_type  = file_ann['af_type']
        episodes = file_ann.get('episodes', [])

        if not is_af:
            return False, 'no_af'
        if af_type == 'sustained':
            return True, 'sustained'
        if af_type == 'paroxysmal':
            if not episodes:
                return True, 'paroxysmal'
            # Only label the window AF if it has >= MIN_AF_OVERLAP_SEC seconds
            # inside a known episode — avoids mislabeling boundary windows
            for ep_start, ep_end in episodes:
                overlap = max(0, min(w_end, ep_end) - max(w_start, ep_start))
                if overlap >= MIN_AF_OVERLAP_SEC:
                    return True, 'paroxysmal'
            return False, 'no_af'

        return None, None

    labels = df.apply(get_window_label, axis=1)
    df['manual_label'] = [x[0] for x in labels]
    df['af_type']      = [x[1] for x in labels]

    matched_count = df['manual_label'].notna().sum()
    print(f"    Matched {matched_count} files to annotations.")

    if matched_count == 0:
        print("Match failure — check that annotation filenames match export filenames.")
        for f in df['file'].head(3).tolist():
            print(f"  result: '{normalize_name(f)}'")
        for k in list(norm_gt_map.keys())[:3]:
            print(f"  annot:  '{k}'")

    df_clean = df.dropna(subset=['manual_label']).copy()
    print(f"\nEntries with annotations: {len(df_clean)} / {len(df)}")
    if len(df_clean) > 0:
        print(f"  AF (sustained):  {(df_clean['af_type']=='sustained').sum()} windows")
        print(f"  AF (paroxysmal): {(df_clean['af_type']=='paroxysmal').sum()} windows  "
              f"[min overlap >= {MIN_AF_OVERLAP_SEC}s]")
        print(f"  No-AF:           {(df_clean['manual_label']==False).sum()} windows")

    if len(df_clean) < 10:
        print("Not enough annotated data to train.")
        return

    target_col   = 'manual_label'
    exclude_cols = ['file', 'window_idx', 'start_time', 'end_time', 'is_af',
                    'note', 'peaks', 'manual_label', 'af_type', 'annotation']
    feature_cols = [c for c in df_clean.columns
                    if c not in exclude_cols
                    and pd.api.types.is_numeric_dtype(df_clean[c])
                    and df_clean[c].notna().sum() > 0]

    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X      = df_clean[feature_cols]
    y      = df_clean[target_col].astype(int)
    groups = df_clean['file']

    imputer  = SimpleImputer(strategy='median')
    X_imp    = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_cols)

    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    spw = neg_count / pos_count

    brf = BalancedRandomForestClassifier(n_estimators=500, random_state=42,
                                          replacement=True, sampling_strategy='auto')
    xgb = XGBClassifier(n_estimators=500, scale_pos_weight=spw, max_depth=5,
                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         eval_metric='logloss', random_state=42, verbosity=0)
    ensemble = VotingClassifier(estimators=[('brf', brf), ('xgb', xgb)], voting='soft')

    candidates = {
        "BalancedRF":  brf,
        "XGBoost":     xgb,
        "Ensemble":    ensemble,
    }

    cv = LeaveOneGroupOut()
    best_name    = None
    best_pr_auc  = -1
    best_scores  = None
    best_model   = None

    print(f"\nLeave-One-File-Out CV ({df_clean['file'].nunique()} groups)...")
    for name, model in candidates.items():
        print(f"  {name}...", end='', flush=True)
        scores = cross_val_predict(model, X_scaled, y, groups=groups,
                                   cv=cv, method="predict_proba")[:, 1]
        p, r, _ = precision_recall_curve(y, scores)
        pa = auc(r, p)
        print(f" PR-AUC = {pa:.4f}")
        if pa > best_pr_auc:
            best_pr_auc  = pa
            best_name    = name
            best_scores  = scores
            best_model   = model

    print(f"\n  Winner: {best_name} (PR-AUC = {best_pr_auc:.4f})")
    y_scores = best_scores

    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    pr_auc = auc(recall, precision)

    # Try to find a threshold satisfying Recall >= 0.80 AND Accuracy >= 0.90
    best_threshold  = None
    best_f1         = -1
    threshold_type  = None
    cand_thresholds = np.unique(np.concatenate([thresholds, [0.5]]))

    for t in cand_thresholds:
        yp   = (y_scores >= t).astype(int)
        tp   = np.sum((yp == 1) & (y == 1))
        fn   = np.sum((yp == 0) & (y == 1))
        fp   = np.sum((yp == 1) & (y == 0))
        rec  = tp / (tp + fn + 1e-10)
        prec = tp / (tp + fp + 1e-10)
        acc  = accuracy_score(y, yp)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        if rec >= 0.80 and acc >= 0.90 and f1 > best_f1:
            best_f1, best_threshold, threshold_type = f1, t, "Recall>=0.80 & Accuracy>=0.90"

    if best_threshold is None:
        print("\n[NOTE] No threshold satisfies both constraints simultaneously.")
        print("  Using best accuracy among thresholds with Recall >= 0.80.\n")
        best_acc = -1
        for t in cand_thresholds:
            yp  = (y_scores >= t).astype(int)
            tp  = np.sum((yp == 1) & (y == 1))
            fn  = np.sum((yp == 0) & (y == 1))
            rec = tp / (tp + fn + 1e-10)
            acc = accuracy_score(y, yp)
            if rec >= 0.80 and acc > best_acc:
                best_acc, best_threshold = acc, t
                threshold_type = f"Best accuracy ({acc:.3f}) at Recall>=0.80"

    if best_threshold is None:
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-10)
        idx = np.argmax(f2)
        best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        threshold_type = "max F2 (fallback)"

    optimal_threshold = best_threshold

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Threshold ({threshold_type}): {optimal_threshold:.4f}")

    y_pred = (y_scores >= optimal_threshold).astype(int)

    print(f"\nAccuracy: {accuracy_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # PR curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, color='darkorange', label=f'PR Curve (AUC={pr_auc:.3f})')
    if len(thresholds) > 0:
        t_idx = min(np.searchsorted(thresholds, optimal_threshold), len(recall) - 1)
        plt.plot(recall[t_idx], precision[t_idx], 'ko', markersize=8,
                 label=f'Cutoff={optimal_threshold:.2f}')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AF+ = minority class)')
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    pr_path = os.path.join(results_dir, "rf_pr_curve.png")
    plt.savefig(pr_path); plt.close()
    print(f"\nPR curve → {pr_path}")

    # Feature importance (final fit on all data)
    best_model.fit(X_scaled, y)
    feature_imp = (pd.DataFrame({'Feature': feature_cols,
                                  'Importance': best_model.feature_importances_})
                   .sort_values('Importance', ascending=False))

    print("\nTop 10 features:")
    print(feature_imp.head(10).to_string(index=False))

    imp_path = os.path.join(results_dir, "rf_feature_importance.csv")
    feature_imp.to_csv(imp_path, index=False)

    plt.figure(figsize=(10, 8))
    top20 = feature_imp.head(20).sort_values('Importance')
    plt.barh(top20['Feature'], top20['Importance'], color='teal')
    plt.xlabel('Importance')
    plt.title('Top 20 AF Predictors (LOFO-CV, BalancedRF)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rf_feature_importance.png"))
    plt.close()
    print(f"Feature importance → {imp_path}")


if __name__ == "__main__":
    train_rf()
