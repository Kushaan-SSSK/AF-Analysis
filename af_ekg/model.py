"""Window-level XGBoost classifier with two operating modes.

The classifier is trained on the windows of the 10-40 s block. A recording's
score is its highest window score:
  balanced   over the 10-40 s block; threshold maximises balanced accuracy
  sensitive  over every 30 s block of the recording; threshold keeps 97% recall
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from .features import ANALYSIS_START_S, MODEL_FEATURES

# SF and MS are pooled for stratification and threshold choice because each
# lab has fewer than five non-AF animals.
COHORT = {'JJ': 'jj', 'SF': 'sf_ms', 'MS': 'sf_ms'}
MODES = ('balanced', 'sensitive')
SENSITIVE_RECALL = 0.97


def feature_matrix(windows):
    return windows[[c for c in MODEL_FEATURES if c in windows]].astype(float)


def fit(windows, seed=42):
    x = feature_matrix(windows)
    y = windows.af_label.astype(int).to_numpy()
    classifier = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9,
                               colsample_bytree=0.9, scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                               reg_lambda=1, n_jobs=4, random_state=seed,
                               objective='binary:logistic', eval_metric='logloss')
    pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), RobustScaler(), classifier)
    pipeline.fit(x, y)
    return {'pipeline': pipeline, 'features': list(x.columns)}


def predict(model, windows):
    """AF score per recording (max over its windows)."""
    x = windows.reindex(columns=model['features']).astype(float)
    scores = pd.DataFrame({'recording_id': windows.recording_id.to_numpy(),
                           'score': model['pipeline'].predict_proba(x)[:, 1]})
    return scores.groupby('recording_id').score.max()


def first_block(windows):
    return windows[windows.block_start == ANALYSIS_START_S]


def mode_scores(model, windows):
    return {'balanced': predict(model, first_block(windows)), 'sensitive': predict(model, windows)}


def animal_folds(recordings, n_splits, seed):
    animals = recordings.groupby('animal', as_index=False).agg(cohort=('cohort', 'first'), label=('af_label', 'max'))
    strata = animals.cohort + '_' + animals.label.astype(str)
    if strata.value_counts().min() < n_splits:
        raise ValueError(f'Too few animals in a cohort/label group for {n_splits} folds')
    animals['fold'] = -1
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(animals, strata)):
        animals.loc[test, 'fold'] = fold
    return animals.set_index('animal').fold


def choose_threshold(recordings, scores):
    """Threshold maximising balanced accuracy, averaged over cohorts."""
    best_value, best_threshold = -np.inf, 0.5
    for threshold in np.unique(np.r_[0, scores.to_numpy(), 1]):
        values = []
        for _, group in recordings.groupby('cohort'):
            y = group.af_label.to_numpy()
            called = scores.loc[group.recording_id].to_numpy() >= threshold
            if len(np.unique(y)) == 2:
                values.append(0.5 * (called[y == 1].mean() + (~called[y == 0]).mean()))
        value = np.mean(values)
        if value > best_value or (value == best_value and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_value, best_threshold = value, float(threshold)
    return best_threshold


def threshold_for_recall(recordings, scores, target=SENSITIVE_RECALL):
    """Highest threshold that still calls `target` of the AF recordings AF."""
    af = np.sort(scores.loc[recordings[recordings.af_label == 1].recording_id].to_numpy())
    return float(af[int(np.floor((1 - target) * len(af) + 1e-9))])


def tune_thresholds(windows, recordings, seed):
    """Both mode thresholds from animal-grouped inner cross-validation on the training data only."""
    per_cell = recordings.groupby('animal').agg(cohort=('cohort', 'first'), label=('af_label', 'max'))
    n_folds = min(3, int(per_cell.groupby(['cohort', 'label']).size().min()))
    in_fold = windows.animal.map(animal_folds(recordings, n_folds, seed))
    parts = [mode_scores(fit(first_block(windows[in_fold != k]), seed + k), windows[in_fold == k])
             for k in range(n_folds)]
    scores = {mode: pd.concat([part[mode] for part in parts]) for mode in MODES}
    return {'balanced': choose_threshold(recordings, scores['balanced']),
            'sensitive': threshold_for_recall(recordings, scores['sensitive'])}


def cross_validate(windows, recordings, repeats=3, n_splits=5):
    """Repeated animal-grouped CV; each outer fold tunes its own thresholds."""
    rows = []
    for repeat in range(repeats):
        folds = animal_folds(recordings, n_splits, 42 + repeat)
        for fold in range(n_splits):
            test_mask = windows.animal.map(folds).eq(fold)
            train, test = windows[~test_mask], windows[test_mask]
            train_recordings = recordings[recordings.recording_id.isin(train.recording_id)]
            thresholds = tune_thresholds(train, train_recordings, 1000 + 100 * repeat + fold)
            scores = mode_scores(fit(first_block(train), 42 + repeat), test)
            for mode in MODES:
                rows.append(pd.DataFrame({'recording_id': scores[mode].index, 'mode': mode,
                                          'score': scores[mode].values, 'threshold': thresholds[mode],
                                          'called_af': scores[mode].values >= thresholds[mode],
                                          'repeat': repeat, 'fold': fold}))
    predictions = pd.concat(rows, ignore_index=True)
    averaged = predictions.groupby(['mode', 'recording_id']).agg(score=('score', 'mean'), votes=('called_af', 'mean'))
    averaged['called_af'] = averaged.votes > 0.5
    averaged = averaged.reset_index().merge(recordings, on='recording_id', validate='many_to_one')
    return predictions, averaged


def summarize(results):
    rows = []
    groups = [((mode, 'All'), g) for mode, g in results.groupby('mode')] + list(results.groupby(['mode', 'lab']))
    for (mode, name), group in groups:
        y, called = group.af_label.astype(int), group.called_af.astype(int)
        tn, fp, fn, tp = confusion_matrix(y, called, labels=[0, 1]).ravel()
        both = y.nunique() == 2
        rows.append({'mode': mode, 'group': name, 'recordings': len(group), 'af': int(y.sum()),
                     'auc': roc_auc_score(y, group.score) if both else np.nan,
                     'average_precision': average_precision_score(y, group.score) if both else np.nan,
                     'recall': tp / (tp + fn) if tp + fn else np.nan,
                     'specificity': tn / (tn + fp) if tn + fp else np.nan,
                     'precision': tp / (tp + fp) if tp + fp else np.nan,
                     'accuracy': float((y == called).mean()), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
    return pd.DataFrame(rows)
