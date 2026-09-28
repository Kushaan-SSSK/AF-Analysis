"""Train and cross-validate the AF model on the annotated databank.

    python train.py --databank path/to/AF_EKG_Databank
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import pandas as pd

from af_ekg import analyze_recording
from af_ekg.model import COHORT, cross_validate, first_block, fit, summarize, tune_thresholds


def recording_windows(args):
    recording_id, path, noise_step = args
    return analyze_recording(path, noise_step=noise_step).windows.assign(recording_id=recording_id)


def build_windows(databank, recordings, workers, noise_step):
    jobs = [(r.recording_id, databank / r.file, noise_step) for r in recordings.itertuples()]
    with ProcessPoolExecutor(workers) as pool:
        return pd.concat(pool.map(recording_windows, jobs), ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--databank', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('models'))
    parser.add_argument('--cache', type=Path, default=Path('build/windows.csv'), help='window features of every block')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--no-noise-step', action='store_true', help='score Ch1 only, without noise removal')
    parser.add_argument('--row-order-seed', type=int, help='shuffle recording order before training')
    args = parser.parse_args()

    table = pd.read_csv(args.databank / 'recordings.csv', dtype={'animal_id': str})
    recordings = table[table.used_in_model == 'yes'].copy()
    recordings['af_label'] = recordings.af_label.astype(int)
    recordings['animal'] = recordings.lab + ':' + recordings.animal_id
    recordings['cohort'] = recordings.lab.map(COHORT)
    recordings = recordings[['recording_id', 'file', 'lab', 'animal', 'cohort', 'af_label']]

    if args.cache.exists():
        windows = pd.read_csv(args.cache)
    else:
        windows = build_windows(args.databank, recordings, args.workers, not args.no_noise_step)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        windows.to_csv(args.cache, index=False)
    windows = windows.merge(recordings.drop(columns='file'), on='recording_id', validate='many_to_one')
    order = sorted(recordings.recording_id)
    if args.row_order_seed is not None:
        order = recordings.recording_id.sample(frac=1, random_state=args.row_order_seed).tolist()
    windows['order'] = windows.recording_id.map({rid: i for i, rid in enumerate(order)})
    windows = windows.sort_values(['order', 'block_start', 'window_idx']).drop(columns='order').reset_index(drop=True)

    _, results = cross_validate(windows, recordings, args.repeats)
    metrics = summarize(results)
    args.out.mkdir(parents=True, exist_ok=True)
    results.round({'score': 4, 'votes': 4}).to_csv(args.out / 'cv_predictions.csv', index=False)
    metrics.round(4).to_csv(args.out / 'cv_metrics.csv', index=False)

    model = fit(first_block(windows), seed=42)
    model['thresholds'] = tune_thresholds(windows, recordings, seed=7001)
    model['training_recordings'] = len(recordings)
    joblib.dump(model, args.out / 'af_model.joblib')
    print(metrics.round(3).to_string(index=False))
    print('Thresholds:', {mode: round(t, 4) for mode, t in model['thresholds'].items()})


if __name__ == '__main__':
    main()
