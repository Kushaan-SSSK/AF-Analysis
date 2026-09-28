"""HTML report covering every recording in the databank.

Recordings used in training show their cross-validated scores and calls for
both modes (from models/cv_predictions.csv), so no recording is scored by a
model that saw it. Other recordings are scored by the final model.

    python databank_report.py --databank path/to/AF_EKG_Databank --out databank_report.html
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from af_ekg import analyze_recording, load_model
from af_ekg.model import MODES
from af_ekg.report import render_report

METADATA = [('animal_id', 'Animal {}'), ('sex', '{}'), ('age', '{}'), ('genotype', '{}'),
            ('treatment', '{}'), ('disease_or_control', '{}')]


def _analyze(args):
    path, model_path = args
    return analyze_recording(path, load_model(model_path))


def details_for(row, cv):
    info = [row.lab] + [template.format(getattr(row, column)) for column, template in METADATA
                        if pd.notna(getattr(row, column)) and str(getattr(row, column)).strip()]
    label = f'Label: {row.rhythm} ({row.label_source})'
    if row.used_in_model != 'yes':
        label += f'; not used in training: {row.not_used_reason}'
    info.append(label)
    info.append(f'{row.duration_s:.0f} s recorded at {row.sample_rate_hz:.0f} Hz')
    if pd.notna(row.annotator_notes) and str(row.annotator_notes).strip():
        info.append(f'Notes: {row.annotator_notes}')
    detail = {'name': row.recording_id, 'title': row.original_filename, 'lab': row.lab,
              'label': 'AF' if row.af_label == 1 else 'Not AF' if row.af_label == 0 else row.rhythm,
              'info': info}
    detail['modes'] = {m.mode: {'score': round(m.score, 3), 'af': bool(m.called_af),
                                'basis': f'AF in {round(m.votes * 3)} of 3 cross-validation runs'}
                       for m in cv[cv.recording_id == row.recording_id].itertuples()}
    return detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--databank', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/af_model.joblib'))
    parser.add_argument('--cv', type=Path, default=Path('models/cv_predictions.csv'))
    parser.add_argument('--out', type=Path, default=Path('databank_report.html'))
    parser.add_argument('--mode', choices=MODES, default='balanced')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    table = pd.read_csv(args.databank / 'recordings.csv', dtype={'animal_id': str})
    table = table[table.file.notna()].reset_index(drop=True)
    cv = pd.read_csv(args.cv)
    paths = [args.databank / f for f in table.file]
    with ProcessPoolExecutor(args.workers) as pool:
        analyses = list(pool.map(_analyze, [(p, args.model) for p in paths]))
    details = {str(a.path): details_for(row, cv) for row, a in zip(table.itertuples(), analyses)}
    note = 'training recordings show cross-validated results'
    args.out.write_text(render_report(analyses, args.mode, details, note), encoding='utf-8')
    print(f'{len(analyses)} recordings written to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
