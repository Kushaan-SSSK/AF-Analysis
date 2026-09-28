"""Add a lab's LabScribe exports to a dataset in the published format.

    python prepare_dataset.py --exports path/to/exports --metadata metadata.csv --lab XX --out path/to/dataset

See docs/data_preparation.md.
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['recording_id', 'lab', 'file', 'original_filename', 'session_date', 'animal_id', 'sex', 'age',
           'genotype', 'treatment', 'disease_or_control', 'rhythm', 'af_label', 'label_source',
           'used_in_model', 'not_used_reason', 'duration_s', 'sample_rate_hz', 'annotator_notes']
METADATA_COLUMNS = ['original_filename', 'animal_id', 'session_date', 'sex', 'age', 'genotype', 'treatment',
                    'disease_or_control', 'rhythm', 'annotator_notes']
AF_RHYTHMS = {'af', 'paroxysmal af', 'af/atrial flutter'}
NOT_AF_RHYTHMS = {'sinus rhythm', 'junctional rhythm', 'atrial bigeminy'}
EXPORT_SUFFIXES = {'.txt', '.xls'}


def stem_key(name):
    """File name without extension or LabScribe's '_Export' suffix, lower case."""
    stem = Path(str(name)).stem
    return re.sub(r'_export( \(\d+\))?$', '', stem, flags=re.I).strip().lower()


def describe_export(path):
    """(duration in s, sampling rate in Hz) of a LabScribe text export; raises if the format is wrong."""
    header = pd.read_csv(path, sep='\t', nrows=0).columns
    if 'Time' not in header or 'Raw Ch 1' not in header:
        raise ValueError(f'{path.name}: needs tab-separated columns Time and Raw Ch 1 (found {list(header)[:5]})')
    time = pd.to_numeric(pd.read_csv(path, sep='\t', usecols=['Time'], on_bad_lines='skip').Time,
                         errors='coerce').dropna().to_numpy()
    if len(time) < 2:
        raise ValueError(f'{path.name}: no numeric samples')
    step = float(np.median(np.diff(time[:2000])))
    return round(len(time) * step, 1), int(round(1 / step))


def af_label(rhythm):
    value = str(rhythm).strip().lower()
    if value in AF_RHYTHMS:
        return 1
    if value in NOT_AF_RHYTHMS:
        return 0
    return None


def next_number(table, lab):
    numbers = table.loc[table.lab == lab, 'recording_id'].str.extract(r'-(\d+)$')[0].astype(float)
    return int(numbers.max()) + 1 if len(numbers.dropna()) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exports', type=Path, required=True, help='folder of LabScribe text exports')
    parser.add_argument('--metadata', type=Path, required=True, help='CSV with one row per recording')
    parser.add_argument('--lab', required=True, help='short lab code used in IDs, e.g. JJ')
    parser.add_argument('--label-source', default=None, help='where the rhythm calls come from')
    parser.add_argument('--out', type=Path, required=True, help='dataset folder (created if needed)')
    args = parser.parse_args()

    lab = args.lab.upper()
    metadata = pd.read_csv(args.metadata, dtype=str).fillna('')
    missing = [c for c in METADATA_COLUMNS if c not in metadata]
    if missing:
        raise SystemExit(f'Metadata is missing columns: {missing}')
    exports = {stem_key(p.name): p for p in args.exports.iterdir() if p.suffix.lower() in EXPORT_SUFFIXES}

    table_path = args.out / 'recordings.csv'
    table = pd.read_csv(table_path, dtype=str) if table_path.exists() else pd.DataFrame(columns=COLUMNS)
    number = next_number(table, lab)
    rows, problems = [], []
    for record in metadata.itertuples():
        source = exports.get(stem_key(record.original_filename))
        if source is None:
            problems.append(f'{record.original_filename}: no matching export in {args.exports}')
            continue
        try:
            duration, rate = describe_export(source)
        except ValueError as error:
            problems.append(str(error))
            continue
        recording_id = f'{lab}-{number:03d}'
        number += 1
        target = args.out / 'recordings' / lab.lower() / f'{recording_id}.txt'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        label = af_label(record.rhythm)
        rows.append({'recording_id': recording_id, 'lab': lab, 'file': target.relative_to(args.out).as_posix(),
                     'original_filename': Path(record.original_filename).stem,
                     'session_date': record.session_date, 'animal_id': record.animal_id, 'sex': record.sex,
                     'age': record.age, 'genotype': record.genotype, 'treatment': record.treatment,
                     'disease_or_control': record.disease_or_control, 'rhythm': record.rhythm,
                     'af_label': '' if label is None else label,
                     'label_source': args.label_source or f'{lab} lab spreadsheet',
                     'used_in_model': 'no' if label is None else 'yes',
                     'not_used_reason': 'Rhythm not determined' if label is None else '',
                     'duration_s': duration, 'sample_rate_hz': rate, 'annotator_notes': record.annotator_notes})

    matched = {stem_key(r) for r in metadata.original_filename}
    problems += [f'{p.name}: export has no row in the metadata' for key, p in exports.items() if key not in matched]
    if rows:
        args.out.mkdir(parents=True, exist_ok=True)
        pd.concat([table, pd.DataFrame(rows)], ignore_index=True)[COLUMNS].to_csv(table_path, index=False)
    print(f'Added {len(rows)} recordings from lab {lab} to {table_path}')
    for problem in problems:
        print('  not added:', problem)


if __name__ == '__main__':
    main()
