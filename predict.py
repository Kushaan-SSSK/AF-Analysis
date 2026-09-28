"""Score EKG exports and write a review list.

    python predict.py recordings/*.txt --mode sensitive --csv review_list.csv --report report.html

balanced   best overall accuracy; scores seconds 10-40
sensitive  for screening; scans the whole recording and misses very little AF,
           at the cost of flagging more recordings for manual review
"""
import argparse
from pathlib import Path

import pandas as pd

from af_ekg import analyze_recording, load_model
from af_ekg.model import MODES
from af_ekg.report import render_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recordings', nargs='+', type=Path)
    parser.add_argument('--mode', choices=MODES, default='balanced')
    parser.add_argument('--model', type=Path, default=Path('models/af_model.joblib'))
    parser.add_argument('--csv', type=Path, default=Path('scores.csv'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    model = load_model(args.model)
    analyses, rows = [], []
    for path in args.recordings:
        try:
            analysis = analyze_recording(path, model)
        except ValueError as error:
            print(f'Skipped {path.name}: {error}')
            continue
        analyses.append(analysis)
        row = {'recording': path.name, 'flagged': analysis.called_af(args.mode)}
        for mode in MODES:
            row[f'{mode}_score'] = round(analysis.scores[mode], 4)
            row[f'{mode}_call'] = 'AF' if analysis.called_af(mode) else 'Not AF'
        row['af_like_blocks_s'] = '; '.join(f'{a}-{b}' for a, b, _ in analysis.af_like_blocks())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(['flagged', f'{args.mode}_score'], ascending=False)
    table.to_csv(args.csv, index=False)
    print(f'{int(table.flagged.sum())} of {len(table)} recordings flagged as possible AF ({args.mode} mode); '
          f'list written to {args.csv}')
    if args.report and analyses:
        args.report.write_text(render_report(analyses, mode=args.mode), encoding='utf-8')
        print(f'Report written to {args.report}')


if __name__ == '__main__':
    main()
