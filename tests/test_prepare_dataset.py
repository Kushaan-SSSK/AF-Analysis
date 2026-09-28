import sys

import pandas as pd

import prepare_dataset
from conftest import write_export


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['prepare_dataset.py', *map(str, args)])
    prepare_dataset.main()


def metadata(path, rows):
    frame = pd.DataFrame(rows, columns=['original_filename', 'rhythm'])
    for column in prepare_dataset.METADATA_COLUMNS:
        if column not in frame:
            frame[column] = ''
    frame.to_csv(path, index=False)
    return path


def test_exports_are_renamed_labelled_and_numbered(tmp_path, monkeypatch, ecg):
    exports = tmp_path / 'exports'
    exports.mkdir()
    x = ecg(seconds=45)
    for name in ('mouse1_Export.txt', 'mouse2_Export.txt', 'mouse3_Export.txt'):
        write_export(exports / name, x, 0.6 * x)
    sheet = metadata(tmp_path / 'meta.csv', [('mouse1.iwxdata', 'AF'), ('mouse2.iwxdata', 'Sinus rhythm'),
                                             ('mouse3.iwxdata', 'Unknown')])
    out = tmp_path / 'dataset'
    run(monkeypatch, '--exports', exports, '--metadata', sheet, '--lab', 'xy', '--out', out)

    table = pd.read_csv(out / 'recordings.csv')
    assert list(table.recording_id) == ['XY-001', 'XY-002', 'XY-003']
    assert list(table.af_label.fillna(-1)) == [1, 0, -1]
    assert list(table.used_in_model) == ['yes', 'yes', 'no']
    assert (out / 'recordings/xy/XY-001.txt').exists()
    assert table.duration_s.iloc[0] == 45.0 and table.sample_rate_hz.iloc[0] == 1000


def test_numbering_continues_and_bad_files_are_reported(tmp_path, monkeypatch, ecg, capsys):
    exports = tmp_path / 'exports'
    exports.mkdir()
    x = ecg(seconds=45)
    write_export(exports / 'a_Export.txt', x, x)
    (exports / 'b_Export.txt').write_text('Seconds\tVoltage\n0\t1\n')
    out = tmp_path / 'dataset'
    sheet = metadata(tmp_path / 'meta.csv', [('a', 'AF'), ('b', 'AF')])
    run(monkeypatch, '--exports', exports, '--metadata', sheet, '--lab', 'XY', '--out', out)
    run(monkeypatch, '--exports', exports, '--metadata', metadata(tmp_path / 'm2.csv', [('a', 'AF')]),
        '--lab', 'XY', '--out', out)
    assert list(pd.read_csv(out / 'recordings.csv').recording_id) == ['XY-001', 'XY-002']
    assert 'needs tab-separated columns Time and Raw Ch 1' in capsys.readouterr().out
