import pandas as pd

from af_ekg import analyze_recording
from af_ekg.features import block_starts, clean_ecg, window_features
from af_ekg.model import choose_threshold, threshold_for_recall
from conftest import FS, write_export


def test_irregular_rhythm_scores_as_more_irregular(ecg):
    regular = window_features(clean_ecg(ecg(seconds=45), FS), FS)
    irregular = window_features(clean_ecg(ecg(seconds=45, irregular=True), FS), FS)
    assert len(regular) == len(irregular) == 11
    assert 550 < regular.HR_Mean.median() < 650
    assert irregular.RMSSD.median() > 5 * regular.RMSSD.median()
    assert irregular['pRR_3.25'].median() > regular['pRR_3.25'].median() + 30


def test_short_recording_is_rejected(ecg):
    assert window_features(ecg(seconds=30), FS) is None


def test_end_to_end_on_an_export_file(tmp_path, ecg):
    x = ecg(seconds=50, seed=2)
    result = analyze_recording(write_export(tmp_path / 'rec.txt', x, 0.6 * x))
    assert len(result.windows) == 11 and result.decisions.kept.all()


def test_threshold_separates_perfectly_ranked_scores():
    recordings = pd.DataFrame({'recording_id': list('abcd'), 'cohort': 'x', 'af_label': [0, 0, 1, 1]})
    scores = pd.Series([0.1, 0.2, 0.8, 0.9], index=list('abcd'))
    assert 0.2 < choose_threshold(recordings, scores) <= 0.8


def test_blocks_cover_the_recording_in_30_s_steps_from_10_s():
    assert block_starts(39.9) == []
    assert block_starts(40) == [10]
    assert block_starts(135) == [10, 40, 70, 100]


def test_long_recording_is_scanned_block_by_block(tmp_path, ecg):
    x = ecg(seconds=105, seed=4)
    result = analyze_recording(write_export(tmp_path / 'long.txt', x, 0.6 * x))
    assert sorted(result.windows.block_start.unique()) == [10, 40, 70]
    assert (result.windows.groupby('block_start').size() == 11).all()


def test_recall_threshold_keeps_the_requested_share_of_af():
    recordings = pd.DataFrame({'recording_id': [f'r{i}' for i in range(12)], 'af_label': [1] * 10 + [0] * 2})
    scores = pd.Series([0.05, *[0.5 + i / 20 for i in range(9)], 0.3, 0.9], index=recordings.recording_id)
    assert threshold_for_recall(recordings, scores, 0.9) == 0.5
    assert threshold_for_recall(recordings, scores, 1.0) == 0.05
