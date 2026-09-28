import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from af_ekg.noise import label_runs, merge_runs, score_lead
from conftest import FS


def burst(n, start, end, seed=1):
    rng = np.random.default_rng(seed)
    b, a = butter(2, [20, 250], 'bandpass', fs=FS)
    noise = np.zeros(n)
    s, e = start * FS, end * FS
    noise[s:e] = filtfilt(b, a, rng.normal(0, 1.0, e - s)) * 1.5
    noise[s:e] += 0.8 * np.sign(np.sin(np.linspace(0, 6 * np.pi, e - s)))
    return noise


def runs_for(leads):
    return label_runs({lead: merge_runs(score_lead(x, FS)) for lead, x in leads.items()})


def test_clean_signal_has_no_runs(ecg):
    x = ecg()
    assert runs_for({'Raw Ch 1': x, 'Raw Ch 2': 0.6 * x}) == []


def test_af_like_irregular_rhythm_is_not_noise(ecg):
    x = ecg(irregular=True, seed=3)
    assert score_lead(x, FS).noisy.mean() < 0.05
    assert runs_for({'Raw Ch 1': x, 'Raw Ch 2': 0.6 * x}) == []


def test_burst_found_on_the_right_lead(ecg):
    x = ecg(seed=5)
    runs = runs_for({'Raw Ch 1': x, 'Raw Ch 2': 0.6 * x + burst(len(x), 20, 35)})
    assert len(runs) == 1 and runs[0]['attribution'] == 'ch2_only'
    assert abs(runs[0]['start_s'] - 20) <= 1 and abs(runs[0]['end_s'] - 35) <= 1


def test_inverted_polarity_gives_the_same_runs(ecg):
    x = ecg(seed=7)
    noisy = x + burst(len(x), 10, 22, seed=2)
    a = runs_for({'Raw Ch 1': noisy, 'Raw Ch 2': 0.6 * x})
    b = runs_for({'Raw Ch 1': -noisy, 'Raw Ch 2': -0.6 * x})
    assert [r['attribution'] for r in a] == [r['attribution'] for r in b] == ['ch1_only']


def test_one_large_artifact_does_not_flag_the_whole_lead(ecg):
    x = ecg(seed=11)
    x[30 * FS:34 * FS] += 25 * np.sign(np.sin(np.linspace(0, 20 * np.pi, 4 * FS)))
    table = score_lead(x, FS)
    assert table[(table.end_s < 29) | (table.start_s > 35)].noisy.mean() < 0.02


def test_merge_bridges_short_gaps_and_drops_blips():
    starts = np.arange(0, 30, 0.5)
    noisy = ((starts >= 5) & (starts < 9)) | ((starts >= 10) & (starts < 14)) | ((starts >= 20) & (starts < 21))
    table = pd.DataFrame({'start_s': starts, 'end_s': starts + 1.0, 'noisy': noisy})
    assert merge_runs(table) == [(5.0, 14.5)]
