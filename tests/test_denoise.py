import numpy as np
import pandas as pd

from af_ekg.denoise import denoise


def windows(value=1.0, quality='ok'):
    starts = np.arange(0, 21, 2.0)
    frame = pd.DataFrame({'window_idx': range(len(starts)), 'start_time': starts, 'end_time': starts + 10,
                          'pRR_3.25': value + np.arange(len(starts)), 'window_quality': quality})
    frame['prev_pRR_3.25'] = frame['pRR_3.25'].shift(1).fillna(frame['pRR_3.25'])
    frame['next_pRR_3.25'] = frame['pRR_3.25'].shift(-1).fillna(frame['pRR_3.25'])
    return frame


def runs(*rows):
    return pd.DataFrame(rows, columns=['lead', 'start_s', 'end_s', 'attribution'])


def test_switches_to_the_cleaner_lead():
    kept, decisions = denoise(windows(), windows(100.0), runs(('Raw Ch 1', 10, 13, 'ch1_only')))
    assert list(decisions.lead[:2]) == ['Ch2', 'Ch2'] and (decisions.lead[2:] == 'Ch1').all()
    assert decisions.kept.all() and kept['pRR_3.25'].iloc[0] == 100.0


def test_noise_on_both_leads_is_dropped_not_switched():
    kept, decisions = denoise(windows(), windows(100.0), runs(('Raw Ch 1', 10, 15, 'both')))
    assert (decisions.lead == 'Ch1').all()
    assert list(kept.window_idx) == list(range(3, 11))


def test_failed_ch2_window_is_not_used_and_context_is_rebuilt():
    kept, decisions = denoise(windows(), windows(100.0, 'low'), runs(('Raw Ch 1', 10, 13, 'ch1_only')))
    assert (decisions.lead == 'Ch1').all()
    assert list(kept.window_idx) == list(range(2, 11))
    assert kept['prev_pRR_3.25'].iloc[0] == kept['pRR_3.25'].iloc[0]


def test_unmapped_recording_is_unchanged():
    kept, decisions = denoise(windows(), windows(100.0), None)
    assert decisions.kept.all() and (decisions.lead == 'Ch1').all()
    pd.testing.assert_frame_equal(kept, windows())


def test_noise_map_that_misses_the_window_is_ignored():
    kept, decisions = denoise(windows(), windows(100.0), runs(('Raw Ch 1', 0, 50, 'ch1_only')), (94.4, 215.0))
    assert decisions.kept.all() and (decisions.lead == 'Ch1').all()
