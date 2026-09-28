"""Choose the cleaner lead for each analysis window and drop windows that stay noisy.

For each 10 s window, Ch2 replaces Ch1 when Ch2 has strictly less noise and its
window passes the peak-count/heart-rate check. The window is then dropped if its
chosen lead still overlaps a noise run, unless every window of the block is
noisy. Context features are rebuilt over the kept windows. A block outside the
span the noise map covers is left unchanged.
"""
import numpy as np
import pandas as pd

from .features import ANALYSIS_START_S, BLOCK_S, NOT_CONTEXT

LEAD_KEYS = {'Raw Ch 1': 'ch1', 'Raw Ch 2': 'ch2'}


def noise_seconds(windows, runs, scored_span=None, block_start=ANALYSIS_START_S):
    """Seconds of each window covered by a noise run, per lead (NaN when unknown).

    `runs` is None when the recording could not be noise-mapped.
    """
    lo = block_start + windows.start_time.to_numpy()
    hi = block_start + windows.end_time.to_numpy()
    span_ok = scored_span is None or (scored_span[0] <= block_start + 1e-6
                                      and scored_span[1] >= block_start + BLOCK_S)
    known = runs is not None and span_ok
    out = {}
    for lead, key in LEAD_KEYS.items():
        if not known:
            out[key] = np.full(len(windows), np.nan)
            continue
        spans = runs.loc[(runs.lead == lead) | (runs.attribution == 'both'), ['start_s', 'end_s']].to_numpy()
        covered = np.array([sum(max(0.0, min(h, e) - max(l, s)) for s, e in spans) for l, h in zip(lo, hi)])
        out[key] = covered
    return pd.DataFrame(out, index=windows.index)


def _rebuild_context(windows):
    bases = [c[5:] for c in windows.columns if c.startswith('prev_') and c[5:] in windows]
    for column in bases:
        windows['prev_' + column] = windows[column].shift(1).fillna(windows[column])
        windows['next_' + column] = windows[column].shift(-1).fillna(windows[column])
    return windows


def denoise(ch1, ch2, runs, scored_span=None, block_start=ANALYSIS_START_S):
    """Returns (kept windows, per-window decisions) for one 30 s block."""
    ch1 = ch1.sort_values('window_idx').reset_index(drop=True)
    noise = noise_seconds(ch1, runs, scored_span, block_start)
    known = noise.ch1.notna().to_numpy()
    if ch2 is None:
        ch2 = ch1[['window_idx']].assign(window_quality='low')
    ch2 = ch2.set_index('window_idx').reindex(ch1.window_idx)
    use_ch2 = known & ch2.window_quality.eq('ok').to_numpy() & (noise.ch2 < noise.ch1).to_numpy()

    shared = [c for c in ch1.columns if c in ch2.columns and c not in NOT_CONTEXT
              and not c.startswith(('prev_', 'next_'))]
    windows = ch1.copy()
    for column in shared:
        values = windows[column].to_numpy(copy=True).astype(object)
        values[use_ch2] = ch2[column].to_numpy()[use_ch2]
        windows[column] = pd.Series(values, index=windows.index).infer_objects()

    chosen_noise = np.where(use_ch2, noise.ch2, noise.ch1)
    noisy = known & (chosen_noise > 0)
    keep = ~noisy | noisy.all()
    decisions = pd.DataFrame({'window_idx': ch1.window_idx, 'lead': np.where(use_ch2, 'Ch2', 'Ch1'),
                              'ch1_noise_s': noise.ch1, 'ch2_noise_s': noise.ch2, 'kept': keep})
    return _rebuild_context(windows[keep].reset_index(drop=True)), decisions
