"""Per-lead noise detection.

Each lead is scored in 1 s segments (0.5 s hop) on beat-shape and amplitude
only. RR irregularity is never used, because it is the AF signal itself.
Flagged segments are merged into runs of at least 3 s.
"""
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

LEADS = ('Raw Ch 1', 'Raw Ch 2')
TARGET_FS = 1000
SEGMENT_S, HOP_S = 1.0, 0.5
MERGE_GAP_S, MIN_RUN_S = 2.0, 3.0

TEMPLATE_CORR_MIN = 0.70
FLOOR_REL_MAX = 2.5
FLOOR_ABS_MAX = 0.30
HF_REL_MAX = 2.5
AMP_RATIO_RANGE = (0.3, 3.0)


def load_leads(path):
    """Both leads at up to 1 kHz from the longest continuous timestamp segment.

    Returns (leads, fs, offset_s, end_s); offset_s/end_s place the segment on the
    sample-index clock of the full export.
    """
    frame = pd.read_csv(path, sep='\t', usecols=lambda c: c in ('Time', *LEADS),
                        on_bad_lines='skip', low_memory=False)
    frame = frame.apply(pd.to_numeric, errors='coerce').dropna(subset=['Time'])
    time = frame.Time.to_numpy()
    cuts = np.r_[0, np.flatnonzero(np.diff(time) < 0) + 1, len(time)]
    lo, hi = max(zip(cuts[:-1], cuts[1:]), key=lambda b: b[1] - b[0])
    frame = frame.iloc[lo:hi].reset_index(drop=True)
    for lead in LEADS:
        if lead in frame:
            frame[lead] = frame[lead].interpolate(limit_direction='both')
    # LabScribe prints Time at 1 ms precision after 100 s; rebuild it from the sample index.
    t = frame.Time.to_numpy()
    step = float(np.median(np.diff(t[:2000])))
    if step <= 0 or abs((t[-1] - t[0]) - (len(t) - 1) * step) > max(0.01 * (t[-1] - t[0]), 0.01):
        raise ValueError(f'{path}: sample count disagrees with the printed time span')
    fs_in = int(round(1 / step))
    fs = min(fs_in, TARGET_FS)
    divisor = gcd(fs_in, fs)
    up, down = fs // divisor, fs_in // divisor
    leads = {lead: (resample_poly(frame[lead].to_numpy(float), up, down) if up != down
                    else frame[lead].to_numpy(float)) for lead in LEADS if lead in frame}
    return leads, fs, lo * step, hi * step


def _highpass(x, fs):
    b, a = butter(2, 3, 'highpass', fs=fs)
    return filtfilt(b, a, x)


def _segment_starts(n, fs):
    seg, hop = int(SEGMENT_S * fs), int(HOP_S * fs)
    return range(0, n - seg + 1, hop), seg


def _beats(hp, fs):
    """R-peaks and a median QRS template, with reference levels from the typical segment."""
    starts, seg = _segment_starts(len(hp), fs)
    hi = np.median([np.percentile(hp[s:s + seg], 99.5) for s in starts])
    lo = np.median([np.percentile(hp[s:s + seg], 0.5) for s in starts])
    y = hp if abs(hi) >= abs(lo) else -hp
    ref = max(abs(hi), abs(lo)) - np.median(y)
    peaks = find_peaks(y, distance=int(0.055 * fs), prominence=0.35 * ref)[0]
    pre, post = int(0.020 * fs), int(0.030 * fs)
    peaks = peaks[(peaks >= pre) & (peaks < len(y) - post)]
    typical = peaks[(y[peaks] > 0.5 * ref) & (y[peaks] < 2.0 * ref)]
    template = (np.median(np.stack([y[p - pre:p + post] for p in typical]), axis=0)
                if len(typical) >= 5 else None)
    return y, peaks, template, float(ref), (pre, post)


def _template_corr(y, peaks, template, span):
    pre, post = span
    if template is None or len(peaks) == 0:
        return np.full(len(peaks), np.nan)
    beats = np.stack([y[p - pre:p + post] for p in peaks])
    beats = beats - beats.mean(1, keepdims=True)
    t = template - template.mean()
    return (beats @ t) / (np.linalg.norm(beats, axis=1) * np.linalg.norm(t) + 1e-12)


def score_lead(raw, fs):
    """One row per 1 s segment; `noisy` marks flagged segments."""
    y, peaks, template, r_amp, span = _beats(_highpass(raw, fs), fs)
    corr = _template_corr(y, peaks, template, span)
    starts, seg = _segment_starts(len(raw), fs)
    raw_hi, raw_lo = np.max(raw), np.min(raw)
    typical_std = np.median([np.std(raw[s:s + seg]) for s in starts]) + 1e-12
    r_amp = r_amp if np.isfinite(r_amp) and r_amp > 0 else 1e-12
    rows = []
    for start in starts:
        ys, rs = y[start:start + seg], raw[start:start + seg]
        in_seg = (peaks >= start) & (peaks < start + seg)
        rows.append({
            'start_s': start / fs, 'end_s': (start + seg) / fs, 'n_beats': int(in_seg.sum()),
            'template_corr': float(np.nanmedian(corr[in_seg])) if in_seg.sum() >= 2 else np.nan,
            'floor_abs': float(np.median(np.abs(ys - np.median(ys))) / r_amp),
            'hf': float(np.mean(np.abs(np.diff(ys, 2))) / r_amp),
            'p2p': float(np.percentile(ys, 99.5) - np.percentile(ys, 0.5)),
            'flat': bool(np.std(rs) < 0.02 * typical_std),
            'clip_frac': float(np.mean((rs >= raw_hi) | (rs <= raw_lo))),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table.assign(noisy=pd.Series(dtype=bool))
    for column in ('floor_abs', 'hf', 'p2p'):
        table[column + '_rel'] = table[column] / (np.median(table[column]) + 1e-12)
    table['noisy'] = ((table.template_corr < TEMPLATE_CORR_MIN)
                      | (table.floor_abs_rel > FLOOR_REL_MAX) | (table.floor_abs > FLOOR_ABS_MAX)
                      | (table.hf_rel > HF_REL_MAX)
                      | (table.p2p_rel < AMP_RATIO_RANGE[0]) | (table.p2p_rel > AMP_RATIO_RANGE[1])
                      | table.flat | (table.clip_frac > 0.005) | (table.n_beats == 0))
    return table


def merge_runs(table):
    runs = []
    for start, end in table.loc[table.noisy, ['start_s', 'end_s']].to_numpy():
        if runs and start - runs[-1][1] <= MERGE_GAP_S:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end])
    return [(s, e) for s, e in runs if e - s >= MIN_RUN_S]


def _overlap(run, others):
    s, e = run
    return sum(max(0.0, min(e, oe) - max(s, os)) for os, oe in others) / (e - s)


def label_runs(runs_by_lead):
    """A run overlapping the other lead's runs by at least half is reported once, as 'both'."""
    rows = []
    names = list(runs_by_lead)
    for lead in names:
        other = [n for n in names if n != lead]
        for run in runs_by_lead[lead]:
            overlap = _overlap(run, runs_by_lead[other[0]]) if other else 0.0
            if overlap >= 0.5 and lead != names[0]:
                continue
            rows.append({'lead': lead, 'start_s': run[0], 'end_s': run[1],
                         'attribution': 'both' if overlap >= 0.5 else f'ch{lead[-1]}_only'})
    return rows


def find_noise(path):
    """Noise runs on the export's sample-index clock, plus the span that was scored."""
    leads, fs, offset_s, end_s = load_leads(path)
    runs = {}
    for lead, raw in leads.items():
        table = score_lead(raw, fs)
        runs[lead] = merge_runs(table) if len(table) else []
    table = pd.DataFrame(label_runs(runs), columns=['lead', 'start_s', 'end_s', 'attribution'])
    table[['start_s', 'end_s']] += offset_s
    return table, (offset_s, end_s)
