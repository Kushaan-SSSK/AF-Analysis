"""RR-interval features from LabScribe/iWorx EKG exports.

A recording is analysed in 30 s blocks starting at 10 s (10-40 s, 40-70 s, ...).
Each block has eleven 10 s windows with a 2 s step. R-peaks are found on a
3 Hz high-passed, cleaned signal.
"""
import warnings
from dataclasses import dataclass

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ANALYSIS_START_S = 10
ANALYSIS_END_S = 40
BLOCK_S = ANALYSIS_END_S - ANALYSIS_START_S
WINDOW_S = 10
STEP_S = 2
TARGET_FS = 1000
PRR_THRESHOLD_PCT = 3.25
MOUSE_BPM_RANGE = (300, 900)

CORE = ['pRR_3.25', 'RR_Mean', 'RR_Min', 'RR_Max', 'HR_Mean', 'HR_Std',
        'SDNN', 'RMSSD', 'SDSD', 'pNN20', 'pNN50', 'pNN100',
        'SD1', 'SD2', 'SD1SD2', 'CSI', 'CVI', 'LFHF']
WINDOW_FEATURES = ['peak_count', *CORE, 'HR_Min', 'HR_Max', 'NN20', 'NN50', 'NN100',
                   'VLF', 'LF', 'HF', 'Area', 'qc_impossible_hr']
MODEL_FEATURES = [*WINDOW_FEATURES, *['prev_' + c for c in WINDOW_FEATURES],
                  *['next_' + c for c in WINDOW_FEATURES]]
NOT_CONTEXT = {'window_idx', 'start_time', 'end_time'}


@dataclass
class Lead:
    fs: int
    cleaned: np.ndarray
    display: np.ndarray

    @property
    def duration_s(self):
        return len(self.cleaned) / self.fs

    def peaks(self):
        """R-peaks over the whole recording, detected block by block."""
        found = []
        for start in range(0, len(self.cleaned), BLOCK_S * self.fs):
            chunk = self.cleaned[start:start + BLOCK_S * self.fs]
            if len(chunk) >= 3 * self.fs:
                found.append(start + detect_r_peaks(chunk, self.fs))
        return np.concatenate(found) if found else np.array([], int)

    def rr(self):
        """(time of each beat in s, interval to the previous beat in ms), within blocks."""
        peaks = self.peaks()
        same_block = np.diff(peaks // (BLOCK_S * self.fs)) == 0
        return peaks[1:][same_block] / self.fs, np.diff(peaks)[same_block] / self.fs * 1000


def read_export(path):
    frame = pd.read_csv(path, sep='\t', low_memory=False, on_bad_lines='skip')
    if 'Raw Ch 1' not in frame.columns:
        raise ValueError(f'{path}: no "Raw Ch 1" column')
    for column in ['Time', 'Raw Ch 1', 'Raw Ch 2']:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['Time', 'Raw Ch 1']).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f'{path}: no numeric samples')
    return frame


def decimate(frame, target_fs=TARGET_FS):
    fs = 1 / (frame.Time.iloc[1] - frame.Time.iloc[0])
    if fs > target_fs * 1.5:
        factor = int(round(fs / target_fs))
        if factor > 1:
            frame = frame.iloc[::factor].reset_index(drop=True)
    return frame, int(round(1 / (frame.Time.iloc[1] - frame.Time.iloc[0])))


def clean_ecg(signal, fs):
    b, a = butter(2, 3, 'highpass', fs=fs)
    return nk.ecg_clean(filtfilt(b, a, signal), sampling_rate=fs, method='biosppy')


def display_filter(signal, fs):
    """1-100 Hz band-pass that keeps the QRS shape for plotting."""
    b, a = butter(2, [1, min(100, 0.45 * fs)], 'bandpass', fs=fs)
    return filtfilt(b, a, signal)


def _threshold_peaks(signal, fs, window_s=0.1, upshift=0.035):
    moving_avg = pd.Series(signal).rolling(window=int(window_s * fs), center=True).mean().fillna(0).values
    above = signal > moving_avg + upshift * (np.max(signal) - np.min(signal))
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(signal))
    n = min(len(starts), len(ends))
    return np.array([s + np.argmax(signal[s:e]) for s, e in zip(starts[:n], ends[:n]) if e > s])


def _candidate_peaks(signal, fs):
    for inverted in (False, True):
        sig = -signal if inverted else signal
        try:
            yield nk.ecg_peaks(sig, sampling_rate=fs, method='neurokit')[1]['ECG_R_Peaks']
        except Exception:
            pass
        try:
            p5, p95 = np.percentile(sig, [5, 95])
            yield find_peaks(sig, distance=int(fs * 0.055), prominence=0.25 * (p95 - p5))[0]
        except Exception:
            pass
        try:
            yield _threshold_peaks(sig, fs)
        except Exception:
            pass


def detect_r_peaks(signal, fs):
    """Most regular candidate (lowest RR CV) whose rate is plausible for a mouse."""
    best, best_cv = np.array([]), np.inf
    for peaks in _candidate_peaks(signal, fs):
        if len(peaks) <= 10:
            continue
        rr = np.diff(peaks) / fs
        if MOUSE_BPM_RANGE[0] < 60 / np.mean(rr) < MOUSE_BPM_RANGE[1]:
            cv = np.std(rr) / np.mean(rr)
            if cv < best_cv:
                best, best_cv = peaks, cv
    return best


def rr_metrics(peaks, fs):
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks)
    rr_ms = rr / fs * 1000
    previous = rr[:-1]
    valid = previous > 0
    if not valid.any():
        return None
    relative_change = np.abs(np.diff(rr))[valid] / previous[valid] * 100
    hr = 60000 / rr_ms
    successive = np.abs(np.diff(rr_ms))
    m = {
        'pRR_3.25': np.mean(relative_change >= PRR_THRESHOLD_PCT) * 100,
        'RR_Mean': np.mean(rr_ms), 'RR_Min': np.min(rr_ms), 'RR_Max': np.max(rr_ms),
        'HR_Mean': np.mean(hr), 'HR_Min': np.min(hr), 'HR_Max': np.max(hr), 'HR_Std': np.std(hr, ddof=1),
        'SDNN': np.std(rr_ms, ddof=1), 'RMSSD': np.sqrt(np.mean(successive ** 2)),
        'SDSD': np.std(np.diff(rr_ms), ddof=1),
    }
    for ms in (20, 50, 100):
        m[f'NN{ms}'] = np.sum(successive > ms)
        m[f'pNN{ms}'] = m[f'NN{ms}'] / len(successive) * 100 if len(successive) else 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            hrv = nk.hrv(peaks, sampling_rate=fs, show=False)
            for name in ['VLF', 'LF', 'HF', 'LFHF', 'SD1', 'SD2', 'SD1SD2', 'CSI', 'CVI']:
                if 'HRV_' + name in hrv:
                    m[name] = hrv['HRV_' + name].values[0]
            if 'SD1' in m and 'SD2' in m:
                m['Area'] = np.pi * m['SD1'] * m['SD2']
        except Exception:
            pass
    return m


def add_context(windows):
    """Previous/next-window copies of every numeric feature, so a window sees its neighbours."""
    columns = [c for c in windows.columns if c not in NOT_CONTEXT and pd.api.types.is_numeric_dtype(windows[c])]
    for column in columns:
        windows['prev_' + column] = windows[column].shift(1).fillna(windows[column])
        windows['next_' + column] = windows[column].shift(-1).fillna(windows[column])
    return windows


def block_starts(duration_s):
    return list(range(ANALYSIS_START_S, int(duration_s) - BLOCK_S + 1, BLOCK_S))


def window_features(signal, fs, block_start=ANALYSIS_START_S):
    """Features for the 11 windows of one 30 s block of a cleaned lead (None if too short)."""
    start, end = block_start * fs, (block_start + BLOCK_S) * fs
    if len(signal) < end:
        return None
    peaks = detect_r_peaks(signal[start:end], fs)
    rows = []
    for idx, t0 in enumerate(np.arange(0, BLOCK_S - WINDOW_S + STEP_S / 2, STEP_S)):
        in_window = peaks[(peaks >= int(t0 * fs)) & (peaks < int((t0 + WINDOW_S) * fs))]
        row = {'window_idx': idx, 'start_time': t0, 'end_time': t0 + WINDOW_S, 'peak_count': len(in_window)}
        row.update(rr_metrics(in_window, fs) or {'pRR_3.25': 0})
        impossible_hr = pd.notna(row.get('HR_Mean', np.nan)) and row['HR_Mean'] > 900
        row['window_quality'] = 'low' if impossible_hr or row['peak_count'] < 3 else 'ok'
        row['qc_impossible_hr'] = bool(impossible_hr)
        rows.append(row)
    return add_context(pd.DataFrame(rows))


def load_leads(path, leads=('Raw Ch 1', 'Raw Ch 2')):
    """Cleaned and display-filtered signal for each lead in the export."""
    export = read_export(path)
    out = {}
    for lead in leads:
        if lead in export:
            frame, fs = decimate(export.dropna(subset=[lead]).reset_index(drop=True))
            raw = frame[lead].to_numpy()
            out[lead] = Lead(fs, clean_ecg(raw, fs), display_filter(raw, fs))
    return out
