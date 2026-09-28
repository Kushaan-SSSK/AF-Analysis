from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from .denoise import denoise
from .features import ANALYSIS_END_S, BLOCK_S, block_starts, load_leads, window_features
from .model import MODES, mode_scores, predict
from .noise import find_noise


@dataclass
class RecordingAnalysis:
    path: Path
    windows: pd.DataFrame
    decisions: pd.DataFrame
    noise_runs: pd.DataFrame | None
    leads: dict
    scores: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    block_scores: pd.Series | None = None
    warnings: list = field(default_factory=list)

    def called_af(self, mode):
        return self.scores[mode] >= self.thresholds[mode]

    def af_like_blocks(self):
        """(start, end, score) of every 30 s block at or above the sensitive threshold."""
        threshold = self.thresholds['sensitive']
        return [(start, start + BLOCK_S, score) for start, score in self.block_scores.items() if score >= threshold]


def load_model(path):
    return joblib.load(path)


def analyze_recording(path, model=None, noise_step=True):
    path = Path(path)
    leads = load_leads(path)
    ch1 = leads.get('Raw Ch 1')
    if ch1 is None or ch1.duration_s < ANALYSIS_END_S:
        raise ValueError(f'{path.name}: recording is shorter than 40 s or has no usable Ch1 signal')
    notes = []
    runs, scored_span = None, None
    if noise_step:
        try:
            runs, scored_span = find_noise(path)
        except ValueError as error:
            notes.append(f'Noise check skipped: {error}')
    windows, decisions = [], []
    for start in block_starts(ch1.duration_s):
        features = {lead: window_features(signal.cleaned, signal.fs, start) for lead, signal in leads.items()}
        if features['Raw Ch 1'] is None:
            continue
        kept, decided = denoise(features['Raw Ch 1'], features.get('Raw Ch 2'), runs, scored_span, start)
        windows.append(kept.assign(block_start=start))
        decisions.append(decided.assign(block_start=start))
    result = RecordingAnalysis(path, pd.concat(windows, ignore_index=True), pd.concat(decisions, ignore_index=True),
                               runs, leads, warnings=notes)
    if model is not None:
        result.block_scores = predict(model, result.windows.assign(recording_id=result.windows.block_start))
        tagged = result.windows.assign(recording_id=path.name)
        result.scores = {mode: float(score.iloc[0]) for mode, score in mode_scores(model, tagged).items()}
        result.thresholds = {mode: model['thresholds'][mode] for mode in MODES}
    return result
