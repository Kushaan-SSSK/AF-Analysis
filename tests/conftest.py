import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FS = 1000


def mouse_ecg(seconds=60, irregular=False, seed=0):
    """About 600 bpm with a narrow QRS and small T wave; irregular=True gives AF-like RR."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    x = np.zeros(n)
    t = 0.05
    while t < seconds - 0.05:
        k = np.arange(-15, 40)
        idx = int(t * FS) + k
        ok = (idx >= 0) & (idx < n)
        qrs = np.exp(-0.5 * (k / 2.5) ** 2) - 0.25 * np.exp(-0.5 * ((k - 6) / 2.5) ** 2)
        x[idx[ok]] += (qrs + 0.15 * np.exp(-0.5 * ((k - 25) / 6.0) ** 2))[ok]
        t += rng.uniform(0.07, 0.16) if irregular else 0.100 + rng.normal(0, 0.003)
    return x + rng.normal(0, 0.01, n)


@pytest.fixture
def ecg():
    return mouse_ecg


def write_export(path, ch1, ch2, fs=FS):
    t = np.arange(len(ch1)) / fs
    with open(path, 'w') as f:
        f.write('Time\tRaw Ch 1\tRaw Ch 2\n')
        for row in zip(t, ch1, ch2):
            f.write('%.4f\t%.6f\t%.6f\n' % row)
    return path
