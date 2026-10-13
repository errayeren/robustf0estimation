"""Pitch-bin grid and salience decoding.

The head predicts 360 per-bin logits on a 20-cent grid (the CREPE / RMVPE / FCPE grid). Salience is the
per-bin sigmoid of the logits; F0 is a local weighted average of bin centres around the salience peak,
and the peak salience itself is the frame confidence.
"""
from __future__ import annotations

import numpy as np

N_BINS = 360
CENTS_PER_BIN = 7200.0 / N_BINS            # 20 cents
CENTS0 = 1997.3794084376191                # cents (re 10 Hz) of bin 0, i.e. ~31.7 Hz
DECODE_WIN = 4                             # local-average half-window, in bins

BIN_CENTS = CENTS0 + CENTS_PER_BIN * np.arange(N_BINS)    # (360,)


def cents_to_hz(cents, base: float = 10.0):
    """Cents above `base` Hz -> Hz."""
    return base * 2.0 ** (np.asarray(cents, dtype=np.float64) / 1200.0)


def hz_to_cents(hz, base: float = 10.0):
    """Hz -> cents above `base` Hz (-inf for non-positive input)."""
    hz = np.asarray(hz, dtype=np.float64)
    out = np.full_like(hz, -np.inf)
    pos = hz > 0
    out[pos] = 1200.0 * np.log2(hz[pos] / base)
    return out


def decode(salience, threshold: float = 0.0, win: int = DECODE_WIN):
    """(T, 360) salience -> (f0_hz (T,), confidence (T,)).

    F0 is the salience-weighted mean of the bin centres within +-`win` bins of the peak (computed in
    the cents domain, then converted to Hz). `confidence` is the peak salience. Frames whose confidence
    is <= `threshold` get f0 = 0. `CUBE.predict` calls this with threshold=0 and applies the voicing
    threshold `tau` afterwards.
    """
    sal = np.asarray(salience, dtype=np.float64)
    T = sal.shape[0]
    peak = sal.argmax(1)
    conf = sal[np.arange(T), peak]
    cents = np.zeros(T)
    for t in range(T):
        lo = max(0, peak[t] - win)
        hi = min(N_BINS, peak[t] + win + 1)
        w = sal[t, lo:hi]
        s = w.sum()
        cents[t] = (w * BIN_CENTS[lo:hi]).sum() / s if s > 0 else 0.0
    f0 = cents_to_hz(cents)
    f0[conf <= threshold] = 0.0
    return f0, conf
