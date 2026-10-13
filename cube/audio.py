"""Waveform normalisation: any shape / sample rate -> mono float64 @ 16 kHz."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from .frontend import SAMPLE_RATE


def to_mono_16k(waveform, sr: int) -> np.ndarray:
    """Convert a waveform to a 1-D float64 array at 16 kHz.

    `waveform` may be a numpy array or a torch tensor: 1-D (samples,) or 2-D (channels, samples), which is
    mixed down to mono by averaging. Audio at other sample rates is resampled with soxr (HQ).
    """
    if hasattr(waveform, "detach"):                      # torch.Tensor
        waveform = waveform.detach().cpu().numpy()
    y = np.asarray(waveform, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(0)
    elif y.ndim != 1:
        raise ValueError(f"expected a 1-D or (channels, samples) waveform, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("waveform contains NaN or Inf")
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"invalid sample rate {sr}")
    if sr != SAMPLE_RATE:
        import soxr
        y = np.asarray(soxr.resample(y, sr, SAMPLE_RATE), dtype=np.float64)
    return y


def load_audio(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Read an audio file with soundfile -> (mono float64 samples, sample rate)."""
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float64", always_2d=True)    # (samples, channels)
    return y.mean(1), int(sr)
