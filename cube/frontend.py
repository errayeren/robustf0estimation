"""Log-mel front end and frame count."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16000
HOP = 160                 # 10 ms at 16 kHz


def n_frames(n_samples: int) -> int:
    """Number of 10 ms output frames for `n_samples` samples of 16 kHz audio.

    One frame per 160 samples; a trailing partial hop of at least 156 samples also counts as a frame.
    This is the frame count the models were trained and evaluated with.
    """
    return (int(n_samples) + 4) // HOP


class LogMel(nn.Module):
    """(B, L) waveform -> (B, T, n_mels) natural-log power mel spectrogram (centred frames)."""

    def __init__(self, sr: int = SAMPLE_RATE, n_fft: int = 1024, hop: int = HOP, n_mels: int = 256,
                 f_min: float = 30.0, f_max: float = 8000.0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            f_min=f_min, f_max=f_max, power=2.0, center=True)
        # Centred frames reflect-pad n_fft // 2 samples on each side, which needs a longer input.
        self._min_len = n_fft // 2 + 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[-1] < self._min_len:
            audio = F.pad(audio, (0, self._min_len - audio.shape[-1]))
        m = self.mel(audio)                       # (B, n_mels, T)
        m = torch.log(m + 1e-6)
        return m.transpose(1, 2)                  # (B, T, n_mels)
