"""CUBE network: log-mel -> 2-D U-Net -> local-only E-Branchformer blocks -> 360-bin pitch head."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CubeConfig
from .frontend import LogMel


# --------------------------------------------------------------------------------------------- U-Net
class ChanLN2d(nn.Module):
    """LayerNorm over channels at every (freq, time) position of a (B, C, M, T) tensor."""

    def __init__(self, ch: int):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FreqLN(nn.Module):
    """LayerNorm over the mel axis of a (B, C, M, T) tensor (normalises every frame's spectrum)."""

    def __init__(self, n_mels: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_mels)

    def forward(self, x):
        return self.ln(x.transpose(-1, -2)).transpose(-1, -2)


class ResBlock2D(nn.Module):
    """Two 3x3 convs with channel-LN and ReLU, a (1x1) residual skip, and an optional 2x2 max-pool."""

    def __init__(self, cin: int, cout: int, pool: bool = False):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = ChanLN2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.b2 = ChanLN2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool2d((2, 2)) if pool else nn.Identity()

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        r = F.relu(h + self.skip(x))
        return self.pool(r)


class UNet2D(nn.Module):
    """Encoder-decoder over (mel, time) with skip connections.

    `widths` = channels of [stem, down_1 .. down_depth]. Time is zero-padded to a multiple of 2**depth so
    every pool is an exact halving, and cropped back afterwards. The output keeps the full mel resolution,
    is flattened (channels x mel) and projected to `d_out` per frame by a 1x1 Conv1d.
    """

    def __init__(self, d_out: int, n_mels: int, widths: Sequence[int], n_bottleneck: int = 2):
        super().__init__()
        depth = len(widths) - 1
        self.in_norm = FreqLN(n_mels)
        self.e0 = ResBlock2D(1, widths[0])
        self.downs = nn.ModuleList([ResBlock2D(widths[i], widths[i + 1], pool=True) for i in range(depth)])
        self.bottleneck = nn.ModuleList([ResBlock2D(widths[-1], widths[-1]) for _ in range(n_bottleneck)])
        ups, c = [], widths[-1]
        for i in range(depth):
            skip = widths[depth - 1 - i]
            ups.append(ResBlock2D(c + skip, skip))
            c = skip
        self.ups = nn.ModuleList(ups)
        self.proj = nn.Conv1d(widths[0] * n_mels, d_out, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=(ref.shape[2], ref.shape[3]), mode="nearest")

    def forward(self, x):                                   # (B, 1, M, T) -> (B, T, d_out)
        x = self.in_norm(x)
        t0 = x.shape[-1]
        mult = 1 << len(self.downs)
        pad = (mult - t0 % mult) % mult
        if pad:
            x = F.pad(x, (0, pad))
        skips = [self.e0(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        h = skips[-1]
        for b in self.bottleneck:
            h = b(h)
        for i, u in enumerate(self.ups):
            skip = skips[len(skips) - 2 - i]
            h = u(torch.cat([self._up(h, skip), skip], 1))
        h = h[..., :t0]
        b, c, f, t = h.shape
        return self.proj(h.reshape(b, c * f, t)).transpose(1, 2)


# --------------------------------------------------------------------------------- temporal stack
class FFN(nn.Module):
    """Pre-LN feed-forward (GELU)."""

    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.ln(x))))


class ConvGatingMLP(nn.Module):
    """Convolutional gating MLP (cgMLP): one half of the expanded features gates the other through
    parallel depthwise convolutions over time."""

    def __init__(self, d: int, kernels: Sequence[int], dilations: Sequence[int]):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.ln_g = nn.LayerNorm(d)
        self.dws = nn.ModuleList([
            nn.Conv1d(d, d, k, padding=((k - 1) * dl) // 2, dilation=dl, groups=d)
            for k, dl in zip(kernels, dilations)])
        self.gate_proj = nn.Conv1d(d * len(kernels), d, 1)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):                                   # (B, T, d)
        x = F.gelu(self.fc1(self.ln(x)))
        a, b = x.chunk(2, -1)
        b = self.ln_g(b).transpose(1, 2)                    # (B, d, T)
        b = self.gate_proj(torch.cat([dw(b) for dw in self.dws], 1)).transpose(1, 2)
        return self.fc2(a * b)


class ConvBlock(nn.Module):
    """Local-only E-Branchformer block: the global self-attention branch is removed, leaving
    1/2 FFN -> cgMLP -> merge conv (k=3) -> 1/2 FFN -> LayerNorm, with residual connections."""

    def __init__(self, d: int, kernels: Sequence[int], dilations: Sequence[int], ffn_mult: int = 4):
        super().__init__()
        self.ffn1 = FFN(d, ffn_mult)
        self.cgmlp = ConvGatingMLP(d, kernels, dilations)
        self.merge = nn.Conv1d(d, d, 3, padding=1)
        self.ffn2 = FFN(d, ffn_mult)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        loc = self.cgmlp(x)
        x = x + self.merge(loc.transpose(1, 2)).transpose(1, 2)
        x = x + 0.5 * self.ffn2(x)
        return self.ln_out(x)


class ConvTemporal(nn.Module):
    """Stack of `ConvBlock`s followed by a linear output projection. Block `i` dilates only its widest
    kernel, by `layer_dilations[i]`."""

    def __init__(self, d: int, d_out: int, kernels: Sequence[int], layer_dilations: Sequence[int],
                 ffn_mult: int = 4):
        super().__init__()
        widest = max(kernels)
        self.blocks = nn.ModuleList([
            ConvBlock(d, kernels, tuple(ld if k == widest else 1 for k in kernels), ffn_mult)
            for ld in layer_dilations])
        self.out = nn.Linear(d, d_out)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.out(x)


# ------------------------------------------------------------------------------------------- model
def _match_t(x: torch.Tensor, t: int) -> torch.Tensor:
    """Crop or zero-pad (B, T', D) along time to exactly `t` frames."""
    tx = x.shape[1]
    if tx == t:
        return x
    if tx > t:
        return x[:, :t]
    return F.pad(x, (0, 0, 0, t - tx))


class CUBENet(nn.Module):
    """waveform (B, L) @ 16 kHz, number of frames T  ->  pitch logits (B, T, 360)."""

    def __init__(self, cfg: CubeConfig):
        super().__init__()
        self.cfg = cfg
        self.front = LogMel(cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels, cfg.f_min, cfg.f_max)
        self.unet = UNet2D(cfg.d_model, cfg.n_mels, cfg.unet_widths, cfg.unet_bottleneck_blocks)
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.temporal = ConvTemporal(cfg.d_model, 2 * cfg.d_model, cfg.temporal_kernels,
                                     cfg.temporal_dilations, cfg.ffn_mult)
        self.pitch_head = nn.Linear(2 * cfg.d_model, cfg.n_bins)

    def forward(self, audio: torch.Tensor, n_frames: int) -> torch.Tensor:
        m = self.front(audio.float())                       # (B, T', n_mels)
        x = m.transpose(1, 2).unsqueeze(1)                  # (B, 1, n_mels, T')
        x = _match_t(self.unet(x), n_frames)                # (B, T, d)
        x = self.in_proj(x)
        h = self.temporal(x)                                # (B, T, 2d)
        return self.pitch_head(h)                           # (B, T, 360)

    def load_weights(self, state: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=True)
