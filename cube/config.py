"""Model configuration, stored as a small JSON file next to each weights file."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Tuple, Union

from .decoding import N_BINS


@dataclass(frozen=True)
class CubeConfig:
    # front end
    sample_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 1024
    n_mels: int = 256
    f_min: float = 30.0
    f_max: float = 8000.0
    # U-Net
    d_model: int = 512                                         # E-Branchformer width d
    unet_widths: Tuple[int, ...] = (32, 64, 128, 128, 128)     # stem, then one entry per level
    unet_bottleneck_blocks: int = 2
    # local-only E-Branchformer blocks
    temporal_kernels: Tuple[int, ...] = (5, 17, 49)            # depthwise kernels of the cgMLP gate
    temporal_dilations: Tuple[int, ...] = (1, 1, 2)            # dilation of the widest kernel, per block
    ffn_mult: int = 4
    # output
    n_bins: int = N_BINS

    def __post_init__(self):
        for f in ("unet_widths", "temporal_kernels", "temporal_dilations"):
            object.__setattr__(self, f, tuple(int(v) for v in getattr(self, f)))
        if self.sample_rate != 16000 or self.hop_length != 160:
            raise ValueError("CUBE operates on 16 kHz audio with a 160-sample (10 ms) hop")
        if self.n_bins != N_BINS:
            raise ValueError(f"the pitch grid has {N_BINS} bins, config says {self.n_bins}")
        if len(self.temporal_dilations) < 1 or len(self.unet_widths) < 2:
            raise ValueError("config describes an empty network")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CubeConfig":
        extra = set(d) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"unknown config keys: {sorted(extra)}")
        return cls(**dict(d))

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CubeConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
