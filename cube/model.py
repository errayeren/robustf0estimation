"""Public API: `CUBE.from_pretrained(...)` and `CUBE.predict(...)`."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from .audio import to_mono_16k
from .config import CubeConfig
from .decoding import N_BINS, decode
from .frontend import HOP, SAMPLE_RATE, n_frames
from .network import CUBENet

DEFAULT_TAU = 0.12
MODEL_NAMES = ("cube-l", "cube-m", "cube-s", "cube-t")

# Audio longer than this is run in overlapping windows to bound memory (see `CUBE._logits_16k`).
_MAX_WHOLE_S = 60.0
_CHUNK_S = 30.0
_CONTEXT_FRAMES = 320            # context on each side of a window: wider than the network's receptive field,
                                 # and a multiple of the U-Net's 16-frame stride


def _search_dirs():
    env = os.environ.get("CUBE_WEIGHTS_DIR")
    if env:
        yield Path(env).expanduser()
    yield Path(__file__).resolve().parent.parent / "weights"
    yield Path.cwd() / "weights"
    yield Path.home() / ".cache" / "cube"


def _find_weights(path_or_name: Union[str, Path]):
    """-> (weights file, config file). Accepts a weights file, a directory holding one, or a model name."""
    p = Path(path_or_name).expanduser()
    if p.is_file():
        return p, p.with_suffix(".json")
    if p.is_dir():
        cands = sorted(list(p.glob("*.safetensors")) + list(p.glob("*.pt")))
        if len(cands) != 1:
            raise FileNotFoundError(f"expected exactly one weights file in {p}, found {len(cands)}")
        return cands[0], cands[0].with_suffix(".json")
    name = str(path_or_name).lower()
    if name in MODEL_NAMES:
        tried = []
        for d in _search_dirs():
            for ext in (".safetensors", ".pt"):
                f = d / f"{name}{ext}"
                tried.append(str(f))
                if f.is_file():
                    return f, f.with_suffix(".json")
        raise FileNotFoundError(
            f"weights for '{name}' not found. Looked for:\n  " + "\n  ".join(tried) +
            "\nPlace the weight and config files in ./weights, or set CUBE_WEIGHTS_DIR, "
            "or pass a path to the weights file.")
    raise FileNotFoundError(f"'{path_or_name}' is neither a file, a directory, nor one of {MODEL_NAMES}")


def _read_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:                                    # pragma: no cover
            raise ImportError("loading .safetensors weights requires `pip install safetensors`") from e
        return load_file(str(path), device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state dict")
    return state


class CUBE:
    """CUBE F0 estimator.

    >>> model = CUBE.from_pretrained("cube-l")
    >>> out = model.predict(waveform, sr)            # any sample rate, mono or (channels, samples)
    >>> out["time_s"], out["f0_hz"], out["confidence"], out["voiced"]
    """

    def __init__(self, net: CUBENet, name: Optional[str] = None, device: Union[str, torch.device] = "cpu"):
        self.net = net.eval().requires_grad_(False)
        self.config: CubeConfig = net.cfg
        self.name = name
        self.device = torch.device(device)
        self.net.to(self.device)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, path_or_name: Union[str, Path], device: Optional[Union[str, torch.device]] = None,
                        config: Optional[Union[str, Path, CubeConfig]] = None) -> "CUBE":
        """Load a model from a weights file (`.pt` / `.safetensors`), a directory, or a name
        (`cube-l`, `cube-m`, `cube-s`, `cube-t`) resolved in `$CUBE_WEIGHTS_DIR`, `./weights`, or
        `~/.cache/cube`. The architecture config is read from the JSON file next to the weights unless
        `config` is given."""
        wfile, cfile = _find_weights(path_or_name)
        if isinstance(config, CubeConfig):
            cfg = config
        else:
            cfile = Path(config) if config is not None else cfile
            if not cfile.is_file():
                raise FileNotFoundError(f"config file not found: {cfile}")
            cfg = CubeConfig.load(cfile)
        net = CUBENet(cfg)
        net.load_weights(_read_state(wfile))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(net, name=wfile.stem, device=device)

    # ---------------------------------------------------------------- inference
    @torch.inference_mode()
    def _logits_whole(self, y: np.ndarray, t: int) -> torch.Tensor:
        audio = torch.from_numpy(np.asarray(y[: t * HOP], np.float32))[None].to(self.device)
        return self.net(audio, t)[0]                                 # (T, 360), fp32

    @torch.inference_mode()
    def _logits_16k(self, y: np.ndarray) -> torch.Tensor:
        """Pitch logits (T, 360) for a mono 16 kHz float64 waveform.

        Clips up to a minute are processed whole, exactly. Longer clips are cut into 30 s windows with
        3.2 s of context on each side (a multiple of the U-Net's 16-frame stride, so every window sees the
        same pooling phase as the whole clip). Only the context, which is thrown away, differs from the
        whole-clip computation.
        """
        t = n_frames(len(y))
        if t == 0:
            return torch.zeros((0, N_BINS))
        if len(y) <= _MAX_WHOLE_S * SAMPLE_RATE:
            return self._logits_whole(y, t).cpu()
        core = int(_CHUNK_S * 100) // 16 * 16                        # frames per window, multiple of 16
        out = torch.zeros((t, N_BINS))
        for s in range(0, t, core):
            e = min(t, s + core)
            a = max(0, s - _CONTEXT_FRAMES)
            b = min(t, e + _CONTEXT_FRAMES)
            out[s:e] = self._logits_whole(y[a * HOP: b * HOP], b - a)[s - a: e - a].cpu()
        return out

    def logits(self, waveform, sr: int) -> np.ndarray:
        """Raw pitch logits, shape (n_frames, 360), float32."""
        y = to_mono_16k(waveform, sr)
        return self._logits_16k(y).numpy()

    def predict(self, waveform, sr: int, tau: float = DEFAULT_TAU) -> Dict[str, np.ndarray]:
        """Estimate F0 for a waveform.

        Args:
            waveform: 1-D samples, or (channels, samples); numpy array or torch tensor.
            sr: sample rate in Hz. Audio that is not 16 kHz is resampled.
            tau: voicing threshold on the confidence (peak salience). A frame is voiced iff confidence > tau.

        Returns a dict of equal-length arrays, one entry per 10 ms frame:
            time_s      frame centre times in seconds (0, 0.01, 0.02, ...)
            f0_hz       F0 in Hz; 0 on unvoiced frames
            confidence  peak salience in [0, 1] (float32)
            voiced      bool, confidence > tau
        """
        y = to_mono_16k(waveform, sr)
        logits = self._logits_16k(y)
        sal = torch.sigmoid(logits).numpy()                          # (T, 360) float32
        f0, _ = decode(sal, threshold=0.0)
        conf = sal.max(1)
        voiced = conf > tau
        f0 = np.where(voiced, f0, 0.0)
        return {
            "time_s": np.arange(len(conf), dtype=np.float64) * (HOP / SAMPLE_RATE),
            "f0_hz": f0,
            "confidence": conf,
            "voiced": voiced,
        }

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.net.parameters())
        return f"CUBE(name={self.name!r}, params={n / 1e6:.2f}M, device={self.device})"
