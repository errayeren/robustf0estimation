"""CUBE: efficient and noise-robust F0 estimation for speech, singing voice and instrumental music."""
from .audio import load_audio, to_mono_16k
from .config import CubeConfig
from .decoding import decode
from .frontend import n_frames
from .model import CUBE, DEFAULT_TAU, MODEL_NAMES

__version__ = "0.1.0"
__all__ = ["CUBE", "CubeConfig", "DEFAULT_TAU", "MODEL_NAMES", "decode", "load_audio", "n_frames",
           "to_mono_16k", "__version__"]
