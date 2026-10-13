"""Tests for the CUBE package.

`tests/data/reference_outputs.npz` holds a 4 s synthetic test signal and the F0 / confidence that the
original implementation, which produced the paper's numbers, computes for it with each model. The
model tests check that this installation reproduces them. They need the weight files (see README) and
are skipped for any model whose weights are not found.

    pip install ".[test]"
    pytest
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr
import torch

import cube
from cube import CUBE, decode, n_frames
from cube.decoding import BIN_CENTS, N_BINS, cents_to_hz

REPO = Path(__file__).resolve().parent.parent
REF = np.load(Path(__file__).parent / "data" / "reference_outputs.npz")
AUDIO = REF["audio"].astype(np.float64)
SR = int(REF["sample_rate"])
TAU = float(REF["tau"])

# Tolerances for comparing with the reference outputs. On the machine the reference was made on, the
# outputs are bit-identical; other CPUs and BLAS libraries change the last digits of float32 arithmetic.
CONF_TOL = 1e-4          # peak salience
CENTS_TOL = 1.0          # F0 on frames both voice
TAU_MARGIN = 1e-3        # voicing decisions are compared only where the reference is not this close to tau


def _weights_dir() -> Path:
    return Path(os.environ.get("CUBE_WEIGHTS_DIR", REPO / "weights"))


_MODELS = {}


def _model(name: str) -> CUBE:
    if not (_weights_dir() / f"{name}.pt").is_file() and not (_weights_dir() / f"{name}.safetensors").is_file():
        pytest.skip(f"weights for {name} not found in {_weights_dir()}")
    if name not in _MODELS:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        _MODELS[name] = CUBE.from_pretrained(_weights_dir() / f"{name}.pt", device="cpu")
    return _MODELS[name]


def _cents(a, b):
    return 1200.0 * np.abs(np.log2(a / b))


# ------------------------------------------------------------------------------------- no weights needed
def test_n_frames():
    assert [n_frames(n) for n in (0, 155, 156, 159, 160, 16000, 16155, 16156)] == [0, 0, 1, 1, 1, 100, 100, 101]


def test_decode_recovers_bin_centre():
    sal = np.zeros((3, N_BINS), np.float32)
    for t, k in enumerate((0, 100, N_BINS - 1)):
        sal[t, k] = 0.9
        if 0 < k < N_BINS - 1:
            sal[t, k - 1] = sal[t, k + 1] = 0.3                   # symmetric neighbours keep the centre
    f0, conf = decode(sal)
    assert np.allclose(f0, cents_to_hz(BIN_CENTS[[0, 100, N_BINS - 1]]))
    assert np.allclose(conf, 0.9)
    assert 31.6 < f0[0] < 31.8 and 2005 < f0[2] < 2006


def test_decode_threshold():
    sal = np.full((2, N_BINS), 0.01, np.float32)
    sal[0, 50], sal[1, 50] = 0.05, 0.5
    f0, _ = decode(sal, threshold=0.1)
    assert f0[0] == 0.0 and f0[1] > 0.0


def test_audio_input_handling():
    y = AUDIO[:SR]
    assert np.array_equal(cube.to_mono_16k(np.stack([y, y]), SR), y)
    assert np.array_equal(cube.to_mono_16k(torch.from_numpy(y), SR), y)
    assert abs(len(cube.to_mono_16k(soxr.resample(y, SR, 44100), 44100)) - len(y)) <= 1
    with pytest.raises(ValueError):
        cube.to_mono_16k(np.array([0.0, np.nan]), SR)
    with pytest.raises(ValueError):
        cube.to_mono_16k(np.zeros((2, 2, 2)), SR)


def test_config_roundtrip(tmp_path):
    cfg = cube.CubeConfig(d_model=128, unet_widths=(8, 16, 32, 32, 32))
    cfg.save(tmp_path / "c.json")
    assert cube.CubeConfig.load(tmp_path / "c.json") == cfg


def test_missing_weights_raise():
    with pytest.raises(FileNotFoundError):
        CUBE.from_pretrained("no_such_weights.pt", device="cpu")


# ------------------------------------------------------------------------------------------ models
@pytest.mark.parametrize("name", cube.MODEL_NAMES)
def test_matches_reference_outputs(name):
    out = _model(name).predict(AUDIO, SR, tau=TAU)
    ref_f0, ref_conf, ref_v = REF[f"{name}/f0_hz"], REF[f"{name}/confidence"], REF[f"{name}/voiced"]
    assert len(out["f0_hz"]) == len(ref_f0) == n_frames(len(AUDIO))
    assert np.allclose(out["time_s"], np.arange(len(ref_f0)) * 0.01)
    assert np.abs(out["confidence"] - ref_conf).max() <= CONF_TOL
    clear = np.abs(ref_conf - TAU) > TAU_MARGIN
    assert np.array_equal(out["voiced"][clear], ref_v[clear])
    both = out["voiced"] & ref_v
    assert both.sum() > 0.5 * ref_v.sum()
    assert _cents(out["f0_hz"][both], ref_f0[both]).max() <= CENTS_TOL
    assert np.all(out["f0_hz"][~out["voiced"]] == 0.0)


@pytest.mark.parametrize("name", cube.MODEL_NAMES)
def test_tracks_synthetic_f0(name):
    """Sanity check against the known F0 of the synthetic signal: a glide and a sung tone with vibrato."""
    out = _model(name).predict(AUDIO, SR, tau=TAU)
    true = REF["true_f0_hz"]
    ref_v = true > 0
    both = ref_v & out["voiced"]
    rpa = (_cents(out["f0_hz"][both], true[both]) <= 50).sum() / ref_v.sum()
    assert rpa > 0.95, rpa
    assert (out["voiced"] == ref_v).mean() > 0.95


def test_resampled_input_matches_16k():
    model = _model("cube-t")
    base = model.predict(AUDIO, SR)
    for sr in (8000, 22050, 44100, 48000):
        out = model.predict(soxr.resample(AUDIO, SR, sr), sr)
        n = min(len(out["f0_hz"]), len(base["f0_hz"]))
        both = out["voiced"][:n] & base["voiced"][:n]
        assert abs(len(out["f0_hz"]) - len(base["f0_hz"])) <= 1
        assert (out["voiced"][:n] == base["voiced"][:n]).mean() > 0.95, sr
        assert np.median(_cents(out["f0_hz"][:n][both], base["f0_hz"][:n][both])) < 5.0, sr


def test_long_audio_windowing():
    """Audio over a minute is processed in overlapping windows; the result matches a single pass."""
    model = _model("cube-t")
    y = np.tile(AUDIO, 17)[: 66 * SR + 157]                        # 66 s
    t = n_frames(len(y))
    whole = model._logits_whole(y, t).numpy()
    windowed = model.logits(y, SR)
    assert windowed.shape == whole.shape == (t, N_BINS)
    assert np.abs(windowed - whole).max() < 1e-3


def test_cli(tmp_path):
    name = "cube-t"
    model = _model(name)
    wav, csv = tmp_path / "in.wav", tmp_path / "out.csv"
    sf.write(wav, AUDIO.astype(np.float32), SR, subtype="FLOAT")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    r = subprocess.run([sys.executable, "-m", "cube", str(wav), "--model", str(_weights_dir() / f"{name}.pt"),
                        "--out", str(csv), "--device", "cpu"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert csv.read_text().splitlines()[0] == "time_s,f0_hz,confidence,voiced"
    arr = np.loadtxt(csv, delimiter=",", skiprows=1)
    out = model.predict(AUDIO, SR)
    assert arr.shape == (len(out["f0_hz"]), 4)
    assert np.allclose(arr[:, 1], out["f0_hz"], atol=1e-3)
    assert np.array_equal(arr[:, 3].astype(bool), out["voiced"])
