"""Command line: `python -m cube input.wav --model cube-l --out out.csv`."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .audio import load_audio
from .model import CUBE, DEFAULT_TAU, MODEL_NAMES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cube", description="Estimate F0 with CUBE.")
    p.add_argument("input", help="audio file (anything libsndfile reads: wav, flac, ogg, ...)")
    p.add_argument("--model", default="cube-l",
                   help=f"model name ({', '.join(MODEL_NAMES)}) or path to a weights file (default: cube-l)")
    p.add_argument("--out", default="-", help="output CSV path, or '-' for stdout (default)")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU,
                   help=f"voicing threshold on the confidence: voiced iff confidence > tau (default {DEFAULT_TAU})")
    p.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda (default: cuda if available)")
    p.add_argument("--threads", type=int, default=None, help="number of CPU threads for torch")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    try:
        y, sr = load_audio(args.input)
        model = CUBE.from_pretrained(args.model, device=args.device)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = model.predict(y, sr, tau=args.tau)
    rows = np.column_stack([res["time_s"], res["f0_hz"], res["confidence"].astype(np.float64),
                            res["voiced"].astype(np.float64)])
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8", newline="")
    try:
        np.savetxt(out, rows, delimiter=",", header="time_s,f0_hz,confidence,voiced", comments="",
                   fmt=["%.2f", "%.3f", "%.6f", "%d"])
    finally:
        if out is not sys.stdout:
            out.close()
    return 0
