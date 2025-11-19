#!/usr/bin/env python3
"""Plot two wordfreq model arrays stacked to inspect their alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FS = 100.0
DEFAULT_PATHS: Tuple[Path, Path] = (
    Path(
        "derivatives/Models/wordfreq/sub-01/concatenated/"
        "sub-01_concatenated_wordfreq_100Hz.npy"
    ),
    Path(
        "derivatives/Models/wordfreq/sub-02/concatenated/"
        "sub-02_concatenated_wordfreq_100Hz.npy"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot two wordfreq model outputs in a single figure to check their alignment."
        )
    )
    parser.add_argument(
        "--paths",
        "-p",
        nargs=2,
        type=Path,
        default=DEFAULT_PATHS,
        metavar=("PATH1", "PATH2"),
        help="Paths to the two .npy arrays to plot (default: sub-01 and sub-02).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS,
        help="Sampling rate in Hz for the time axis (default: 100).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to save the figure. If omitted, the plot is only shown.",
    )
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    try:
        arr = np.load(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Could not find array file: {path}") from exc
    arr = np.squeeze(arr)  # some stored arrays are shaped (1, N); squeeze to 1D
    if arr.ndim != 1:
        raise SystemExit(
            f"Expected 1D array from {path} after squeeze, got shape {arr.shape}."
        )
    return arr


def trim_to_shared_length(arrays: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    lengths = [len(arr) for arr in arrays]
    min_len = min(lengths)
    if any(length != min_len for length in lengths):
        print(f"Warning: arrays differ in length {lengths}; trimming to {min_len}.")
    trimmed = [arr[:min_len] for arr in arrays]
    return trimmed, min_len


def build_time_axis(n_points: int, fs: float) -> np.ndarray:
    return np.arange(n_points) / fs


def plot_wordfreq(
    arrays: Sequence[np.ndarray], labels: Iterable[str], fs: float
) -> plt.Figure:
    if len(arrays) != 2:
        raise SystemExit("This script currently expects exactly two arrays to compare.")
    trimmed_arrays, shared_len = trim_to_shared_length(arrays)
    time = build_time_axis(shared_len, fs)
    colors = ("tab:blue", "tab:orange")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, arr, label, color in zip(axes[:2], trimmed_arrays, labels, colors):
        ax.plot(time, arr, color=color, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)

    overlay_ax = axes[2]
    for arr, label, color in zip(trimmed_arrays, labels, colors):
        overlay_ax.plot(time, arr, color=color, linewidth=0.9, alpha=0.6, label=label)
    overlay_ax.set_ylabel("Overlay")
    overlay_ax.set_xlabel("Time (s)")
    overlay_ax.grid(True, alpha=0.2)
    overlay_ax.legend(loc="upper right")

    fig.suptitle("Wordfreq model alignment")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    arrays = [load_array(path) for path in args.paths]
    labels = [path.stem for path in args.paths]
    fig = plot_wordfreq(arrays, labels, fs=args.fs)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved figure to {args.output}")

    plt.show()


if __name__ == "__main__":
    main()
