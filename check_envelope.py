#!/usr/bin/env python3
"""Plot an envelope time window with a sentence mask overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FS = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a time window of an envelope model with a semi-transparent "
            "sentence mask overlay to highlight in-sentence periods."
        )
    )
    parser.add_argument(
        "--subject",
        "-s",
        required=True,
        help=(
            "Subject identifier (e.g. 1, 01, sub-01). Automatically resolves "
            "paths within derivatives/ for the envelope and sentence mask."
        ),
    )
    parser.add_argument(
        "--window",
        "-w",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        required=True,
        help="Time window in seconds to display, e.g. --window 1 3.5.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Optional output path or filename. If omitted, the figure is saved in "
            "the subject's envelope folder with a window-specific suffix."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the plot.",
    )
    return parser.parse_args()


def format_subject_label(subject_argument: str) -> str:
    value = subject_argument.strip()
    if not value:
        raise SystemExit("Subject identifier cannot be empty.")

    lower_value = value.lower()
    if lower_value.startswith("sub-"):
        return lower_value

    trimmed = lower_value
    if trimmed.startswith("sub"):
        trimmed = trimmed[3:]
    trimmed = trimmed.lstrip("-_")
    if not trimmed:
        raise SystemExit(f"Could not parse subject identifier from '{value}'.")
    try:
        number = int(trimmed)
    except ValueError as exc:
        raise SystemExit(f"Subject '{value}' is not numeric.") from exc
    return f"sub-{number:02d}"


def build_subject_paths(subject_label: str) -> Tuple[Path, Path]:
    envelope_dir = (
        Path("derivatives") / "Models" / "envelope" / subject_label / "concatenated"
    )
    envelope_path = envelope_dir / f"{subject_label}_concatenated_envelope_100Hz.npy"
    mask_path = (
        Path("derivatives")
        / "preprocessed"
        / subject_label
        / "concatenated"
        / f"{subject_label}_concatenated_sentence_mask_100Hz.npy"
    )
    return envelope_path, mask_path


def load_array(path: Path, label: str) -> np.ndarray:
    try:
        arr = np.load(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Could not find {label} file: {path}") from exc
    if arr.ndim != 1:
        raise SystemExit(f"Expected 1D {label} array, got shape {arr.shape} from {path}")
    return arr


def validate_window(window: Iterable[float], duration: float) -> Tuple[float, float]:
    start, end = window
    if start >= end:
        raise SystemExit("Window start must be smaller than end.")
    if start < 0 or end > duration:
        raise SystemExit(
            f"Window {window} falls outside of available duration ({duration:.2f}s)."
        )
    return float(start), float(end)


def format_window_suffix(start: float, end: float) -> str:
    def fmt(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)}"
        text = format(value, "g")
        return text.replace("-", "m").replace(".", "p")

    return f"{fmt(start)}_{fmt(end)}"


def resolve_output_path(user_path: Path | None, default_dir: Path, default_name: str) -> Path:
    if user_path is None:
        return default_dir / default_name
    if user_path.is_absolute():
        return user_path
    if user_path.parent == Path('.'):
        return default_dir / user_path.name
    return user_path


def mask_regions(mask: np.ndarray, times: np.ndarray) -> List[Tuple[float, float]]:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return []
    mask_int = mask_bool.astype(int)
    starts = np.where(np.diff(np.concatenate(([0], mask_int))) == 1)[0]
    ends = np.where(np.diff(np.concatenate((mask_int, [0]))) == -1)[0]
    regions: List[Tuple[float, float]] = []
    dt = times[1] - times[0] if len(times) > 1 else 0.0
    for start_idx, end_idx in zip(starts, ends):
        start_t = times[start_idx]
        end_t = times[min(end_idx, len(times) - 1)] + dt
        regions.append((start_t, end_t))
    return regions


def plot_window(
    envelope: np.ndarray,
    mask: np.ndarray,
    fs: float,
    window: Tuple[float, float],
    title: str | None = None,
) -> plt.Figure:
    start, end = window
    start_idx = int(np.floor(start * fs))
    end_idx = int(np.ceil(end * fs))
    time = np.arange(len(envelope)) / fs
    window_time = time[start_idx:end_idx]
    window_env = envelope[start_idx:end_idx]
    window_mask = mask[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window_time, window_env, label="Envelope", color="tab:blue")

    y_min, y_max = np.nanmin(window_env), np.nanmax(window_env)
    padding = 0.05 * (y_max - y_min if y_max > y_min else 1)
    y_min -= padding
    y_max += padding

    for idx, (region_start, region_end) in enumerate(mask_regions(window_mask, window_time)):
        ax.axvspan(
            region_start,
            region_end,
            color="tab:orange",
            alpha=0.25,
            label="Sentence" if idx == 0 else None,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope amplitude")
    window_title = title or "Envelope window"
    ax.set_title(f"{window_title}: {start:.2f}s to {end:.2f}s")
    ax.set_xlim(start, end)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    subject_label = format_subject_label(args.subject)
    envelope_path, mask_path = build_subject_paths(subject_label)

    envelope = load_array(envelope_path, "envelope")
    mask = load_array(mask_path, "sentence mask")
    if len(envelope) != len(mask):
        min_len = min(len(envelope), len(mask))
        raise SystemExit(
            "Envelope and mask arrays must be the same length; "
            f"got {len(envelope)} and {len(mask)}."
        )
    fs = DEFAULT_FS
    window = tuple(args.window)
    duration = len(envelope) / fs
    start, end = validate_window(window, duration)

    fig = plot_window(envelope, mask, fs, (start, end), title=args.title)

    default_dir = envelope_path.parent
    suffix = format_window_suffix(start, end)
    default_filename = f"{envelope_path.stem}_{suffix}.png"
    output_path = resolve_output_path(args.output, default_dir, default_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
