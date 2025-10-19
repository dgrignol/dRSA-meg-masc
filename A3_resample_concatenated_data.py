"""Command-line helper for downsampling concatenated MEG derivatives.

The script resamples MEG sensor data and quality-control masks to a lower
sampling rate. Downsampling is done in a conservative manner: it uses
polyphase filtering for the continuous signals and logical AND across blocks
for the binary masks to ensure that previously excluded samples remain
masked. A companion plotting helper is also provided so specific time windows
can be inspected without repeating the resampling procedure.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

from functions.generic_helpers import read_repository_root
from fractions import Fraction


@dataclass
class ResampleConfig:
    """Container bundling all state needed for a resampling run."""

    subject_label: str
    original_rate: float
    target_rate: float
    factor: int
    concatenated_dir: Path
    output_suffix: str
    plot_start: float
    plot_end: float | None
    output_plot: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and populate the resampling options."""

    parser = argparse.ArgumentParser(
        description="Resample concatenated MEG derivatives to a lower sampling rate."
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
        help="Subject identifier (e.g., 'sub-01' or '1').",
    )
    parser.add_argument(
        "--original-rate",
        type=float,
        default=1000.0,
        help="Original sampling rate in Hz (default: 1000).",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=100.0,
        help="Target sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--plot-start",
        type=float,
        default=0.0,
        help="Start time (seconds) for the diagnostic plot window (default: 0).",
    )
    parser.add_argument(
        "--plot-end",
        type=float,
        default=60.0,
        help="End time (seconds) for the diagnostic plot window (default: 60).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where diagnostic plots are stored.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="100Hz",
        help="Suffix appended to resampled file names (default: 100Hz).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing resampled files.",
    )
    return parser.parse_args()


def normalise_subject(subject_arg: str) -> str:
    """Return a subject string in `sub-XX` format regardless of input."""

    if subject_arg.startswith("sub-"):
        return subject_arg
    return f"sub-{int(subject_arg):02d}"


def build_config(args: argparse.Namespace) -> ResampleConfig:
    """Derive absolute paths and resampling metadata from CLI arguments."""

    subject_label = normalise_subject(args.subject)
    # The script assumes the ratio between the original rate and the target rate
    # is an integer so the mask downsampling remains trivial.
    factor = int(round(args.original_rate / args.target_rate))
    if not np.isclose(args.original_rate / args.target_rate, factor):
        raise ValueError(
            f"Resampling factor must be integer: {args.original_rate} / {args.target_rate}"
        )
    repo_root = Path(read_repository_root())
    concatenated_dir = (
        repo_root
        / "derivatives"
        / "preprocessed"
        / subject_label
        / "concatenated"
    )
    if not concatenated_dir.exists():
        raise FileNotFoundError(f"Concatenated directory not found: {concatenated_dir}")

    if args.plot_end is not None and args.plot_end <= args.plot_start:
        raise ValueError("--plot-end must be greater than --plot-start.")

    output_plot = args.output_dir / f"{subject_label}_mask_resampling_diagnostic.png"

    return ResampleConfig(
        subject_label=subject_label,
        original_rate=args.original_rate,
        target_rate=args.target_rate,
        factor=factor,
        concatenated_dir=concatenated_dir,
        output_suffix=args.suffix,
        plot_start=args.plot_start,
        plot_end=args.plot_end,
        output_plot=output_plot,
    )


def resample_timeseries(data: np.ndarray, original_rate: float, target_rate: float) -> np.ndarray:
    """
    Resample a continuous signal using a polyphase FIR filter.

    Parameters
    ----------
    data:
        Array with timepoints on the last axis.
    original_rate:
        Sampling frequency (Hz) of the input data.
    target_rate:
        Desired sampling frequency (Hz) of the output data.
    """

    data = np.asarray(data, dtype=np.float32)
    ratio = Fraction(target_rate / original_rate).limit_denominator(1000)
    return resample_poly(data, up=ratio.numerator, down=ratio.denominator, axis=-1).astype(np.float32, copy=False)


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a boolean mask conservatively by AND-ing each block.

    The mask is treated as True only if every sample inside the block was True,
    so previously excluded samples remain excluded after resampling.
    """

    mask_bool = np.asarray(mask).astype(bool)
    remainder = mask_bool.shape[-1] % factor
    if remainder:
        raise ValueError(
            f"Mask length {mask_bool.shape[-1]} is not divisible by factor {factor}"
        )
    reshaped = mask_bool.reshape(-1, factor)
    # Require all samples in the window to be True to remain True in the downsampled mask.
    condensed = reshaped.all(axis=1)
    return condensed.astype(mask.dtype, copy=False)


def load_optional(path: Path) -> np.ndarray:
    """Load an .npy file, raising a clear error when it is missing."""

    if path.exists():
        return np.load(path)
    raise FileNotFoundError(f"Expected file not found: {path}")


def ensure_output(path: Path, overwrite: bool) -> None:
    """Create parent directories and guard against accidental overwrites."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def resample_subject(config: ResampleConfig, overwrite: bool) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Resample all derivatives for a single subject.

    Returns a dictionary mapping each derivative ID to a tuple containing the
    original and resampled arrays. The caller can use the original arrays for
    QC plots without hitting the disk a second time.
    """

    subject = config.subject_label
    suffix = config.output_suffix
    factor = config.factor
    out_info: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # MEG data
    meg_path = config.concatenated_dir / f"{subject}_concatenated_meg.npy"
    meg_out = config.concatenated_dir / f"{subject}_concatenated_meg_{suffix}.npy"
    meg = load_optional(meg_path)
    resampled_meg = resample_timeseries(meg, config.original_rate, config.target_rate)
    ensure_output(meg_out, overwrite)
    np.save(meg_out, resampled_meg)
    out_info["meg"] = (meg, resampled_meg)

    # Masks
    sentence_mask_path = config.concatenated_dir / f"{subject}_concatenated_sentence_mask.npy"
    sentence_mask_out = config.concatenated_dir / f"{subject}_concatenated_sentence_mask_{suffix}.npy"
    sentence_mask = np.load(sentence_mask_path)
    resampled_sentence_mask = downsample_mask(sentence_mask, factor)
    ensure_output(sentence_mask_out, overwrite)
    np.save(sentence_mask_out, resampled_sentence_mask)
    out_info["sentence_mask"] = (sentence_mask, resampled_sentence_mask)

    boundary_mask_path = config.concatenated_dir / f"{subject}_concatenation_boundaries_mask.npy"
    boundary_mask_out = config.concatenated_dir / f"{subject}_concatenation_boundaries_mask_{suffix}.npy"
    boundary_mask = np.load(boundary_mask_path)
    resampled_boundary_mask = downsample_mask(boundary_mask, factor)
    ensure_output(boundary_mask_out, overwrite)
    np.save(boundary_mask_out, resampled_boundary_mask)
    out_info["boundary_mask"] = (boundary_mask, resampled_boundary_mask)

    return out_info


def _extract_segment(
    mask: np.ndarray,
    rate: float,
    start_sec: float,
    end_sec: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    total_points = mask.shape[-1]
    if end_sec is None:
        end_idx = total_points
    else:
        end_idx = min(total_points, int(round(end_sec * rate)))
    start_idx = max(0, int(round(start_sec * rate)))
    segment = mask[start_idx:end_idx]
    time = np.arange(segment.shape[-1]) / rate + start_idx / rate
    return time, segment


def plot_masks_comparison(
    sentence_original: np.ndarray,
    sentence_resampled: np.ndarray,
    boundary_original: np.ndarray,
    boundary_resampled: np.ndarray,
    original_rate: float,
    target_rate: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    output_path: Path | None = None,
    title_prefix: str = "",
) -> None:
    """
    Draw a comparison plot for the original and resampled masks.

    Parameters
    ----------
    sentence_original / sentence_resampled:
        1-D arrays containing the sentence mask before and after resampling.
    boundary_original / boundary_resampled:
        1-D arrays marking concatenation boundaries before and after resampling.
    original_rate / target_rate:
        Sampling rates (Hz) associated with the original and resampled data.
    start_sec / end_sec:
        Interval to visualise. `end_sec=None` means "plot until the end".
    output_path:
        Optional path to write the figure; when None the plot is shown interactively.
    title_prefix:
        String prepended to the subplot titles (usually the subject label).
    """

    time_sentence_orig, sentence_orig = _extract_segment(sentence_original, original_rate, start_sec, end_sec)
    time_boundary_orig, boundary_orig = _extract_segment(boundary_original, original_rate, start_sec, end_sec)
    time_sentence_res, sentence_res = _extract_segment(sentence_resampled, target_rate, start_sec, end_sec)
    time_boundary_res, boundary_res = _extract_segment(boundary_resampled, target_rate, start_sec, end_sec)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].step(
        time_sentence_orig,
        sentence_orig,
        where="post",
        label=f"Sentence mask ({original_rate:.0f} Hz)",
        linewidth=1.0,
    )
    axes[0].step(
        time_boundary_orig,
        boundary_orig,
        where="post",
        label="Concatenation mask",
        linewidth=1.0,
    )
    axes[0].set_ylabel("Mask value")
    axes[0].set_title(f"{title_prefix}Original masks [{start_sec}s – {end_sec if end_sec is not None else 'end'}s]")
    axes[0].legend()
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].step(
        time_sentence_res,
        sentence_res,
        where="post",
        label=f"Sentence mask ({target_rate:.0f} Hz)",
        linewidth=1.0,
    )
    axes[1].step(
        time_boundary_res,
        boundary_res,
        where="post",
        label="Concatenation mask",
        linewidth=1.0,
    )
    axes[1].set_ylabel("Mask value")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"{title_prefix}Resampled masks [{start_sec}s – {end_sec if end_sec is not None else 'end'}s]")
    axes[1].legend()
    axes[1].set_ylim(-0.1, 1.1)

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """CLI entry point: parse arguments, resample derivatives, emit plots."""

    args = parse_args()
    config = build_config(args)
    resampled_data = resample_subject(config, overwrite=args.overwrite)

    plot_masks_comparison(
        sentence_original=resampled_data["sentence_mask"][0],
        sentence_resampled=resampled_data["sentence_mask"][1],
        boundary_original=resampled_data["boundary_mask"][0],
        boundary_resampled=resampled_data["boundary_mask"][1],
        original_rate=config.original_rate,
        target_rate=config.target_rate,
        start_sec=config.plot_start,
        end_sec=config.plot_end,
        output_path=config.output_plot,
        title_prefix=f"{config.subject_label} ",
    )

    print(f"Resampled files saved with suffix _{config.output_suffix} in {config.concatenated_dir}")
    print(f"Diagnostic plot saved to {config.output_plot}")


if __name__ == "__main__":
    main()
