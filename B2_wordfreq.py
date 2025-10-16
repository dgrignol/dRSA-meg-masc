#!/usr/bin/env python3
"""
Construct word-frequency feature trajectories aligned to the MEG timeline.

For every preprocessed MEG run, the script:
1. Reads the companion BIDS events table to locate word annotations.
2. Computes log word frequencies (Zipf scale) using the ``wordfreq`` library.
3. Expands each word's value across all MEG samples spanned by that word.

Outputs are saved under ``derivatives/Models/wordfreq`` mirroring the session/task
hierarchy. Each run produces:
    - ``*_wordfreq_megfs.npy``: array with shape ``(1, n_timepoints)``.
    - ``*_wordfreq_metadata.json``: provenance and basic quality metrics.

Optionally, subject-level concatenated arrays and a resampled (e.g., 100 Hz) variant
are generated when the requisite concatenation metadata is available.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly
from wordfreq import zipf_frequency

from functions.generic_helpers import read_repository_root
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RunDescriptor:
    subject: str  # sub-XX
    session: str  # ses-X
    task: str  # task-X
    metadata_path: Path
    events_path: Path
    sfreq: float
    n_samples: int


@dataclass
class RunProduct:
    descriptor: RunDescriptor
    array_path: Path
    metadata_path: Path
    n_words: int
    nonzero_samples: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalise_subject(label: str) -> str:
    label = label.strip()
    if label.startswith("sub-"):
        return label
    return f"sub-{int(label):02d}"


def normalise_session(label: str) -> str:
    label = label.strip()
    if label.startswith("ses-"):
        return label
    return f"ses-{label}"


def normalise_task(label: str) -> str:
    label = label.strip()
    if label.startswith("task-"):
        return label
    return f"task-{label}"


def iter_preprocessed_runs(
    preproc_root: Path, bids_root: Path, subjects: Optional[Sequence[str]]
) -> Iterable[RunDescriptor]:
    subject_filter = None
    if subjects:
        subject_filter = {normalise_subject(s) for s in subjects}

    for subj_dir in sorted(preproc_root.glob("sub-*")):
        subject = subj_dir.name
        if subject_filter and subject not in subject_filter:
            continue

        for meta_path in sorted(
            subj_dir.glob("ses-*/task-*/sub-*_metadata.json")
        ):
            with meta_path.open("r") as fh:
                meta = json.load(fh)

            session = normalise_session(meta["session"])
            task = normalise_task(meta["task"])
            events_path = (
                bids_root
                / subject
                / session
                / "meg"
                / f"{subject}_{session}_{task}_events.tsv"
            )

            if not events_path.exists():
                LOGGER.warning("Events TSV missing for %s; skipping.", meta_path)
                continue

            sfreq = float(meta["sfreq"])
            n_samples = int(meta["n_samples"])

            yield RunDescriptor(
                subject=subject,
                session=session,
                task=task,
                metadata_path=meta_path,
                events_path=events_path,
                sfreq=sfreq,
                n_samples=n_samples,
            )


def load_word_events(events_path: Path) -> List[Dict]:
    events: List[Dict] = []
    with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                trial_info = ast.literal_eval(row["trial_type"])
            except (ValueError, SyntaxError):
                LOGGER.debug("Unable to parse trial_type in %s", events_path)
                continue

            if trial_info.get("kind") != "word":
                continue
            if float(trial_info.get("pronounced", 1.0)) == 0.0:
                continue

            try:
                onset_sample = int(row["sample"])
                duration = float(row["duration"])
            except (TypeError, ValueError):
                continue

            events.append(
                {
                    "word": trial_info.get("word", ""),
                    "sample": onset_sample,
                    "duration": duration,
                }
            )
    return events


def build_wordfreq_series(
    descriptor: RunDescriptor, events: Sequence[Dict]
) -> Tuple[np.ndarray, int, int]:
    data = np.zeros((1, descriptor.n_samples), dtype=np.float32)
    n_words = 0
    nonzero_samples = 0

    for event in events:
        word = event["word"]
        if not word:
            continue

        freq = float(zipf_frequency(word, "en"))
        start = max(0, event["sample"])
        length = int(round(event["duration"] * descriptor.sfreq))
        if length <= 0:
            length = 1
        stop = min(descriptor.n_samples, start + length)
        if stop <= start:
            continue

        data[0, start:stop] = freq
        n_words += 1
        if freq != 0.0:
            nonzero_samples += stop - start

    return data, n_words, nonzero_samples


def save_run_product(
    descriptor: RunDescriptor,
    models_root: Path,
    overwrite: bool,
) -> Optional[RunProduct]:
    out_dir = models_root / descriptor.subject / descriptor.session / descriptor.task
    out_dir.mkdir(parents=True, exist_ok=True)

    array_path = out_dir / f"{descriptor.subject}_{descriptor.session}_{descriptor.task}_wordfreq_megfs.npy"
    meta_path = out_dir / f"{descriptor.subject}_{descriptor.session}_{descriptor.task}_wordfreq_metadata.json"

    if array_path.exists() and not overwrite:
        LOGGER.info("Word-frequency model exists for %s; skipping.", array_path)
        existing = np.load(array_path)
        if existing.shape[-1] != descriptor.n_samples:
            LOGGER.warning(
                "Existing array length (%d) does not match metadata (%d) for %s.",
                existing.shape[-1],
                descriptor.n_samples,
                array_path,
            )
        nonzero = int(np.count_nonzero(existing))
        return RunProduct(
            descriptor=descriptor,
            array_path=array_path,
            metadata_path=meta_path,
            n_words=-1,
            nonzero_samples=nonzero,
        )

    events = load_word_events(descriptor.events_path)
    series, n_words, nonzero_samples = build_wordfreq_series(descriptor, events)
    np.save(array_path, series.astype(np.float32))

    metadata = {
        "subject": descriptor.subject,
        "session": descriptor.session,
        "task": descriptor.task,
        "sfreq": descriptor.sfreq,
        "n_samples": descriptor.n_samples,
        "n_words": n_words,
        "nonzero_samples": nonzero_samples,
        "events_path": str(descriptor.events_path),
        "source_metadata": str(descriptor.metadata_path),
        "feature_names": ["zipf_frequency_en"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    LOGGER.info(
        "Saved word-frequency model (%d words) to %s",
        n_words,
        array_path,
    )
    return RunProduct(
        descriptor=descriptor,
        array_path=array_path,
        metadata_path=meta_path,
        n_words=n_words,
        nonzero_samples=nonzero_samples,
    )


def plot_concatenated_wordfreq(
    data: np.ndarray,
    sfreq: float,
    output_path: Path,
    max_points: int = 20000,
) -> Path:
    """Plot a downsampled trajectory for quick inspection."""
    series = np.asarray(data, dtype=float)
    if series.ndim == 2:
        if series.shape[0] == 1:
            series = series[0]
        else:
            raise ValueError("Expected a single-feature array for plotting.")
    if series.ndim != 1:
        raise ValueError("Word-frequency trajectory must be 1D for plotting.")

    n_samples = series.size
    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    step = max(1, int(np.ceil(n_samples / max_points)))
    idx = np.arange(0, n_samples, step, dtype=int)
    times = idx / float(sfreq)
    values = series[idx]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, values, linewidth=0.8, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Zipf frequency")
    ax.set_title("Concatenated word-frequency trajectory")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def load_concatenation_order(subject_dir: Path) -> Optional[List[Dict]]:
    concat_meta = subject_dir / "concatenated" / f"{subject_dir.name}_concatenation_metadata.json"
    if not concat_meta.exists():
        LOGGER.warning("Concatenation metadata missing for %s; skipping subject-level merge.", subject_dir.name)
        return None
    with concat_meta.open("r") as fh:
        data = json.load(fh)
    return data.get("segments", [])


def concatenate_subject_runs(
    subject: str,
    run_products: Dict[Tuple[str, str], RunProduct],
    preproc_root: Path,
    models_root: Path,
    overwrite: bool,
    target_rate: Optional[float],
    plot: bool,
    plot_max_points: int,
) -> None:
    subject_dir = preproc_root / subject
    segments = load_concatenation_order(subject_dir)
    if not segments:
        return

    first_run = next(iter(run_products.values()))
    sfreq = first_run.descriptor.sfreq

    ordered_arrays: List[np.ndarray] = []
    segment_info: List[Dict] = []
    for segment in segments:
        session = segment["session"]
        task = segment["task"]
        key = (session, task)
        product = run_products.get(key)
        if product is None:
            LOGGER.warning(
                "Missing word-frequency data for %s %s %s; subject-level array incomplete.",
                subject,
                session,
                task,
            )
            return
        data = np.load(product.array_path)
        ordered_arrays.append(data)
        segment_info.append(
            {
                "session": session,
                "task": task,
                "samples": int(data.shape[-1]),
                "input_file": str(product.array_path),
            }
        )

    concatenated = np.concatenate(ordered_arrays, axis=1)
    out_dir = models_root / subject / "concatenated"
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_path = out_dir / f"{subject}_concatenated_wordfreq_megfs.npy"
    concat_meta_path = out_dir / f"{subject}_concatenated_wordfreq_metadata.json"

    if concat_path.exists() and not overwrite:
        LOGGER.info("Concatenated word-frequency array exists for %s; skipping.", subject)
        return

    np.save(concat_path, concatenated.astype(np.float32))
    concat_meta = {
        "subject": subject,
        "sfreq": sfreq,
        "samples": int(concatenated.shape[-1]),
        "segments": segment_info,
        "output_file": str(concat_path),
    }

    plot_path = None
    if plot:
        plot_path = plot_concatenated_wordfreq(
            data=concatenated,
            sfreq=sfreq,
            output_path=out_dir / f"{subject}_concatenated_wordfreq_plot.png",
            max_points=plot_max_points,
        )
        concat_meta["plot"] = str(plot_path)

    if target_rate:
        if target_rate <= 0:
            raise ValueError("target_rate must be positive when specified.")
        ratio = Fraction(target_rate / sfreq).limit_denominator(1000)
        approx = ratio.numerator / ratio.denominator
        if not np.isclose(approx, target_rate / sfreq, rtol=1e-6, atol=1e-12):
            raise ValueError(
                f"Unable to express resampling ratio for {sfreq}→{target_rate} Hz with rational approximation."
            )
        resampled = resample_poly(
            concatenated, up=ratio.numerator, down=ratio.denominator, axis=-1
        ).astype(np.float32)
        resampled_path = out_dir / f"{subject}_concatenated_wordfreq_{int(target_rate)}Hz.npy"
        np.save(resampled_path, resampled)
        concat_meta["resampled"] = {
            "target_rate": target_rate,
            "output_file": str(resampled_path),
            "resample_ratio": [ratio.numerator, ratio.denominator],
        }
        if plot:
            resampled_plot_path = plot_concatenated_wordfreq(
                data=resampled,
                sfreq=target_rate,
                output_path=out_dir / f"{subject}_concatenated_wordfreq_{int(target_rate)}Hz_plot.png",
                max_points=plot_max_points,
            )
            concat_meta.setdefault("resampled", {})["plot"] = str(resampled_plot_path)

    concat_meta_path.write_text(json.dumps(concat_meta, indent=2))
    LOGGER.info("Saved concatenated word-frequency arrays for %s", subject)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate word-frequency feature trajectories aligned with MEG timepoints."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional subset of subjects (e.g., 01 02 or sub-01 sub-02). Defaults to all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--no-concat",
        dest="concat",
        action="store_false",
        help="Disable subject-level concatenation.",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=None,
        help="Optional sampling rate (Hz) for resampling concatenated arrays (e.g., 100).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for console logging.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots for concatenated trajectories.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=20000,
        help="Maximum number of samples plotted (downsamples if longer).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    repo_root = read_repository_root()
    preproc_root = repo_root / "derivatives" / "preprocessed"
    models_root = repo_root / "derivatives" / "Models" / "wordfreq"
    models_root.mkdir(parents=True, exist_ok=True)
    bids_root = repo_root / "bids_anonym"

    subject_products: Dict[str, Dict[Tuple[str, str], RunProduct]] = {}
    processed_runs = 0

    for descriptor in iter_preprocessed_runs(preproc_root, bids_root, args.subjects):
        product = save_run_product(descriptor, models_root, overwrite=args.overwrite)
        if not product:
            continue
        processed_runs += 1
        subject_products.setdefault(descriptor.subject, {})[
            (descriptor.session, descriptor.task)
        ] = product

    LOGGER.info("Finished generating word-frequency models for %d runs.", processed_runs)

    if args.target_rate and not args.concat:
        LOGGER.warning(
            "--target-rate requested but concatenation disabled; resampling step will be skipped."
        )
    if args.plot and not args.concat:
        LOGGER.warning(
            "--plot requested but concatenation disabled; no plots will be generated."
        )

    if args.concat:
        for subject, runs in subject_products.items():
            concatenate_subject_runs(
                subject=subject,
                run_products=runs,
                preproc_root=preproc_root,
                models_root=models_root,
                overwrite=args.overwrite,
                target_rate=args.target_rate,
                plot=args.plot,
                plot_max_points=args.plot_max_points,
            )

    return 0 if processed_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
