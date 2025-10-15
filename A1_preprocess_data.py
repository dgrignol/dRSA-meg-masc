#!/usr/bin/env python3
"""Lightweight MEG-MASC preprocessing with sentence masks.

This script mirrors the minimal filtering performed in ``check_decoding.py``
while keeping every MEG sample intact. For each available subject/session/task
run in the local BIDS directory it:

* Selects MEG channels, loads the data, and applies a 0.5–30 Hz FIR filter.
* Builds a binary mask marking sentence blocks (1) versus everything else (0)
  based on the ``sequence_id`` annotations shipped with the dataset.
* Reconstructs continuous audio time courses aligned to both the MEG sampling
  grid and the native audio sampling rate.
* Stores the filtered MEG, the mask, both audio representations, and rich
  metadata under ``derivatives/preprocessed``.

No epoching, ICA, baseline correction, or clipping is applied.
Run ``python A1_preprocess_data.py --help`` for usage information.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import soundfile as sf
from mne_bids import BIDSPath, read_raw_bids
from scipy.signal import resample

LOGGER = logging.getLogger(__name__)


@dataclass
class SequenceSegment:
    """Structured representation of a contiguous annotation sequence."""

    sequence_id: float
    label: str
    onset: float
    offset: float
    story: str
    sound_path: Optional[str]
    audio_onset: Optional[float]
    audio_offset: Optional[float]

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class RunProduct:
    """Artifacts generated for a subject/session/task run."""

    meg_path: Path
    mask_path: Path
    audio_native_path: Path
    audio_meg_path: Path
    metadata_path: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_repository_root() -> Path:
    pointer = Path(__file__).parent / "data_path.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            "data_path.txt is missing. Create the file with the absolute path "
            "to the repository root."
        )
    root = Path(pointer.read_text().strip()).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    return root


def safe_literal_eval(item: str) -> dict:
    import ast

    try:
        parsed = ast.literal_eval(item)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot parse annotation description: {item}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"Annotation payload is not a dictionary: {parsed!r}")
    return parsed


def dataframe_from_annotations(raw: mne.io.BaseRaw) -> pd.DataFrame:
    rows: List[dict] = []
    for onset, duration, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        payload = safe_literal_eval(desc)
        payload.update(
            onset=float(onset),
            duration=float(duration),
            offset=float(onset + duration),
        )
        rows.append(payload)
    df = pd.DataFrame(rows)
    for column in ("condition", "kind", "sequence_id", "story", "sound", "start"):
        if column not in df.columns:
            df[column] = np.nan
    return df


def label_sequence(block: pd.DataFrame) -> str:
    condition_values = {
        str(val).lower() for val in block["condition"].dropna().unique()
    }
    sound_paths = {str(val).lower() for val in block["sound"].dropna().unique()}
    if any("question" in path for path in sound_paths):
        return "question"
    if any("wordlist" in path for path in sound_paths):
        return "word_list"
    if "word_list" in condition_values or "wordlist" in condition_values:
        return "word_list"
    if "pseudo_words" in condition_values or "pseudowords" in condition_values:
        return "pseudo_words"
    if "sentence" in condition_values:
        return "sentence"
    if condition_values:
        return "other"
    return "unknown"


def collect_sequence_segments(df: pd.DataFrame) -> List[SequenceSegment]:
    segments: List[SequenceSegment] = []
    if "sequence_id" not in df.columns:
        return segments
    df = df[df["sequence_id"].notna()].copy()
    if df.empty:
        return segments
    df["sequence_id"] = df["sequence_id"].astype(float)
    for seq_id, block in df.groupby("sequence_id", sort=True):
        label = label_sequence(block)
        onset = float(block["onset"].min())
        offset = float((block["onset"] + block["duration"]).max())
        story_values = block["story"].dropna().unique()
        story = str(story_values[0]) if len(story_values) else "unknown"
        sound_values = block["sound"].dropna().unique()
        sound_path = str(sound_values[0]) if len(sound_values) else None
        audio_onset = (
            float(block["start"].dropna().min()) if block["start"].notna().any() else None
        )
        audio_offset = (
            float((block["start"] + block["duration"]).dropna().max())
            if block["start"].notna().any()
            else None
        )
        segments.append(
            SequenceSegment(
                sequence_id=float(seq_id),
                label=label,
                onset=onset,
                offset=offset,
                story=story,
                sound_path=sound_path,
                audio_onset=audio_onset,
                audio_offset=audio_offset,
            )
        )
    return sorted(segments, key=lambda seg: seg.onset)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _downsample_for_plot(
    values: np.ndarray, max_points: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """Return down-sampled indices and values suitable for plotting."""
    n = len(values)
    if n <= max_points:
        idx = np.arange(n)
        return idx, values
    step = max(1, int(math.ceil(n / max_points)))
    idx = np.arange(0, n, step)
    return idx, values[idx]


def create_preprocessing_report(
    raw: mne.io.BaseRaw,
    mask: np.ndarray,
    metadata: dict,
    reports_root: Path,
    base_name: str,
) -> Path:
    """Generate an MNE report for the preprocessed run."""
    subject = metadata.get("subject", "unknown")
    session = metadata.get("session", "unknown")
    task = metadata.get("task", "unknown")

    report_dir = (
        reports_root
        / f"sub-{subject}"
        / (f"ses-{session}" if session is not None else "ses-unknown")
        / f"task-{task}"
    )
    ensure_dir(report_dir)

    report_path = report_dir / f"{base_name}_preprocessing_report.html"

    report = mne.Report(title=f"Preprocessing summary: sub-{subject} ses-{session} task-{task}")
    report.add_raw(raw, title="Filtered MEG", psd=False, tags=("meg", "filtered"))

    idx, mask_values = _downsample_for_plot(mask.astype(float))
    times = idx / raw.info["sfreq"]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.step(times, mask_values, where="post")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sentence mask")
    ax.set_title("Sentence vs non-sentence segments")
    report.add_figure(fig, title="Sentence mask", tags=("mask",))
    plt.close(fig)

    report.save(report_path, overwrite=True, open_browser=False)
    return report_path


def gather_audio_cache(
    bids_path: BIDSPath, segments: Sequence[SequenceSegment]
) -> Tuple[Dict[Path, Tuple[np.ndarray, int]], List[int]]:
    cache: Dict[Path, Tuple[np.ndarray, int]] = {}
    sample_rates: List[int] = []
    for seg in segments:
        if not seg.sound_path:
            continue
        audio_path = (bids_path.root / seg.sound_path).resolve()
        if not audio_path.exists():
            LOGGER.warning("Audio file missing on disk: %s", audio_path)
            continue
        if audio_path not in cache:
            data, sr = sf.read(audio_path, always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            cache[audio_path] = (data.astype(np.float64), sr)
            sample_rates.append(sr)
    return cache, sample_rates


def choose_base_sr(sample_rates: Sequence[int]) -> Optional[int]:
    if not sample_rates:
        return None
    unique = sorted(set(sample_rates))
    if len(unique) > 1:
        LOGGER.warning(
            "Found multiple audio sampling rates %s; resampling to %s Hz.",
            unique,
            unique[-1],
        )
    return unique[-1]


def build_sentence_mask(
    segments: Sequence[SequenceSegment], sfreq: float, n_times: int
) -> np.ndarray:
    mask = np.zeros(n_times, dtype=np.uint8)
    for seg in segments:
        if seg.label != "sentence":
            continue
        start = max(0, int(round(seg.onset * sfreq)))
        stop = min(n_times, int(round(seg.offset * sfreq)))
        if stop <= start:
            continue
        mask[start:stop] = 1
    return mask


def build_audio_timecourses(
    bids_path: BIDSPath,
    segments: Sequence[SequenceSegment],
    audio_cache: Dict[Path, Tuple[np.ndarray, int]],
    base_sr: Optional[int],
    sfreq: float,
    n_times: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    duration = n_times / sfreq
    if base_sr is None or math.isclose(base_sr, 0.0):
        LOGGER.warning("No audio available; returning silent tracks.")
        return np.zeros(0, dtype=np.float64), 1, np.zeros(n_times, dtype=np.float64)
    native_len = int(round(duration * base_sr))
    audio_native = np.zeros(native_len, dtype=np.float64)
    audio_meg = np.zeros(n_times, dtype=np.float64)

    for seg in segments:
        if not seg.sound_path or seg.audio_onset is None or seg.audio_offset is None:
            continue
        audio_path = (bids_path.root / seg.sound_path).resolve()
        if audio_path not in audio_cache:
            continue
        clip, sr = audio_cache[audio_path]
        start_idx = int(round(seg.audio_onset * sr))
        stop_idx = int(round(seg.audio_offset * sr))
        start_idx = max(0, start_idx)
        stop_idx = min(stop_idx, len(clip))
        if stop_idx <= start_idx:
            continue
        clip = clip[start_idx:stop_idx]
        clip_duration = seg.audio_offset - seg.audio_onset
        if clip_duration <= 0:
            continue
        target_native = max(1, int(round(clip_duration * base_sr)))
        target_meg = max(1, int(round(clip_duration * sfreq)))
        if sr != base_sr or abs(len(clip) - target_native) > 1:
            clip_native = resample(clip, target_native)
        else:
            clip_native = clip[:target_native]
        clip_meg = resample(clip, target_meg)

        native_start = int(round(seg.onset * base_sr))
        native_end = min(native_start + len(clip_native), audio_native.size)
        audio_native[native_start:native_end] = clip_native[: native_end - native_start]

        meg_start = int(round(seg.onset * sfreq))
        meg_end = min(meg_start + len(clip_meg), audio_meg.size)
        audio_meg[meg_start:meg_end] = clip_meg[: meg_end - meg_start]

    return audio_native, base_sr, audio_meg


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def preprocess_run(
    bids_path: BIDSPath,
    derivatives_root: Path,
    reports_root: Path,
    overwrite: bool = False,
) -> Optional[RunProduct]:
    try:
        raw = read_raw_bids(bids_path, verbose="error")
    except FileNotFoundError:
        LOGGER.debug("Skipping missing run: %s", bids_path)
        return None

    LOGGER.info("Processing %s", bids_path.root / bids_path.fpath.name)

    raw = raw.pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False, misc=False)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=30.0, n_jobs=1)

    annotations_df = dataframe_from_annotations(raw)
    segments = collect_sequence_segments(annotations_df)
    if not segments:
        LOGGER.warning("No annotated sequences found for %s; skipping.", bids_path.basename)
        return None

    sfreq = raw.info["sfreq"]
    n_times = raw.n_times
    mask = build_sentence_mask(segments, sfreq=sfreq, n_times=n_times)

    audio_cache, sample_rates = gather_audio_cache(bids_path, segments)
    base_sr = choose_base_sr(sample_rates)
    audio_native, native_sr, audio_meg = build_audio_timecourses(
        bids_path, segments, audio_cache, base_sr, sfreq, n_times
    )

    info = mne.create_info(
        ch_names=raw.ch_names,
        sfreq=sfreq,
        ch_types=raw.get_channel_types(),
    )
    info["bads"] = raw.info.get("bads", [])
    preproc_raw = mne.io.RawArray(raw.get_data(), info, verbose="error")
    preproc_raw.set_meas_date(raw.info.get("meas_date"))

    run_root = (
        derivatives_root
        / f"sub-{bids_path.subject}"
        / (f"ses-{bids_path.session}" if bids_path.session is not None else "ses-unknown")
        / f"task-{bids_path.task}"
    )
    ensure_dir(run_root / "meg")
    ensure_dir(run_root / "audio")
    ensure_dir(run_root / "masks")

    base_name = f"sub-{bids_path.subject}_ses-{bids_path.session}_task-{bids_path.task}"
    meg_path = run_root / "meg" / f"{base_name}_meg.fif"
    mask_path = run_root / "masks" / f"{base_name}_sentence_mask.npy"
    audio_native_path = run_root / "audio" / f"{base_name}_audio_native.wav"
    audio_meg_path = run_root / "audio" / f"{base_name}_audio_megfs.wav"
    metadata_path = run_root / f"{base_name}_metadata.json"

    if not overwrite:
        existing = [
            p
            for p in (meg_path, mask_path, audio_native_path, audio_meg_path, metadata_path)
            if p.exists()
        ]
        if existing:
            LOGGER.info("Outputs already exist for %s; skipping.", bids_path.basename)
            return RunProduct(
                meg_path=meg_path,
                mask_path=mask_path,
                audio_native_path=audio_native_path,
                audio_meg_path=audio_meg_path,
                metadata_path=metadata_path,
            )

    preproc_raw.save(meg_path, overwrite=True)
    np.save(mask_path, mask.astype(np.uint8))
    if audio_native.size:
        sf.write(audio_native_path, audio_native, native_sr)
    else:
        sf.write(audio_native_path, np.zeros(1, dtype=np.float64), native_sr)
    sf.write(audio_meg_path, audio_meg, int(round(sfreq)))

    story_names = sorted({seg.story for seg in segments if seg.story})
    segments_by_label = {}
    for seg in segments:
        segments_by_label.setdefault(seg.label, []).append(
            {
                "sequence_id": seg.sequence_id,
                "onset_sec": seg.onset,
                "offset_sec": seg.offset,
                "duration_sec": seg.duration,
                "sound_path": seg.sound_path,
                "audio_onset_sec": seg.audio_onset,
                "audio_offset_sec": seg.audio_offset,
            }
        )

    metadata = {
        "subject": bids_path.subject,
        "session": bids_path.session,
        "task": bids_path.task,
        "stories_present": story_names,
        "source_bids_path": str(bids_path.fpath),
        "sfreq": sfreq,
        "n_channels": len(raw.ch_names),
        "n_samples": int(n_times),
        "mask_summary": {
            "mask_name": "sentence",
            "proportion_sentence": float(mask.mean()) if mask.size else 0.0,
        },
        "audio_native_sr": native_sr,
        "audio_native_len": int(audio_native.size),
        "audio_meg_len": int(audio_meg.size),
        "segments": segments_by_label,
        "preprocessing": {
            "channels": "meg",
            "filter": {"l_freq": 0.5, "h_freq": 30.0},
            "ica": False,
            "baseline": False,
            "decimation": False,
        },
        "mask_description": (
            "Binary vector with ones where annotations label the sequence as a "
            "sentence/story segment and zeros elsewhere (word lists, questions, "
            "responses, etc.)."
        ),
    }
    report_path = create_preprocessing_report(
        preproc_raw,
        mask,
        metadata,
        reports_root,
        base_name,
    )
    metadata["reports"] = {"preprocessing_html": str(report_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2))

    LOGGER.info("Saved preprocessed run to %s", run_root)

    return RunProduct(
        meg_path=meg_path,
        mask_path=mask_path,
        audio_native_path=audio_native_path,
        audio_meg_path=audio_meg_path,
        metadata_path=metadata_path,
    )


def iter_available_runs(
    bids_root: Path, subjects: Optional[Sequence[str]] = None
) -> Iterable[BIDSPath]:
    subjects = list(subjects) if subjects else []
    participants_tsv = bids_root / "participants.tsv"
    if participants_tsv.exists() and not subjects:
        df = pd.read_csv(participants_tsv, sep="\t")
        subjects = sorted({pid.split("-")[1] for pid in df["participant_id"]})
    elif not subjects:
        subjects = sorted(
            {p.name.split("-")[1] for p in bids_root.glob("sub-*") if p.is_dir()}
        )

    for subject in subjects:
        subject_dir = bids_root / f"sub-{subject}"
        for session_dir in sorted(subject_dir.glob("ses-*")):
            session = session_dir.name.split("-")[1]
            meg_dir = session_dir / "meg"
            if not meg_dir.exists():
                continue
            for task_file in sorted(meg_dir.glob(f"sub-{subject}_ses-{session}_task-*_meg.con")):
                task = task_file.name.split("_task-")[1].split("_")[0]
                yield BIDSPath(
                    subject=subject,
                    session=session,
                    task=task,
                    datatype="meg",
                    root=bids_root,
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal MEG-MASC preprocessing with sentence masks."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subset of subjects to process (e.g., 01 02). Defaults to all subjects.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing derivatives.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level for console logging.",
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
    bids_root = repo_root / "bids_anonym"
    if not bids_root.exists():
        LOGGER.error("Expected BIDS directory not found: %s", bids_root)
        return 1

    derivatives_root = repo_root / "derivatives" / "preprocessed"
    ensure_dir(derivatives_root)
    reports_root = repo_root / "derivatives" / "reports" / "preprocessing"
    ensure_dir(reports_root)

    processed = 0
    for bids_path in iter_available_runs(bids_root, args.subjects):
        product = preprocess_run(
            bids_path,
            derivatives_root,
            reports_root,
            overwrite=args.overwrite,
        )
        if product:
            processed += 1

    LOGGER.info("Finished preprocessing %d run(s).", processed)
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
