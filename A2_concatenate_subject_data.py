"""Concatenate per-segment MEG derivatives into subject-level arrays.

The pipeline collects raw MEG, envelope, mask, and audio files for each session
and task, checks that they are aligned, concatenates them along the time axis,
and emits a metadata JSON tracking provenance. Optional audio backends are
selected dynamically so the script can run in environments with different
dependencies installed.
"""

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from functions.generic_helpers import read_repository_root

try:
    import mne  # type: ignore
except ImportError as exc:
    raise ImportError(
        "mne is required for concatenating MEG FIF files. "
        "Please install it before running this script."
    ) from exc

try:
    import soundfile as sf  # type: ignore

    def read_wav(path: Path) -> Tuple[np.ndarray, int, Dict[str, str]]:
        info = sf.info(str(path))
        data, rate = sf.read(str(path), always_2d=True)
        return data, rate, {"subtype": info.subtype}

    def write_wav(path: Path, data: np.ndarray, rate: int, params: Dict[str, str]) -> None:
        sf.write(str(path), data, rate, subtype=params["subtype"])

    AUDIO_BACKEND = "soundfile"
except ImportError:
    try:
        from scipy.io import wavfile  # type: ignore

        def read_wav(path: Path) -> Tuple[np.ndarray, int, Dict[str, str]]:
            rate, data = wavfile.read(path)
            data = np.atleast_2d(data)
            if data.shape[0] < data.shape[1]:
                data = data.T
            return data, rate, {"dtype": str(data.dtype)}

        def write_wav(path: Path, data: np.ndarray, rate: int, params: Dict[str, str]) -> None:
            dtype = np.dtype(params["dtype"])
            wavfile.write(str(path), rate, data.astype(dtype))

        AUDIO_BACKEND = "scipy"
    except ImportError:
        import wave

        def read_wav(path: Path) -> Tuple[np.ndarray, int, Dict[str, int]]:
            with wave.open(str(path), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                frames = wav_file.readframes(n_frames)
            dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
            if sampwidth not in dtype_map:
                raise ValueError(
                    f"Unsupported WAV sample width ({sampwidth}) for file {path}. "
                    "Install soundfile or scipy for broader support."
                )
            dtype = dtype_map[sampwidth]
            data = np.frombuffer(frames, dtype=dtype).reshape(-1, n_channels)
            return data, framerate, {"n_channels": n_channels, "sampwidth": sampwidth}

        def write_wav(path: Path, data: np.ndarray, rate: int, params: Dict[str, int]) -> None:
            n_channels = params["n_channels"]
            sampwidth = params["sampwidth"]
            dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
            dtype = dtype_map[sampwidth]
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(n_channels)
                wav_file.setsampwidth(sampwidth)
                wav_file.setframerate(rate)
                wav_file.writeframes(data.astype(dtype).tobytes())

AUDIO_BACKEND = "wave"


def log(message: str) -> None:
    """Standardized console output."""
    print(message)


def normalise_subject_label(subject: str) -> str:
    if subject.startswith("sub-"):
        return subject
    return f"sub-{int(subject):02d}"


@dataclass
class SegmentInfo:
    """Container describing a single session/task segment to concatenate."""

    session: str
    task: str
    meg_path: Path
    mask_paths: Sequence[Path]
    audio_paths: Dict[str, Path]
    envelope_paths: Dict[str, Path]
    n_times: int = 0
    events_path: Path | None = None
    word_onset_samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    sfreq: float | None = None


def load_word_onset_samples(events_path: Path | None) -> np.ndarray:
    """
    Extract word onset sample indices from a BIDS events.tsv file.

    Returns an empty array when the events file is missing or does not contain
    any eligible word annotations.
    """

    if events_path is None or not events_path.exists():
        return np.empty(0, dtype=np.int64)

    onsets: List[int] = []
    with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                trial_info = ast.literal_eval(row.get("trial_type", ""))
            except (ValueError, SyntaxError, TypeError):
                continue
            if not isinstance(trial_info, dict):
                continue
            if trial_info.get("kind") != "word":
                continue
            try:
                pronounced = float(trial_info.get("pronounced", 1.0))
            except (TypeError, ValueError):
                pronounced = 1.0
            if pronounced == 0.0:
                continue
            try:
                onset_sample = int(row.get("sample", ""))
            except (TypeError, ValueError):
                continue
            onsets.append(onset_sample)

    if not onsets:
        return np.empty(0, dtype=np.int64)

    unique_sorted = sorted(set(onsets))
    return np.asarray(unique_sorted, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    """Return CLI arguments controlling the concatenation run."""

    parser = argparse.ArgumentParser(
        description="Concatenate MEG, mask, envelope, and audio data per subject."
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=Path("derivatives"),
        help="Path to the derivatives directory (default: derivatives).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
        help="Subject identifier to process (default: sub-01).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing concatenated outputs.",
    )
    return parser.parse_args()


def natural_sort_key(item: str) -> Tuple:
    """Sort helper that handles numeric substrings."""
    parts: List[str] = []
    num = ""
    for char in item:
        if char.isdigit():
            num += char
        else:
            if num:
                parts.append(int(num))
                num = ""
            parts.append(char)
    if num:
        parts.append(int(num))
    return tuple(parts)


def collect_segments(subject_dir: Path, envelope_root: Path, bids_root: Path) -> List[SegmentInfo]:
    """
    Locate all session/task folders that contain MEG FIF files.

    The function mirrors the BIDS folder hierarchy and records every companion
    mask, audio, and envelope file that will need to be concatenated alongside
    the MEG time series.
    """

    segments: List[SegmentInfo] = []
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    subject_label = subject_dir.name
    log(f"Scanning sessions for subject directory: {subject_dir}")
    for session_dir in sorted(subject_dir.glob("ses-*"), key=lambda p: natural_sort_key(p.name)):
        for task_dir in sorted(session_dir.glob("task-*"), key=lambda p: natural_sort_key(p.name)):
            meg_files = sorted((task_dir / "meg").glob("*.fif"))
            if not meg_files:
                continue
            if len(meg_files) > 1:
                raise RuntimeError(f"Expected a single FIF file in {task_dir / 'meg'}")
            meg_path = meg_files[0]
            mask_dir = task_dir / "masks"
            mask_paths = sorted(mask_dir.glob("*.npy")) if mask_dir.exists() else []
            audio_dir = task_dir / "audio"
            audio_paths = {}
            for key in ("megfs", "native"):
                matches = sorted(audio_dir.glob(f"*audio_{key}.wav")) if audio_dir.exists() else []
                if matches:
                    audio_paths[key] = matches[0]
            session = session_dir.name
            task = task_dir.name
            envelope_paths: Dict[str, Path] = {}
            envelope_session_dir = envelope_root / session / task
            for key in ("megfs", "native"):
                candidate = envelope_session_dir / f"{subject_dir.name}_{session}_{task}_envelope_{key}.npy"
                if candidate.exists():
                    envelope_paths[key] = candidate

            events_path = (
                bids_root
                / subject_label
                / session
                / "meg"
                / f"{subject_label}_{session}_{task}_events.tsv"
            )
            if not events_path.exists():
                log(f"    Warning: events file missing for {session}/{task}: {events_path}")
                events_path = None

            log(f"  Located segment {session}/{task}")
            segments.append(
                SegmentInfo(
                    session=session,
                    task=task,
                    meg_path=meg_path,
                    mask_paths=mask_paths,
                    audio_paths=audio_paths,
                    envelope_paths=envelope_paths,
                    events_path=events_path,
                )
            )
    if not segments:
        raise RuntimeError(f"No MEG segments found for subject directory {subject_dir}")
    log(f"Found {len(segments)} MEG segments")
    return segments


def concatenate_meg(segments: Sequence[SegmentInfo]) -> Tuple[np.ndarray, Sequence[int], List[str]]:
    """
    Load every FIF file, check channel consistency, and concatenate along time.

    Returns the stacked MEG array alongside helper metadata: the number of
    samples per original file (useful for generating masks) and human-readable
    labels identifying each segment in order.
    """

    data_blocks: List[np.ndarray] = []
    lengths: List[int] = []
    file_labels: List[str] = []
    reference_channels: Sequence[str] = ()

    log(f"Concatenating {len(segments)} MEG segments...")
    for seg in segments:
        raw = mne.io.read_raw_fif(seg.meg_path, preload=True, verbose="ERROR")
        data = raw.get_data()
        if not data_blocks:
            reference_channels = raw.info.get("ch_names", [])
        else:
            current_channels = raw.info.get("ch_names", [])
            if list(reference_channels) != list(current_channels):
                raise ValueError(
                    f"Channel mismatch between segments: {segments[0].meg_path.name} "
                    f"and {seg.meg_path.name}"
                )
        seg.sfreq = float(raw.info.get("sfreq", 0.0))
        if seg.sfreq <= 0.0:
            raise ValueError(f"Invalid sampling frequency reported for {seg.meg_path}: {seg.sfreq}")
        data_blocks.append(data)
        seg.n_times = data.shape[1]
        seg.word_onset_samples = load_word_onset_samples(seg.events_path)
        if seg.word_onset_samples.size:
            log(
                f"    {seg.session}/{seg.task}: captured {seg.word_onset_samples.size} word onsets "
                f"(sfreq={seg.sfreq:g} Hz)."
            )
        lengths.append(seg.n_times)
        file_labels.append(f"{seg.session}/{seg.task}")

    concatenated = np.concatenate(data_blocks, axis=1)
    log(
        f"MEG data concatenated into array with shape {concatenated.shape[0]} channels x "
        f"{concatenated.shape[1]} time points."
    )
    return concatenated, lengths, file_labels


def concatenate_masks(
    segments: Sequence[SegmentInfo], lengths: Sequence[int]
) -> Dict[str, np.ndarray]:
    """
    Merge per-task masks and build an additional boundary mask.

    The mask filenames often encode the condition after the task identifier; we
    reuse that tail to produce intuitive keys in the resulting dictionary.
    """

    mask_values: Dict[str, List[np.ndarray]] = defaultdict(list)
    log("Concatenating mask arrays...")
    for seg in segments:
        for mask_path in seg.mask_paths:
            parts = mask_path.stem.split("_")
            mask_key_start = None
            for idx, part in enumerate(parts):
                if part.startswith("task-"):
                    mask_key_start = idx + 1
                    break
            if mask_key_start is None or mask_key_start >= len(parts):
                mask_key = mask_path.stem
            else:
                # Example: sub-01_ses-0_task-0_sentence_mask -> sentence_mask
                mask_key = "_".join(parts[mask_key_start:])
            mask_values[mask_key].append(np.load(mask_path))

    concatenated_masks: Dict[str, np.ndarray] = {}
    for mask_name, arrays in mask_values.items():
        concatenated_masks[mask_name] = np.concatenate(arrays, axis=-1)

    total_points = sum(lengths)
    concatenation_mask = np.ones(total_points, dtype=np.float32)
    cumulative = np.cumsum(lengths)[:-1]
    # Insert a zero at each stitch point so downstream analyses can exclude transitions.
    concatenation_mask[cumulative] = 0.0
    concatenated_masks["concatenation_boundaries"] = concatenation_mask

    log(f"Created {len(concatenated_masks)} mask arrays.")
    return concatenated_masks


def concatenate_audio(
    segments: Sequence[SegmentInfo], output_dir: Path, subject: str
) -> Dict[str, Path]:
    """
    Concatenate WAV audio recordings while preserving backend metadata.

    We ensure the sample rate stays constant across segments and pass through
    subtype/dtype information so the output files match the original encoding.
    """

    audio_output_paths: Dict[str, Path] = {}
    for audio_key in ("megfs", "native"):
        chunks: List[np.ndarray] = []
        backend_params = None
        sample_rate = None

        for seg in segments:
            if audio_key not in seg.audio_paths:
                continue
            data, rate, params = read_wav(seg.audio_paths[audio_key])
            if sample_rate is None:
                sample_rate = rate
                backend_params = params
            elif rate != sample_rate:
                raise ValueError(
                    f"Sample rate mismatch for {audio_key}: expected {sample_rate}, got {rate}"
                )
            chunks.append(data)

        if not chunks:
            log(f"No audio files found for key '{audio_key}', skipping concatenation.")
            continue

        concatenated = np.concatenate(chunks, axis=0)
        output_path = output_dir / f"{subject}_concatenated_audio_{audio_key}.wav"
        write_wav(output_path, concatenated, sample_rate, backend_params or {})
        audio_output_paths[audio_key] = output_path
        log(f"Wrote concatenated {audio_key} audio to {output_path}")
    return audio_output_paths


def concatenate_word_onsets_seconds(segments: Sequence[SegmentInfo]) -> np.ndarray:
    """
    Concatenate per-segment word onset timestamps (seconds) across the subject.

    Word onsets are derived from the BIDS events.tsv files and validated against
    the length of each MEG segment to ensure they fall within the recorded data.
    """

    onsets_seconds: List[float] = []
    cumulative_time = 0.0

    for seg in segments:
        if seg.sfreq is None or seg.sfreq <= 0.0:
            raise ValueError(f"Missing or invalid sampling frequency for segment {seg.session}/{seg.task}.")
        if seg.n_times <= 0:
            raise ValueError(f"Segment {seg.session}/{seg.task} has zero samples.")

        if seg.word_onset_samples.size:
            valid = seg.word_onset_samples[
                (seg.word_onset_samples >= 0) & (seg.word_onset_samples < seg.n_times)
            ]
            if valid.size < seg.word_onset_samples.size:
                dropped = seg.word_onset_samples.size - valid.size
                log(
                    f"    Dropped {dropped} word onsets outside valid range for {seg.session}/{seg.task}."
                )
            if valid.size:
                onsets_seconds.extend(cumulative_time + (valid.astype(np.float64) / seg.sfreq))

        cumulative_time += seg.n_times / seg.sfreq

    if not onsets_seconds:
        return np.empty(0, dtype=np.float64)

    return np.sort(np.asarray(onsets_seconds, dtype=np.float64))


def validate_lengths(
    meg_data: np.ndarray,
    masks: Dict[str, np.ndarray],
    envelopes: Dict[str, np.ndarray] | None = None,
    skip_envelope_keys: Sequence[str] | None = None,
) -> None:
    """
    Ensure every derived array shares the MEG time axis length.

    Certain envelopes (e.g., native sampling) can be skipped from the validation
    via `skip_envelope_keys` when they are expected to live on a different grid.
    """

    if meg_data.ndim != 2:
        raise ValueError(f"Expected MEG data to be 2-D (channels x time); got shape {meg_data.shape}")
    total_points = meg_data.shape[1]
    log(f"Validating lengths against total MEG time points: {total_points}")

    skip_envelope_keys = tuple(skip_envelope_keys or ())

    if envelopes:
        for env_name, env_data in envelopes.items():
            if env_name in skip_envelope_keys:
                log(f"Skipping length validation for envelope '{env_name}'")
                continue
            env_points = env_data.shape[-1]
            if env_points != total_points:
                raise ValueError(
                    f"Envelope '{env_name}' has {env_points} samples; expected {total_points} to match MEG data."
                )

    for mask_name, mask_data in masks.items():
        mask_points = mask_data.shape[-1]
        if mask_points != total_points:
            raise ValueError(
                f"Mask '{mask_name}' has {mask_points} samples; expected {total_points} to match MEG data."
            )
    log("Length validation passed.")


def discover_concatenated_envelopes(models_root: Path, subject: str) -> Dict[str, Path]:
    subject_label = normalise_subject_label(subject)
    subject_dir = models_root / subject_label / "concatenated"
    if not subject_dir.exists():
        return {}
    envelope_paths: Dict[str, Path] = {}
    for path in subject_dir.glob(f"{subject_label}_concatenated_envelope_*.npy"):
        key = path.stem.split("_")[-1]
        envelope_paths[key] = path
    return envelope_paths


def save_outputs(
    output_dir: Path,
    subject: str,
    meg_data: np.ndarray,
    masks: Dict[str, np.ndarray],
    audio_paths: Dict[str, Path],
    envelope_outputs: Dict[str, Path],
    word_onsets_seconds: np.ndarray | None,
    segment_lengths: Sequence[int],
    segments: Sequence[SegmentInfo],
    overwrite: bool,
) -> None:
    """
    Persist concatenated arrays together with a provenance JSON manifest.

    The manifest records the source files and output locations so down-stream
    scripts or manual QC can trace how each derivative was produced.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    meg_path = output_dir / f"{subject}_concatenated_meg.npy"
    if meg_path.exists() and not overwrite:
        raise FileExistsError(f"{meg_path} already exists. Use --overwrite to replace it.")
    np.save(meg_path, meg_data)
    log(f"Saved MEG data to {meg_path}")

    mask_output_paths: Dict[str, Path] = {}
    for mask_name, mask_data in masks.items():
        if mask_name == "concatenation_boundaries":
            filename = f"{subject}_concatenation_boundaries_mask.npy"
            metadata_key = "concatenation_boundaries_mask"
        else:
            filename = f"{subject}_concatenated_{mask_name}.npy"
            metadata_key = f"concatenated_{mask_name}"
        mask_path = output_dir / filename
        if mask_path.exists() and not overwrite:
            raise FileExistsError(f"{mask_path} already exists. Use --overwrite to replace it.")
        np.save(mask_path, mask_data)
        mask_output_paths[metadata_key] = mask_path
        log(f"Saved mask '{mask_name}' to {mask_path}")

    word_onsets_path: Path | None = None
    if word_onsets_seconds is not None:
        word_onsets_path = output_dir / f"{subject}_concatenated_word_onsets_sec.npy"
        if word_onsets_path.exists() and not overwrite:
            raise FileExistsError(f"{word_onsets_path} already exists. Use --overwrite to replace it.")
        np.save(word_onsets_path, word_onsets_seconds.astype(np.float64, copy=False))
        log(f"Saved word onset timestamps (seconds) to {word_onsets_path}")

    metadata_path = output_dir / f"{subject}_concatenation_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Use --overwrite to replace it.")

    metadata = {
        "audio_backend": AUDIO_BACKEND,
        "segments": [
            {
                "session": seg.session,
                "task": seg.task,
                "meg_file": str(seg.meg_path),
                "n_times": length,
                "mask_files": [str(p) for p in seg.mask_paths],
                "envelope_files": {k: str(p) for k, p in seg.envelope_paths.items()},
                "audio_files": {k: str(p) for k, p in seg.audio_paths.items()},
            }
            for seg, length in zip(segments, segment_lengths)
        ],
        "output_files": {
            "meg": str(meg_path),
            "envelopes": {k: str(path) for k, path in envelope_outputs.items()},
            "masks": {name: str(path) for name, path in mask_output_paths.items()},
            "audio": {k: str(path) for k, path in audio_paths.items()},
            "word_onsets_seconds": str(word_onsets_path) if word_onsets_path else None,
        },
        "word_onsets": {
            "count": int(word_onsets_seconds.size if word_onsets_seconds is not None else 0),
            "unit": "seconds",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log(f"Wrote metadata to {metadata_path}")


def main() -> None:
    """Orchestrate the concatenation workflow for the requested subject."""

    args = parse_args()
    repo_root = read_repository_root()
    derivatives_root = args.derivatives_root
    if not derivatives_root.is_absolute():
        derivatives_root = (repo_root / derivatives_root).resolve()
    bids_root = (repo_root / "bids_anonym").resolve()

    subject_label = normalise_subject_label(args.subject)
    subject_dir = derivatives_root / "preprocessed" / subject_label
    envelope_root = derivatives_root / "Models" / "envelope" / subject_label
    output_dir = subject_dir / "concatenated"

    output_dir.mkdir(parents=True, exist_ok=True)
    segments = collect_segments(subject_dir, envelope_root, bids_root)
    meg_data, segment_lengths, _ = concatenate_meg(segments)
    masks = concatenate_masks(segments, segment_lengths)

    models_envelope_root = derivatives_root / "Models" / "envelope"
    envelope_outputs = discover_concatenated_envelopes(models_envelope_root, subject_label)
    envelopes_for_validation: Dict[str, np.ndarray] = {}
    megfs_envelope_path = envelope_outputs.get("megfs")
    if megfs_envelope_path and megfs_envelope_path.exists():
        envelopes_for_validation["megfs"] = np.load(megfs_envelope_path)
    validate_lengths(meg_data, masks, envelopes=envelopes_for_validation, skip_envelope_keys=("native",))

    audio_paths = concatenate_audio(segments, output_dir, subject_label)
    word_onsets_seconds = concatenate_word_onsets_seconds(segments)
    save_outputs(
        output_dir=output_dir,
        subject=subject_label,
        meg_data=meg_data,
        masks=masks,
        audio_paths=audio_paths,
        envelope_outputs=envelope_outputs,
        word_onsets_seconds=word_onsets_seconds,
        segment_lengths=segment_lengths,
        segments=segments,
        overwrite=args.overwrite,
    )
    log("Concatenation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"Concatenation failed: {error}", file=sys.stderr)
        sys.exit(1)
