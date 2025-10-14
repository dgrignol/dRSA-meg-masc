#!/usr/bin/env python3
"""Construct gammatone-based speech envelopes for MEG-MASC audio.

This script reconstructs broadband speech envelopes from the native audio
recordings stored in ``derivatives/preprocessed``. For each subject/session/task
run it:

1. Loads the native audio waveform (preserved sampling rate) and metadata.
2. Passes the signal through a bank of 15 perceptually-uniform gammatone
   filters with centre frequencies between 150 Hz and 4 kHz.
3. Applies full-wave rectification and a power-law compression (exponent 0.6)
   to emulate inner-ear transduction.
4. Averages the sub-band envelopes to form a single wideband envelope.
5. Optionally resamples the envelope so that it aligns sample-by-sample with
   the preprocessed MEG timeline.

Outputs are stored under ``derivatives/Models/envelope`` with the same run
hierarchy as the preprocessed data. For each run we save:

- ``*_envelope_native.npy`` (float32, shape ``[n_samples_native]``)
- ``*_envelope_megfs.npy`` (float32, shape ``[n_samples_meg]``)
- ``*_metadata.json`` summarising the filter bank and processing parameters.

The implementation follows Issa et al. (2024) and Biesmans et al. (2017) in
spirit. Equivalent rectangular bandwidths are spaced uniformly on the ERB-rate
axis; bandwidth scaling is approximated using the default SciPy gammatone
filters (order 4, IIR realisation).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import gammatone, sosfilt, tf2sos, resample

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper dataclasses and utilities
# ---------------------------------------------------------------------------

@dataclass
class EnvelopeProduct:
    native_path: Path
    megfs_path: Path
    metadata_path: Path


def read_repository_root() -> Path:
    pointer = Path("data_path.txt")
    if not pointer.exists():
        raise FileNotFoundError(
            "data_path.txt is missing. Create the file with the absolute path to the repository root."
        )
    root = Path(pointer.read_text().strip()).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    return root


def erb_scale_freqs(f_min: float, f_max: float, n_filters: int) -> np.ndarray:
    """Return centre frequencies spaced uniformly on the ERB-rate scale."""

    def hz_to_erbrate(freq_hz: float) -> float:
        return 21.4 * np.log10(4.37e-3 * freq_hz + 1.0)

    def erbrate_to_hz(erb_rate: float) -> float:
        return (10 ** (erb_rate / 21.4) - 1.0) / 4.37e-3

    erb_min = hz_to_erbrate(f_min)
    erb_max = hz_to_erbrate(f_max)
    erb_points = np.linspace(erb_min, erb_max, n_filters)
    return erbrate_to_hz(erb_points)


def compute_filter_bank(center_freqs: Sequence[float], fs: float) -> List[np.ndarray]:
    """Create SOS filters for each centre frequency."""
    sos_filters: List[np.ndarray] = []
    for fc in center_freqs:
        if fc >= fs / 2:
            LOGGER.warning(
                "Skipping centre frequency %.2f Hz (>= Nyquist %.2f Hz).", fc, fs / 2
            )
            continue
        b, a = gammatone(fc, "iir", fs=fs)
        sos_filters.append(tf2sos(b, a))
    if not sos_filters:
        raise RuntimeError("No valid gammatone filters created. Check sampling rate.")
    return sos_filters


def apply_filter_bank(audio: np.ndarray, sos_filters: Sequence[np.ndarray]) -> np.ndarray:
    """Filter audio with each SOS filter and apply power-law compression."""
    envelopes = []
    for sos in sos_filters:
        filtered = sosfilt(sos, audio)
        envelope = np.abs(filtered) ** 0.6  # full-wave rectification + power-law
        envelopes.append(envelope)
    return np.stack(envelopes, axis=0)


def resample_to_meg(envelope: np.ndarray, native_sr: float, meg_sr: float, n_meg: int) -> np.ndarray:
    """Resample envelope to MEG sampling rate, matching the exact MEG length."""
    if np.isclose(native_sr, meg_sr):
        env_meg = envelope.copy()
    else:
        target_samples = max(1, int(round(len(envelope) * meg_sr / native_sr)))
        env_meg = resample(envelope, target_samples)
    if len(env_meg) != n_meg:
        env_meg = resample(env_meg, n_meg)
    return env_meg


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def find_preprocessed_runs(preproc_root: Path, subjects: Optional[Sequence[str]]) -> Iterable[Path]:
    """Yield metadata JSON paths for available preprocessed runs."""
    if not preproc_root.exists():
        LOGGER.error("Preprocessed directory not found: %s", preproc_root)
        return []
    subject_dirs = sorted(preproc_root.glob("sub-*"))
    if subjects:
        subject_dirs = [d for d in subject_dirs if d.name.split("-")[1] in subjects]
    for subj_dir in subject_dirs:
        for metadata_path in subj_dir.glob("ses-*/task-*/sub-*_metadata.json"):
            yield metadata_path


def build_envelope_for_run(
    metadata_path: Path,
    preproc_root: Path,
    models_root: Path,
    overwrite: bool = False,
) -> Optional[EnvelopeProduct]:
    meta = json.loads(metadata_path.read_text())
    base_dir = metadata_path.parent
    audio_native_path = base_dir / "audio" / f"{metadata_path.stem.replace('_metadata', '')}_audio_native.wav"
    audio_meg_path = base_dir / "audio" / f"{metadata_path.stem.replace('_metadata', '')}_audio_megfs.wav"

    if not audio_native_path.exists():
        LOGGER.warning("Native audio missing for %s; skipping.", metadata_path)
        return None
    if not audio_meg_path.exists():
        LOGGER.warning("MEG-rate audio missing for %s; skipping.", metadata_path)
        return None

    data, native_sr = sf.read(audio_native_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)

    meg_audio, meg_sr = sf.read(audio_meg_path)
    if isinstance(meg_audio, np.ndarray) and meg_audio.ndim > 1:
        meg_audio = meg_audio.mean(axis=1)
    n_meg_samples = len(meg_audio)

    # Build gammatone filter bank.
    center_freqs = erb_scale_freqs(f_min=150.0, f_max=4000.0, n_filters=15)
    sos_filters = compute_filter_bank(center_freqs, native_sr)
    subband_envs = apply_filter_bank(data, sos_filters)
    broadband_envelope = subband_envs.mean(axis=0)

    envelope_meg = resample_to_meg(
        broadband_envelope,
        native_sr=native_sr,
        meg_sr=float(meta["sfreq"]),
        n_meg=n_meg_samples,
    )

    run_rel_path = metadata_path.relative_to(preproc_root)
    run_root = models_root / run_rel_path.parent
    run_root.mkdir(parents=True, exist_ok=True)

    base_name = metadata_path.stem.replace("_metadata", "")
    native_out = run_root / f"{base_name}_envelope_native.npy"
    megfs_out = run_root / f"{base_name}_envelope_megfs.npy"
    model_meta_out = run_root / f"{base_name}_envelope_metadata.json"

    if not overwrite and native_out.exists() and megfs_out.exists():
        LOGGER.info("Envelope already exists for %s; skipping.", base_name)
        return EnvelopeProduct(native_out, megfs_out, model_meta_out)

    np.save(native_out, broadband_envelope.astype(np.float32))
    np.save(megfs_out, envelope_meg.astype(np.float32))

    model_metadata = {
        "subject": meta.get("subject"),
        "session": meta.get("session"),
        "task": meta.get("task"),
        "stories_present": meta.get("stories_present"),
        "audio_native_path": str(audio_native_path),
        "audio_native_sr": native_sr,
        "audio_native_samples": int(len(broadband_envelope)),
        "meg_sr": meta.get("sfreq"),
        "meg_samples": int(n_meg_samples),
        "filter_bank": {
            "n_filters": len(sos_filters),
            "centre_frequencies_hz": center_freqs[: len(sos_filters)].tolist(),
            "power_law_exponent": 0.6,
        },
        "processing": {
            "gammatone_impl": "scipy.signal.gammatone (order=4, IIR)",
            "rectification": "full-wave (absolute value)",
            "compression": "power-law exponent 0.6",
            "aggregation": "mean across sub-band envelopes",
            "resampling": "scipy.signal.resample to match MEG length",
        },
    }
    model_meta_out.write_text(json.dumps(model_metadata, indent=2))

    LOGGER.info("Saved envelope model to %s", run_root)
    return EnvelopeProduct(native_out, megfs_out, model_meta_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate gammatone-based speech envelopes for MEG-MASC runs."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subset of subjects to process (e.g., 01 02). Defaults to all available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing envelope files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level for console output.",
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
    models_root = repo_root / "derivatives" / "Models" / "envelope"
    models_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    for metadata_path in find_preprocessed_runs(preproc_root, args.subjects):
        product = build_envelope_for_run(metadata_path, preproc_root, models_root, overwrite=args.overwrite)
        if product:
            processed += 1

    LOGGER.info("Finished building envelopes for %d run(s).", processed)
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
