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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import gammatone, sosfilt, tf2sos, resample

from functions.generic_helpers import read_repository_root

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper dataclasses and utilities
# ---------------------------------------------------------------------------

@dataclass
class EnvelopeProduct:
    subject: str
    session: str
    task: str
    native_path: Path
    megfs_path: Path
    metadata_path: Path
    native_sr: float
    meg_sr: float
    native_samples: int
    meg_samples: int
    session_index: int
    task_index: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalise_subject_label(value: str | int) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    return f"sub-{int(text):02d}"


def normalise_session_label(value: str | int) -> Tuple[str, int]:
    text = str(value)
    if text.startswith("ses-"):
        digits = "".join(filter(str.isdigit, text))
        return text, int(digits) if digits else 0
    number = int(text)
    return f"ses-{number}", number


def normalise_task_label(value: str | int) -> Tuple[str, int]:
    text = str(value)
    if text.startswith("task-"):
        digits = "".join(filter(str.isdigit, text))
        return text, int(digits) if digits else 0
    number = int(text)
    return f"task-{number}", number


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
    subject_label = normalise_subject_label(meta.get("subject"))
    session_label, session_index = normalise_session_label(meta.get("session"))
    task_label, task_index = normalise_task_label(meta.get("task"))
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
    return EnvelopeProduct(
        subject=subject_label,
        session=session_label,
        task=task_label,
        native_path=native_out,
        megfs_path=megfs_out,
        metadata_path=model_meta_out,
        native_sr=float(native_sr),
        meg_sr=float(meta.get("sfreq")),
        native_samples=int(len(broadband_envelope)),
        meg_samples=int(n_meg_samples),
        session_index=session_index,
        task_index=task_index,
    )


def _downsample_for_plot(
    data: np.ndarray, max_points: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data)
    if n <= max_points:
        idx = np.arange(n)
        return idx, data
    step = max(1, int(np.ceil(n / max_points)))
    idx = np.arange(0, n, step)
    return idx, data[idx]


def generate_envelope_report(metadata_paths: Sequence[Path], report_root: Path) -> None:
    if not metadata_paths:
        LOGGER.info("No envelope metadata found; skipping report generation.")
        return
    ensure_dir(report_root)
    report_path = report_root / "envelope_overview.pdf"

    with PdfPages(report_path) as pdf:
        for meta_path in sorted(metadata_paths):
            metadata = json.loads(meta_path.read_text())
            stem = meta_path.stem.replace("_metadata", "")
            native_path = meta_path.parent / f"{stem}_native.npy"
            meg_path = meta_path.parent / f"{stem}_megfs.npy"
            if not native_path.exists() or not meg_path.exists():
                LOGGER.warning("Missing envelope arrays for %s; skipping.", meta_path)
                continue

            native = np.load(native_path)
            native_sr = float(metadata.get("audio_native_sr", 1.0))
            meg = np.load(meg_path)
            meg_sr = float(metadata.get("meg_sr", 1.0))

            idx_native, native_vals = _downsample_for_plot(native)
            idx_meg, meg_vals = _downsample_for_plot(meg)
            time_native = idx_native / native_sr
            time_meg = idx_meg / meg_sr

            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(time_native, native_vals, label="Native envelope", alpha=0.6)
            ax.plot(time_meg, meg_vals, label="MEG-rate envelope", alpha=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Envelope (a.u.)")
            ax.set_title(
                f"Envelope comparison | sub-{metadata.get('subject')} "
                f"ses-{metadata.get('session')} task-{metadata.get('task')}"
            )
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.2)
            pdf.savefig(fig)
            plt.close(fig)

    LOGGER.info("Saved envelope report to %s", report_path)


def concatenate_subject_envelopes(
    subject_label: str,
    products: Sequence[EnvelopeProduct],
    models_root: Path,
    overwrite: bool,
) -> None:
    if not products:
        return

    products_sorted = sorted(products, key=lambda p: (p.session_index, p.task_index))
    native_parts: List[np.ndarray] = []
    megfs_parts: List[np.ndarray] = []
    segments_meta: List[Dict[str, object]] = []

    for prod in products_sorted:
        native_parts.append(np.load(prod.native_path))
        megfs_parts.append(np.load(prod.megfs_path))
        segments_meta.append(
            {
                "session": prod.session,
                "task": prod.task,
                "native_file": str(prod.native_path),
                "megfs_file": str(prod.megfs_path),
                "native_samples": prod.native_samples,
                "meg_samples": prod.meg_samples,
                "native_sr": prod.native_sr,
                "meg_sr": prod.meg_sr,
            }
        )

    concatenated_native = np.concatenate(native_parts, axis=-1)
    concatenated_megfs = np.concatenate(megfs_parts, axis=-1)

    subject_dir = models_root / subject_label / "concatenated"
    ensure_dir(subject_dir)

    output_paths: Dict[str, Path] = {}
    native_path = subject_dir / f"{subject_label}_concatenated_envelope_native.npy"
    if native_path.exists() and not overwrite:
        LOGGER.info("Subject-level native envelope already exists for %s; skipping.", subject_label)
    else:
        np.save(native_path, concatenated_native.astype(np.float32, copy=False))
        LOGGER.info("Saved native concatenated envelope for %s", subject_label)
    output_paths["native"] = native_path

    megfs_path = subject_dir / f"{subject_label}_concatenated_envelope_megfs.npy"
    if megfs_path.exists() and not overwrite:
        LOGGER.info("Subject-level MEG-rate envelope already exists for %s; skipping.", subject_label)
    else:
        np.save(megfs_path, concatenated_megfs.astype(np.float32, copy=False))
        LOGGER.info("Saved MEG-rate concatenated envelope for %s", subject_label)
    output_paths["megfs"] = megfs_path

    metadata = {
        "subject": subject_label,
        "segments": segments_meta,
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "timepoints": {
            "native": int(concatenated_native.shape[-1]),
            "megfs": int(concatenated_megfs.shape[-1]),
        },
        "sampling_rates": {
            "native": float(products_sorted[0].native_sr),
            "megfs": float(products_sorted[0].meg_sr),
        },
    }
    metadata_path = subject_dir / f"{subject_label}_concatenated_envelope_metadata.json"
    if metadata_path.exists() and not overwrite:
        LOGGER.info("Concatenated envelope metadata already exists for %s; skipping.", subject_label)
        return
    metadata_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Saved envelope concatenation metadata for %s", subject_label)


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
    ensure_dir(models_root)
    report_root = repo_root / "derivatives" / "reports" / "Models" / "envelope"
    ensure_dir(report_root)

    processed = 0
    metadata_paths: List[Path] = []
    subject_products: Dict[str, List[EnvelopeProduct]] = defaultdict(list)
    for metadata_path in find_preprocessed_runs(preproc_root, args.subjects):
        product = build_envelope_for_run(
            metadata_path, preproc_root, models_root, overwrite=args.overwrite
        )
        if product:
            processed += 1
            metadata_paths.append(product.metadata_path)
            subject_products[product.subject].append(product)

    for subject, products in subject_products.items():
        concatenate_subject_envelopes(
            subject_label=subject,
            products=products,
            models_root=models_root,
            overwrite=args.overwrite,
        )

    LOGGER.info("Finished building envelopes for %d run(s).", processed)
    generate_envelope_report(metadata_paths, report_root)
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
