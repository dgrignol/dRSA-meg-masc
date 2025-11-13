import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from argparse import BooleanOptionalAction

import numpy as np
from functions.core_functions import (
    subsampling,
    compute_lag_correlation,
    compute_rsa_matrix_corr,
    compute_rdm_series_from_indices,
    save_subsample_diagnostics,
)
from functions.generic_helpers import (
    ensure_analysis_directories,
    format_log_timestamp,
    read_repository_root,
)
from functions.regression_methods import RegressionConfig, compute_dRSA_regression


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run subject-level dRSA analysis and store outputs in a structured results folder."
    )
    parser.add_argument(
        "subject",
        help="Subject identifier (e.g., 'sub-01' or '1').",
    )
    parser.add_argument(
        "--analysis-name",
        help=(
            "Custom analysis folder name. When omitted, a timestamped name such as "
            "'20240130_143210' is generated automatically."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Parent directory that will contain the analysis folder (default: results/).",
    )
    parser.add_argument(
        "--lock-subsample-to-word-onset",
        action="store_true",
        help="Restrict subsample start positions to word onset timestamps.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow subsample windows to overlap when drawing random starts.",
    )
    parser.add_argument(
        "--word-onset-alignment",
        choices=("center", "start"),
        default="center",
        help=(
            "When locking subsamples to word onsets, center the window on the onset (default) "
            "or treat the onset as the window start."
        ),
    )
    parser.add_argument(
        "--regression-method",
        choices=("correlation", "pcr", "ridge", "lasso", "elasticnet"),
        default="elasticnet",
        help="Model–neural regression method (default: elasticnet).",
    )
    parser.add_argument(
        "--regression-alpha",
        type=float,
        default=1.0,
        help="Regularization strength for Ridge/Lasso/Elastic Net (ignored for correlation/PCR).",
    )
    parser.add_argument(
        "--regression-l1-ratio",
        type=float,
        default=0.5,
        help="Elastic Net mixing parameter between L1=1.0 and L2=0.0 penalties.",
    )
    parser.add_argument(
        "--pcr-variance-threshold",
        type=float,
        default=0.85,
        help="Fraction of variance PCs must explain in PCR mode.",
    )
    parser.add_argument(
        "--regression-border-threshold",
        type=float,
        default=0.1,
        help="Lag-autocorrelation threshold for excluding self-model predictors.",
    )
    parser.add_argument(
        "--regression-mem-threshold-gb",
        type=float,
        default=2.0,
        help="Per-buffer size (GiB) above which regression RDM stacks use on-disk memmaps.",
    )
    parser.add_argument(
        "--plot-regression-borders",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Write per-model autocorrelation plots with regression borders overlaid "
            "(default: enabled; disable with --no-plot-regression-borders)."
        ),
    )
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=10,
        help="Log buffering/regression progress every N iterations (default: 10).",
    )
    parser.add_argument(
        "--progress-neural-step",
        type=int,
        default=50,
        help="During regression, log after every N neural time points per iteration (default: 50).",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help=(
            "Replace the neural signal with each model (plus the neural signal itself) in turn and "
            "run the full dRSA pipeline for every configuration."
        ),
    )
    return parser.parse_args(argv)


def compute_mask_signature(mask_array):
    if mask_array is None:
        return "nomask"
    mask_bool = np.asarray(mask_array, dtype=bool)
    packed = np.packbits(mask_bool)
    return hashlib.sha1(packed.tobytes()).hexdigest()


def compute_array_signature(values, dtype):
    if values is None:
        return "none"
    arr = np.asarray(values, dtype=dtype)
    if arr.size == 0:
        return "empty"
    return hashlib.sha1(arr.tobytes(order="C")).hexdigest()


def slugify_label(label: str) -> str:
    """Create a filesystem-friendly suffix from an arbitrary label."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return cleaned or "unnamed"


def log(message: str) -> None:
    """Print ``message`` with a timestamp prefix."""
    print(f"[{format_log_timestamp()}] {message}")


@dataclass
class RDMSeriesBuffer:
    """Container for an RDM time-series stack, optionally backed by a memmap."""

    label: str
    array: np.ndarray
    path: Path | None
    bytes_allocated: int

    def is_memmap(self) -> bool:
        return isinstance(self.array, np.memmap)

    def flush(self) -> None:
        if self.is_memmap():
            self.array.flush()

    def cleanup(self) -> None:
        if self.is_memmap():
            self.array.flush()
            mmap_obj = getattr(self.array, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            if self.path:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        self.array = None
        self.path = None


def allocate_rdm_buffer(
    label: str,
    shape: tuple[int, ...],
    dtype,
    mem_threshold_bytes: int,
    tmp_dir: Path,
) -> RDMSeriesBuffer:
    """Allocate an RDM buffer either in RAM or via an on-disk memmap."""

    entries = int(np.prod(shape, dtype=np.int64))
    bytes_needed = entries * np.dtype(dtype).itemsize
    use_memmap = bytes_needed >= mem_threshold_bytes
    if use_memmap:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        filename = tmp_dir / f"{slugify_label(label)}_{uuid.uuid4().hex}.dat"
        array = np.memmap(filename, mode="w+", dtype=dtype, shape=shape)
        path = filename
    else:
        array = np.empty(shape, dtype=dtype)
        path = None
    return RDMSeriesBuffer(label=label, array=array, path=path, bytes_allocated=bytes_needed)

# CLI + repository root
args = parse_args()

if not 0.0 <= args.regression_l1_ratio <= 1.0:
    raise ValueError("--regression-l1-ratio must be within [0, 1].")
if args.regression_mem_threshold_gb <= 0:
    raise ValueError("--regression-mem-threshold-gb must be positive.")
if args.progress_log_every <= 0:
    raise ValueError("--progress-log-every must be a positive integer.")
if args.progress_neural_step <= 0:
    raise ValueError("--progress-neural-step must be a positive integer.")

try:  # ensure logs appear promptly on clusters that buffer stdout heavily
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

if not args.lock_subsample_to_word_onset and args.word_onset_alignment != "center":
    log(
        "Warning: --word-onset-alignment has no effect unless --lock-subsample-to-word-onset is set."
    )

repo_root = read_repository_root()

if args.results_root.is_absolute():
    resolved_results_root = args.results_root
else:
    resolved_results_root = (repo_root / args.results_root).resolve()
analysis_name, analysis_root, single_subjects_dir, group_level_dir = ensure_analysis_directories(
    resolved_results_root, args.analysis_name
)
results_root = analysis_root.parent

simulations_dir = analysis_root / "simulations"
simulations_dir.mkdir(parents=True, exist_ok=True)

log(f"Analysis name: {analysis_name}")
log(f"Analysis directory: {analysis_root}")
log(f"Subject-level outputs will be written to: {single_subjects_dir}")
if args.simulation:
    log(f"Simulation outputs will be written to: {simulations_dir}")

# ===== load data =====
# paths
subject_arg = args.subject
if subject_arg.startswith("sub-"):
    subject_label = subject_arg
    subject = int(subject_arg.split("-")[1])
else:
    subject = int(subject_arg)
    subject_label = f"sub-{subject:02d}"
session_label = "all"#"ses-0"
task_label = "all"#"task-0"
log(f"subject: {subject}")

envelope_path_candidates = [
    os.path.join(
        repo_root,
        "derivatives",
        "Models",
        "envelope",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_envelope_100Hz.npy",
    ),
]
envelope_path = next((path for path in envelope_path_candidates if os.path.exists(path)), None)
if envelope_path is None:
    raise FileNotFoundError(
        "Unable to locate an envelope model file. Checked:\n"
        + "\n".join(envelope_path_candidates)
    )

wordfreq_path_candidates = [
    os.path.join(
        repo_root,
        "derivatives",
        "Models",
        "wordfreq",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_wordfreq_100Hz.npy",
    ),
]
wordfreq_path = next((path for path in wordfreq_path_candidates if os.path.exists(path)), None)
if wordfreq_path is None:
    raise FileNotFoundError(
        "Unable to locate a word-frequency model file. Checked:\n"
        + "\n".join(wordfreq_path_candidates)
    )

glove_path_candidates = [
    os.path.join(
        repo_root,
        "derivatives",
        "Models",
        "glove",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_glove_100Hz.npy",
    )
]
glove_path = next((path for path in glove_path_candidates if os.path.exists(path)), None)
if glove_path is None:
    raise FileNotFoundError(
        "Unable to locate a GloVe model file. Checked:\n"
        + "\n".join(glove_path_candidates)
    )

glove_norm_path_candidates = [
    os.path.join(
        repo_root,
        "derivatives",
        "Models",
        "glove",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_glove_100Hz_norm.npy",
    )
]
glove_norm_path = next((path for path in glove_norm_path_candidates if os.path.exists(path)), None)
if glove_norm_path is None:
    raise FileNotFoundError(
        "Unable to locate a GloVe norm file. Checked:\n"
        + "\n".join(glove_norm_path_candidates)
    )

mask_paths = [
    os.path.join(
        repo_root,
        "derivatives",
        "preprocessed",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_sentence_mask_100Hz.npy",
    ),
    os.path.join(
        repo_root,
        "derivatives",
        "preprocessed",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenation_boundaries_mask_100Hz.npy",
    ),
]

voicing_path_candidates = [
    os.path.join(
        repo_root,
        "derivatives",
        "Models",
        "voicing",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_voicing_100Hz.npy",
    )
]
voicing_path = next((path for path in voicing_path_candidates if os.path.exists(path)), None)

concatenated_dir = Path(
    repo_root,
    "derivatives",
    "preprocessed",
    subject_label,
    "concatenated",
)

word_onsets_expected_path = concatenated_dir / f"{subject_label}_concatenated_word_onsets_sec.npy"
word_onsets_path = word_onsets_expected_path if word_onsets_expected_path.exists() else None
concatenation_metadata_path = concatenated_dir / f"{subject_label}_concatenation_metadata.json"

MEG_path = concatenated_dir / f"{subject_label}_concatenated_meg_100Hz.npy"
frequency_band = 'unfiltered'

word_onset_seconds = None
word_onset_seconds_signature = "none"
word_onset_candidate_starts = None
word_onset_candidate_signature = "none"
word_onset_count = 0
word_onset_candidate_total = 0
word_onset_candidate_within_bounds = 0
word_onset_candidate_trimmed = None
word_onset_loading_error = None
word_onset_invalid_dropped = 0
if args.lock_subsample_to_word_onset and word_onsets_path is None:
    raise FileNotFoundError(
        f"--lock-subsample-to-word-onset requested but word onset file is missing: {word_onsets_expected_path}"
    )

# shape of loaded data should be (features, tps)
# raw = mne.io.read_raw_fif(MEG_path, preload=True)
# neural_data, times = raw.get_data(return_times=True)
neural_data = np.load(MEG_path)

model_paths = {}
selected_models = []
selected_models_labels = []
model_rdm_metrics = []
model_regression_borders = {}

model_filter_env = os.getenv("DRSA_MODELS")
if model_filter_env:
    model_filter = {
        token.strip().lower() for token in model_filter_env.split(",") if token.strip()
    }
else:
    model_filter = None


def _register_model(path, label, metric, regression_border=None):
    """Register a model RDM source.

    Parameters
    ----------
    path : str
        Filesystem path to the numpy array (features × timepoints).
    label : str
        Human-readable name used in metadata/plots.
    metric : str
        Distance metric used when building model RDMs (passed to `pdist`).
    regression_border : float, optional
        Per-model autocorrelation threshold for regression border estimation.
        Defaults to the global --regression-border-threshold when omitted.
    """
    if model_filter and label.lower() not in model_filter:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = np.load(path)
    if model.ndim == 1:
        model = model[None, :]
    if model.shape[-1] != neural_data.shape[-1]:
        raise ValueError(
            f"Model '{label}' length mismatch: {model.shape[-1]} vs neural data {neural_data.shape[-1]}"
        )
    selected_models.append(model)
    selected_models_labels.append(label)
    model_rdm_metrics.append(metric)
    model_paths[label] = path
    if regression_border is None:
        regression_border = args.regression_border_threshold
    model_regression_borders[label] = float(regression_border)


def _resolve_model_borders(model_labels):
    """Return per-model border thresholds aligned with `model_labels`."""
    return [
        model_regression_borders.get(lbl, args.regression_border_threshold)
        for lbl in model_labels
    ]
# 0.05 0.1 0.1 0.3 0.1 0.05 0.2
_register_model(envelope_path, "Envelope", "euclidean", regression_border=0.05)
_register_model(voicing_path, "Phoneme Voicing", "euclidean", regression_border=0.05)
_register_model(wordfreq_path, "Word Frequency", "euclidean", regression_border=0.2)
_register_model(glove_path, "GloVe", "correlation", regression_border=0.1)
_register_model(glove_norm_path, "GloVe Norm", "euclidean", regression_border=0.1)

# Optional GPT next-token models
gpt_next_path_candidates = [
    os.path.join(
        repo_root,
        "derivatives",
        "Models",
        "gpt_next",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_gpt_next_100Hz.npy",
    )
]
gpt_next_path = next((p for p in gpt_next_path_candidates if os.path.exists(p)), None)
if gpt_next_path is not None:
    _register_model(gpt_next_path, "GPT Next-Token", "correlation", regression_border=0.25)

# Predictability is currently disabled by request; enable via DRSA_MODELS filter if needed.

gpt_surp_path_candidates = [
    os.path.join(
        repo_root,
        "derivatives",
        "Models",
        "gpt_next",
        subject_label,
        "concatenated",
        f"{subject_label}_concatenated_gpt_surprisal_100Hz.npy",
    )
]
gpt_surp_path = next((p for p in gpt_surp_path_candidates if os.path.exists(p)), None)
if gpt_surp_path is not None:
    _register_model(gpt_surp_path, "GPT Surprisal", "euclidean", regression_border=0.1)


# Report selected models to log
if model_filter_env:
    log(f"DRSA_MODELS filter: {model_filter_env}")
if selected_models_labels:
    log("Selected models (label | features | metric | path):")
    for mdl, lbl, mtr in zip(selected_models, selected_models_labels, model_rdm_metrics):
        feats = int(np.atleast_2d(mdl).shape[0])
        path = model_paths.get(lbl, "<unknown>")
        log(f"  - {lbl} | {feats} | {mtr} | {path}")
else:
    log("Warning: no models selected after filtering and existence checks.")


# ===== settings =====

save_rsa_matrices = True
save_lag_curves = True

mask = None
if mask_paths:
    masks = []
    for path in mask_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask file not found: {path}")
        masks.append(np.load(path))

    if masks[0].shape != masks[1].shape:
        raise ValueError(f"Mask shape mismatch: {masks[0].shape} vs {masks[1].shape}")

    mask = np.logical_and(masks[0], masks[1])

double_precision = False

lag_bootstrap_iterations = 100
lag_bootstrap_confidence = 0.95
lag_bootstrap_random_state = 0


# ===== select data and set RDM metrics =====                    

selected_neural_data = neural_data

neural_rdm_metric = 'correlation'

# Hard-coded neural subsets (currently three identical MEG datasets).
default_neural_signal_sets = [
    ("MEG Full 1", selected_neural_data),
#    ("MEG Full 2", selected_neural_data), # this is just a placeholder for now, disabled to reduce runtime
#    ("MEG Full 3", selected_neural_data), # this is just a placeholder for now, disabled to reduce runtime
]
default_neural_metrics = [neural_rdm_metric for _ in default_neural_signal_sets]

selected_models = [np.atleast_2d(model) for model in selected_models]  # ensure 2D arrays. If 1D, add feature axis.

# model_rdm_metrics already defined during registration; valid options include:
# euclidean - cosine - hamming - correlation - jaccard

rsa_computation_method = args.regression_method.lower()
log(f"Regression method set to: {rsa_computation_method}")

# check if lengths match
if not (len(selected_models) == len(selected_models_labels) == len(model_rdm_metrics)):
    raise ValueError(
        f"Lengths do not match: "
        f"models={len(selected_models)}, "
        f"labels={len(selected_models_labels)}, "
        f"metrics={len(model_rdm_metrics)}"
    )


# ===== set parameters =====

n_subsamples = 70 # e.g. 150 seconds
subsampling_iterations = 80 # e.g. 100 seconds

SubSampleDurSec = 5 # e.g. 5 seconds
averaging_diagonal_time_window_sec = 3 # e.g. 3 seconds

resolution = 100 # in Hz - change as needed

tps = selected_neural_data.shape[1]
adtw_in_tps = averaging_diagonal_time_window_sec * resolution
# this is to avoid noisy diagonal averaging in the dRSA matrix edges
# and also, not to use the models outside the e.g. -3 +3 window for the PCR.
subsample_tps = SubSampleDurSec * resolution # subsample size in tps - also number of RDMs calculated for each subsample

subsampling_random_state = None
rdm_length = n_subsamples * (n_subsamples - 1) // 2

analysis_parameters = {
    "n_subsamples": n_subsamples,
    "subsampling_iterations": subsampling_iterations,
    "subsample_duration_sec": SubSampleDurSec,
    "averaging_diagonal_time_window_sec": averaging_diagonal_time_window_sec,
    "resolution_hz": resolution,
    "tps": tps,
    "averaging_window_tps": adtw_in_tps,
    "subsample_tps": subsample_tps,
    "rdm_length": rdm_length,
    "subsampling_random_state": subsampling_random_state,
    "word_onset_alignment": (
        args.word_onset_alignment if args.lock_subsample_to_word_onset else None
    ),
    "regression_method": rsa_computation_method,
    "regression_alpha": args.regression_alpha,
    "regression_l1_ratio": args.regression_l1_ratio,
    "pcr_variance_threshold": args.pcr_variance_threshold,
    "regression_border_threshold": args.regression_border_threshold,
    "regression_border_per_model": model_regression_borders,
    "regression_mem_threshold_gb": args.regression_mem_threshold_gb,
    "plot_regression_borders": args.plot_regression_borders,
    "progress_log_every": args.progress_log_every,
    "progress_neural_step": args.progress_neural_step,
}

if word_onsets_path is not None:
    try:
        raw_onsets = np.load(word_onsets_path)
    except Exception as exc:  # pragma: no cover - safety logging
        word_onset_loading_error = str(exc)
        if args.lock_subsample_to_word_onset:
            raise RuntimeError(
                f"Failed to load word onset timestamps from {word_onsets_path}: {exc}"
            ) from exc
        log(f"Warning: failed to load word onset timestamps from {word_onsets_path}: {exc}")
    else:
        raw_array = np.asarray(raw_onsets, dtype=np.float64).ravel()
        if raw_array.size:
            finite_mask = np.isfinite(raw_array)
            nonnegative_mask = raw_array >= 0.0
            valid_mask = finite_mask & nonnegative_mask
            if not np.all(valid_mask):
                word_onset_invalid_dropped = int(raw_array.size - np.count_nonzero(valid_mask))
            valid_seconds = raw_array[valid_mask]
            if valid_seconds.size:
                word_onset_seconds = np.unique(valid_seconds)
            else:
                word_onset_seconds = np.empty(0, dtype=np.float64)
        else:
            word_onset_seconds = np.empty(0, dtype=np.float64)
        word_onset_count = int(word_onset_seconds.size)
        word_onset_seconds_signature = compute_array_signature(word_onset_seconds, dtype=np.float64)
        if args.lock_subsample_to_word_onset and word_onset_count == 0:
            raise ValueError(
                f"--lock-subsample-to-word-onset requested but {word_onsets_path} contains no valid entries."
            )
        if not args.lock_subsample_to_word_onset and word_onset_count == 0:
            log(f"Warning: word onset file {word_onsets_path} contains no valid entries.")
else:
    word_onset_seconds = None
    word_onset_seconds_signature = "none"

if word_onset_seconds is not None:
    scaled_onsets = np.rint(word_onset_seconds * resolution)
    finite_scaled = scaled_onsets[np.isfinite(scaled_onsets)]
    onset_indices = np.asarray(finite_scaled, dtype=np.float64)
    if onset_indices.size:
        onset_indices = np.unique(onset_indices[onset_indices >= 0.0])
    else:
        onset_indices = np.empty(0, dtype=np.float64)

    if args.word_onset_alignment == "center":
        half_window = (subsample_tps - 1) / 2.0
        aligned_candidates = np.rint(onset_indices - half_window)
    else:
        aligned_candidates = onset_indices

    aligned_candidates = aligned_candidates.astype(np.int64, copy=False)
    if aligned_candidates.size:
        aligned_candidates = np.unique(aligned_candidates[aligned_candidates >= 0])
    else:
        aligned_candidates = np.empty(0, dtype=np.int64)

    word_onset_candidate_total = int(aligned_candidates.size)
    max_start = max(0, tps - subsample_tps)
    candidate_indices_within_bounds = aligned_candidates[aligned_candidates <= max_start]
    word_onset_candidate_within_bounds = int(candidate_indices_within_bounds.size)
    word_onset_candidate_starts = candidate_indices_within_bounds
    word_onset_candidate_signature = compute_array_signature(candidate_indices_within_bounds, dtype=np.int64)
    word_onset_candidate_trimmed = word_onset_candidate_total - word_onset_candidate_within_bounds
    if args.lock_subsample_to_word_onset and word_onset_candidate_within_bounds == 0:
        raise ValueError(
            "--lock-subsample-to-word-onset requested but no word onset permits a full subsample window "
            f"with word_onset_alignment='{args.word_onset_alignment}'. "
            f"Check {word_onsets_path} and consider adjusting the subsample duration ({SubSampleDurSec} s)."
        )
    if word_onset_count:
        if word_onset_invalid_dropped:
            log(
                f"Loaded {word_onset_count} word onset timestamps (dropped {word_onset_invalid_dropped} invalid entries)."
            )
        else:
            log(f"Loaded {word_onset_count} word onset timestamps.")
    if args.lock_subsample_to_word_onset:
        log(
            f"Locking subsample windows to {word_onset_candidate_within_bounds} "
            f"{args.word_onset_alignment}-aligned onset start(s) (allow_overlap={args.allow_overlap})."
        )
        if word_onset_candidate_trimmed > 0:
            log(
                f"  Ignored {word_onset_candidate_trimmed} onset(s) that cannot fit a {SubSampleDurSec}-s window at {resolution} Hz."
            )
    elif word_onset_candidate_within_bounds:
        log(
            f"Word onsets supply {word_onset_candidate_within_bounds} "
            f"{args.word_onset_alignment}-aligned candidate start(s) at {resolution} Hz."
        )
else:
    word_onset_candidate_signature = "none"
    word_onset_candidate_starts = None

analysis_parameters.update(
    {
        "lock_subsample_to_word_onset": args.lock_subsample_to_word_onset,
        "allow_overlap": args.allow_overlap,
        "word_onset_count": word_onset_count if word_onset_seconds is not None else None,
        "word_onset_candidate_total": (
            word_onset_candidate_total if word_onset_seconds is not None else None
        ),
        "word_onset_candidate_within_bounds": (
            word_onset_candidate_within_bounds if word_onset_seconds is not None else None
        ),
        "word_onset_invalid_entries": (
            word_onset_invalid_dropped if word_onset_seconds is not None else None
        ),
        "word_onset_candidate_trimmed": (
            word_onset_candidate_trimmed if word_onset_seconds is not None else None
        ),
    }
)
base_analysis_run_id = f"{subject_label}_res{resolution}_{rsa_computation_method}"
subject_results_dir = single_subjects_dir
analysis_output_dir = simulations_dir if args.simulation else subject_results_dir
analysis_output_dir.mkdir(parents=True, exist_ok=True)

if args.simulation:
    existing_simulations = sorted(
        analysis_output_dir.glob(f"{base_analysis_run_id}_sim_*_metadata.json")
    )
    if existing_simulations:
        sample_plot = existing_simulations[0].with_name(
            existing_simulations[0].name.replace("_metadata", "_plot")
        )
        log(
            "Simulations already completed for this subject.\n"
            f"Inspect existing outputs under: {sample_plot}"
        )
        sys.exit(0)

analysis_run_id = base_analysis_run_id
rsa_matrices_path = analysis_output_dir / f"{analysis_run_id}_dRSA_matrices.npy"
metadata_path = analysis_output_dir / f"{analysis_run_id}_metadata.json"
plot_path = analysis_output_dir / f"{analysis_run_id}_plot.png"
lag_curves_path = analysis_output_dir / f"{analysis_run_id}_lag_curves.npy"

cache_root = subject_results_dir / "cache"
subsample_cache_dir = cache_root / "subsamples"
cache_root.mkdir(parents=True, exist_ok=True)
subsample_cache_dir.mkdir(parents=True, exist_ok=True)

mask_signature = compute_mask_signature(mask)
cache_onset_signature = (
    word_onset_candidate_signature if args.lock_subsample_to_word_onset else "unused"
)
subsample_cache_key = json.dumps(
    {
        "tps": tps,
        "subsample_tps": subsample_tps,
        "n_subsamples": n_subsamples,
        "iterations": subsampling_iterations,
        "mask_signature": mask_signature,
        "lock_subsample_to_word_onset": args.lock_subsample_to_word_onset,
        "allow_overlap": args.allow_overlap,
        "word_onset_signature": cache_onset_signature,
        "word_onset_alignment": (
            args.word_onset_alignment if args.lock_subsample_to_word_onset else None
        ),
        "word_onset_candidate_within_bounds": (
            word_onset_candidate_within_bounds if args.lock_subsample_to_word_onset else None
        ),
        "word_onset_count": word_onset_count if args.lock_subsample_to_word_onset else None,
        "random_state": subsampling_random_state,
    },
    sort_keys=True,
)
subsample_cache_name = hashlib.sha1(subsample_cache_key.encode("utf-8")).hexdigest()
subsample_cache_path = subsample_cache_dir / f"subsamples_{subsample_cache_name}.npy"
subsample_plot_path = subsample_cache_path.with_suffix(".png")

# ===== run dRSA =====

if subsample_cache_path.exists():
    subsample_indices = np.load(subsample_cache_path)
    log(f"\u2713 loaded cached subsample indices ({subsample_cache_name})")
    diagnostics_needed = not subsample_plot_path.exists()
else:
    subsample_indices = subsampling(
        tps,
        subsample_tps,
        n_subsamples,
        subsampling_iterations,
        mask=mask,
        random_state=subsampling_random_state,
        candidate_starts=(
            word_onset_candidate_starts if args.lock_subsample_to_word_onset else None
        ),
        allow_overlap=args.allow_overlap,
    )
    np.save(subsample_cache_path, subsample_indices, allow_pickle=False)
    log(f"\u2713 subsample indices (cached id {subsample_cache_name})")
    diagnostics_needed = True

if diagnostics_needed:
    try:
        save_subsample_diagnostics(
            subsample_indices=subsample_indices,
            subsample_cache_path=subsample_cache_path,
            mask_array=mask,
            word_onsets_path=word_onsets_path,
            sampling_rate=resolution,
            zoom_window_seconds=(30.0, 45.0),
        )
    except Exception as exc:
        log(f"Warning: failed to create subsample diagnostic plot: {exc}")

lag_bootstrap_settings = {
    "iterations": lag_bootstrap_iterations,
    "confidence": lag_bootstrap_confidence,
    "random_state": lag_bootstrap_random_state,
}

subsample_plot_exists = subsample_plot_path.exists()



def execute_drsa_run(
    run_suffix: str | None,
    neural_signal_sets: list[tuple[str, np.ndarray]],
    neural_metrics: list[str],
    model_arrays: list[np.ndarray],
    model_labels: list[str],
    model_metrics: list[str],
    simulation_info: dict | None,
) -> dict:
    """Execute the dRSA pipeline for a specific neural/model configuration."""
    run_suffix_clean = run_suffix or "default"
    analysis_run_id = base_analysis_run_id
    if run_suffix:
        analysis_run_id = f"{analysis_run_id}_{run_suffix_clean}"
    rsa_matrices_path = analysis_output_dir / f"{analysis_run_id}_dRSA_matrices.npy"
    metadata_path = analysis_output_dir / f"{analysis_run_id}_metadata.json"
    plot_path = analysis_output_dir / f"{analysis_run_id}_plot.png"
    lag_curves_path = analysis_output_dir / f"{analysis_run_id}_lag_curves.npy"

    if len(neural_signal_sets) != len(neural_metrics):
        raise ValueError(
            f"neural_metrics length mismatch: {len(neural_metrics)} metrics for "
            f"{len(neural_signal_sets)} neural signals."
        )
    model_border_thresholds = _resolve_model_borders(model_labels)

    use_regression = rsa_computation_method != "correlation"
    target_dtype = np.float64 if double_precision else np.float32
    mem_threshold_bytes = int(args.regression_mem_threshold_gb * (1024 ** 3))
    regression_tmp_dir = cache_root / "regression_tmp"
    regression_border_dir = cache_root / "regression_borders"
    regression_buffers: list[RDMSeriesBuffer] = []
    regression_buffer_info: list[dict] | None = [] if use_regression else None
    regression_stats_summary: list[dict] = []
    regression_r2 = None
    neural_buffers: list[RDMSeriesBuffer] = []
    model_buffers: list[RDMSeriesBuffer] = []
    rsa_accumulators = None
    lag_curves_samples = None
    lag_curves_array = None

    if use_regression:
        buffer_shape = (subsampling_iterations, subsample_tps, rdm_length)
        series_bytes = (
            buffer_shape[0] * buffer_shape[1] * rdm_length * np.dtype(target_dtype).itemsize
        )
        approx_gb = series_bytes / (1024 ** 3)
        log(
            f"[{rsa_computation_method}] caching RDM stacks "
            f"({buffer_shape[0]} iters × {buffer_shape[1]} TPs × {rdm_length} distances ≈ {approx_gb:.2f} GiB per stack)"
        )
        for label, _ in neural_signal_sets:
            buf = allocate_rdm_buffer(
                f"neural_{label}",
                buffer_shape,
                target_dtype,
                mem_threshold_bytes,
                regression_tmp_dir,
            )
            neural_buffers.append(buf)
            regression_buffers.append(buf)
            regression_buffer_info.append(
                {
                    "label": label,
                    "role": "neural",
                    "gigabytes": round(buf.bytes_allocated / (1024 ** 3), 3),
                    "storage": "disk" if buf.path else "ram",
                    "path": str(buf.path) if buf.path else None,
                }
            )
        for label in model_labels:
            buf = allocate_rdm_buffer(
                f"model_{label}",
                buffer_shape,
                target_dtype,
                mem_threshold_bytes,
                regression_tmp_dir,
            )
            model_buffers.append(buf)
            regression_buffers.append(buf)
            regression_buffer_info.append(
                {
                    "label": label,
                    "role": "model",
                    "gigabytes": round(buf.bytes_allocated / (1024 ** 3), 3),
                    "storage": "disk" if buf.path else "ram",
                    "path": str(buf.path) if buf.path else None,
                }
            )
        regression_r2 = np.zeros(
            (len(neural_signal_sets), len(model_arrays), subsample_tps, subsample_tps),
            dtype=np.float32,
        )
    else:
        rsa_accumulators = np.zeros(
            (len(neural_signal_sets), len(model_arrays), subsample_tps, subsample_tps),
            dtype=np.float32,
        )
        lag_curves_samples = [[[] for _ in model_arrays] for _ in neural_signal_sets]

    store_log_interval = max(1, args.progress_log_every)
    iterations_completed = 0

    border_plot_summary = None
    try:
        log(f"... processing subsamples and computing dRSA (run: {run_suffix_clean})")
        for iteration_idx, window_indices in enumerate(subsample_indices):
            iterations_completed = iteration_idx + 1

            neural_rdm_series_per_signal = []
            for (_, neural_data_array), neural_metric in zip(neural_signal_sets, neural_metrics):
                neural_rdm = compute_rdm_series_from_indices(
                    neural_data_array, window_indices, neural_metric
                )
                if not double_precision:
                    neural_rdm = neural_rdm.astype(np.float32, copy=False)
                neural_rdm_series_per_signal.append(neural_rdm)

            model_rdm_series = []
            for model_idx, model in enumerate(model_arrays):
                model_rdm = compute_rdm_series_from_indices(
                    model, window_indices, model_metrics[model_idx]
                )
                if not double_precision:
                    model_rdm = model_rdm.astype(np.float32, copy=False)
                model_rdm_series.append(model_rdm)

            if use_regression:
                for neural_idx, neural_rdm in enumerate(neural_rdm_series_per_signal):
                    neural_buffers[neural_idx].array[iteration_idx] = neural_rdm.astype(
                        target_dtype, copy=False
                    )
                for model_idx, model_rdm in enumerate(model_rdm_series):
                    model_buffers[model_idx].array[iteration_idx] = model_rdm.astype(
                        target_dtype, copy=False
                    )
                if iteration_idx == 0 or iterations_completed % store_log_interval == 0:
                    log(
                        f"[{rsa_computation_method}] buffered {iterations_completed}/{subsampling_iterations} "
                        f"iterations (run: {run_suffix_clean})"
                    )
            else:
                for neural_idx, neural_rdm in enumerate(neural_rdm_series_per_signal):
                    for model_idx, model_rdm in enumerate(model_rdm_series):
                        rsa_matrix_it, lag_curve_it = compute_rsa_matrix_corr(
                            neural_rdm,
                            model_rdm,
                            return_lag_curves=True,
                            lag_window=adtw_in_tps,
                        )
                        rsa_accumulators[neural_idx, model_idx] += rsa_matrix_it
                        lag_curves_samples[neural_idx][model_idx].append(
                            lag_curve_it.astype(np.float32, copy=False)
                        )

        if iterations_completed != subsampling_iterations:
            raise RuntimeError(
                f"Expected {subsampling_iterations} iterations, but processed {iterations_completed}."
            )

        if use_regression:
            regression_config = RegressionConfig(
                method=rsa_computation_method,
                alpha=args.regression_alpha,
                l1_ratio=args.regression_l1_ratio,
                variance_threshold=args.pcr_variance_threshold,
                border_threshold=args.regression_border_threshold,
                border_thresholds=model_border_thresholds,
                plot_borders=args.plot_regression_borders,
                border_plot_dir=(
                    regression_border_dir if args.plot_regression_borders else None
                ),
                progress_iterations=args.progress_log_every,
                neural_progress_step=args.progress_neural_step,
            )
            if args.plot_regression_borders:
                regression_border_dir.mkdir(parents=True, exist_ok=True)
            for buf in neural_buffers + model_buffers:
                buf.flush()

            rsa_matrices = np.zeros(
                (len(neural_signal_sets), len(model_arrays), subsample_tps, subsample_tps),
                dtype=np.float32,
            )
            for neural_idx, (neural_label, _) in enumerate(neural_signal_sets):
                prefix = f"[{rsa_computation_method}][{run_suffix_clean}:{slugify_label(neural_label)}]"
                logger_fn = lambda message, prefix=prefix: log(f"{prefix} {message}")
                result = compute_dRSA_regression(
                    neural_buffers[neural_idx].array,
                    [buf.array for buf in model_buffers],
                    adtw_in_tps,
                    regression_config,
                    logger=logger_fn,
                    model_labels=model_labels,
                )
                rsa_matrices[neural_idx] = result.betas
                regression_r2[neural_idx] = result.r2
                if border_plot_summary is None and result.stats.get("border_plots"):
                    border_plot_summary = result.stats["border_plots"]
                stats_entry = {
                    "neural_label": neural_label,
                    **result.stats,
                    "r2_mean": float(np.mean(result.r2)),
                    "r2_max": float(np.max(result.r2)),
                }
                regression_stats_summary.append(stats_entry)
                log(
                    f"[{rsa_computation_method}] neural '{neural_label}' R² mean "
                    f"{stats_entry['r2_mean']:.3f} (max {stats_entry['r2_max']:.3f})"
                )
                if "pca_components" in result.stats:
                    pcs = result.stats["pca_components"]
                    log(
                        f"[{rsa_computation_method}] PCA components retained for '{neural_label}': "
                        f"min={pcs['min']} median={pcs['median']:.1f} max={pcs['max']}"
                    )

            lag_len = 2 * adtw_in_tps + 1
            lag_curves_array = np.zeros(
                (len(neural_signal_sets), len(model_arrays), lag_len), dtype=np.float32
            )
            for neural_idx in range(len(neural_signal_sets)):
                for model_idx in range(len(model_arrays)):
                    _, lag_curve_vals = compute_lag_correlation(
                        rsa_matrices[neural_idx, model_idx], adtw_in_tps
                    )
                    lag_curves_array[neural_idx, model_idx] = lag_curve_vals.astype(
                        np.float32, copy=False
                    )
            if save_lag_curves:
                np.save(lag_curves_path, lag_curves_array)
        else:
            rsa_matrices = rsa_accumulators / iterations_completed
            lag_curves_per_signal_model = [
                [np.stack(curves, axis=0) if curves else None for curves in lag_curve_list]
                for lag_curve_list in lag_curves_samples
            ]
            lag_curves_complete = all(
                all(curves is not None for curves in model_list)
                for model_list in lag_curves_per_signal_model
            )
            if lag_curves_complete:
                lag_curves_array = np.stack(
                    [np.stack(model_list, axis=0) for model_list in lag_curves_per_signal_model],
                    axis=0,
                ).astype(np.float32, copy=False)
                if save_lag_curves:
                    np.save(lag_curves_path, lag_curves_array)
    finally:
        for buf in regression_buffers:
            buf.cleanup()

    log("✓ compute dRSA matrices")

    if save_rsa_matrices:
        rsa_matrices_arr = np.asarray(rsa_matrices, dtype=target_dtype)
        np.save(rsa_matrices_path, rsa_matrices_arr)

    plot_exists = plot_path.exists()
    subsample_plot_exists = subsample_plot_path.exists()

    regression_metadata = {
        "method": rsa_computation_method,
        "use_regression": use_regression,
        "alpha": args.regression_alpha if use_regression else None,
        "l1_ratio": args.regression_l1_ratio if use_regression else None,
        "variance_threshold": args.pcr_variance_threshold if use_regression else None,
        "border_threshold": args.regression_border_threshold if use_regression else None,
        "border_thresholds": model_border_thresholds if use_regression else None,
        "plot_regression_borders": args.plot_regression_borders if use_regression else None,
        "mem_threshold_gb": args.regression_mem_threshold_gb if use_regression else None,
        "buffer_info": regression_buffer_info,
        "stats_per_neural": regression_stats_summary if regression_stats_summary else None,
        "tmp_dir": str(regression_tmp_dir) if use_regression else None,
        "border_plots": border_plot_summary if use_regression else None,
        "r2_global_mean": float(np.mean(regression_r2)) if regression_r2 is not None else None,
        "r2_global_max": float(np.max(regression_r2)) if regression_r2 is not None else None,
    }

    analysis_metadata = {
        "subject": subject,
        "subject_label": subject_label,
        "session_label": session_label,
        "task_label": task_label,
        "frequency_band": frequency_band,
        "lock_subsample_to_word_onset": args.lock_subsample_to_word_onset,
        "word_onset_alignment": args.word_onset_alignment,
        "allow_overlap": args.allow_overlap,
        "meg_path": str(MEG_path),
        "model_paths": model_paths,
        "mask_paths": mask_paths,
        "rsa_computation_method": rsa_computation_method,
        "neural_rdm_metric": neural_rdm_metric,
        "neural_signal_metrics": neural_metrics,
        "model_rdm_metrics": model_metrics,
        "double_precision": double_precision,
        "save_rsa_matrices": save_rsa_matrices,
        "save_lag_curves": save_lag_curves,
        "analysis_parameters": analysis_parameters,
        "lag_bootstrap_settings": lag_bootstrap_settings,
        "selected_model_labels": model_labels,
        "neural_signal_labels": [label for label, _ in neural_signal_sets],
        "analysis_name": analysis_name,
        "analysis_run_id": analysis_run_id,
        "simulation": simulation_info,
        "regression": regression_metadata,
        "word_onsets": {
            "expected_path": str(word_onsets_expected_path),
            "resolved_path": str(word_onsets_path) if word_onsets_path else None,
            "count": word_onset_count if word_onset_seconds is not None else None,
            "seconds_signature": word_onset_seconds_signature,
            "candidate_total": (
                word_onset_candidate_total if word_onset_seconds is not None else None
            ),
            "candidate_within_bounds": (
                word_onset_candidate_within_bounds if word_onset_seconds is not None else None
            ),
            "candidate_trimmed_for_window": (
                word_onset_candidate_trimmed if word_onset_seconds is not None else None
            ),
            "candidate_signature": word_onset_candidate_signature,
            "invalid_entries_dropped": (
                word_onset_invalid_dropped if word_onset_seconds is not None else None
            ),
            "loading_error": word_onset_loading_error,
            "sampling_rate_hz": resolution,
            "subsample_window_seconds": SubSampleDurSec,
            "lock_subsample_to_word_onset": args.lock_subsample_to_word_onset,
            "alignment": args.word_onset_alignment if word_onset_seconds is not None else None,
            "allow_overlap": args.allow_overlap,
        },
        "analysis_paths": {
            "results_root": str(results_root),
            "analysis_root": str(analysis_root),
            "single_subjects": str(single_subjects_dir),
            "simulations": str(simulations_dir),
            "group_level": str(group_level_dir),
            "subject_results_dir": str(analysis_output_dir),
            "standard_results_dir": str(subject_results_dir),
            "concatenation_metadata": str(concatenation_metadata_path),
        },
        "subsample_cache": {
            "id": subsample_cache_name,
            "path": str(subsample_cache_path),
            "plot": str(subsample_plot_path),
            "plot_exists": subsample_plot_exists,
            "lock_subsample_to_word_onset": args.lock_subsample_to_word_onset,
            "word_onset_alignment": (
                args.word_onset_alignment if args.lock_subsample_to_word_onset else None
            ),
            "allow_overlap": args.allow_overlap,
            "word_onset_signature": cache_onset_signature,
            "word_onset_seconds_signature": word_onset_seconds_signature,
            "word_onset_candidate_count": (
                word_onset_candidate_within_bounds if word_onset_seconds is not None else None
            ),
            "word_onset_candidate_trimmed": (
                word_onset_candidate_trimmed if word_onset_seconds is not None else None
            ),
        },
        "rsa_matrix_shape": list(rsa_matrices.shape),
        "lag_curves_shape": (
            list(lag_curves_array.shape) if lag_curves_array is not None else None
        ),
        "outputs": {
            "rsa_matrices": str(rsa_matrices_path) if save_rsa_matrices else None,
            "lag_curves": (
                str(lag_curves_path)
                if save_lag_curves and lag_curves_array is not None
                else None
            ),
            "plot": str(plot_path) if plot_exists else None,
            "plot_target": str(plot_path),
            "output_dir": str(analysis_output_dir),
            "cache_root": str(cache_root),
            "metadata": str(metadata_path),
            "subsample_diagnostics": (
                str(subsample_plot_path) if subsample_plot_exists else None
            ),
            "subsample_diagnostics_target": str(subsample_plot_path),
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(analysis_metadata, f, indent=2)

    return {
        "analysis_run_id": analysis_run_id,
        "metadata_path": metadata_path,
        "rsa_matrices_path": rsa_matrices_path if save_rsa_matrices else None,
        "lag_curves_path": (
            lag_curves_path if save_lag_curves and lag_curves_array is not None else None
        ),
        "plot_path": plot_path,
    }

if args.simulation:
    neural_reference_label = f"{subject_label} Neural"
    neural_reference_array = np.atleast_2d(selected_neural_data)

    simulation_targets = [
        {
            "label": neural_reference_label,
            "neural_tuple": (neural_reference_label, selected_neural_data),
            "neural_metric": neural_rdm_metric,
            "origin": "neural",
            "model_arrays": [neural_reference_array],
            "model_labels": [neural_reference_label],
            "model_metrics": [neural_rdm_metric],
            "models_include_neural_signal": False,
        }
    ]
    for label, model, metric in zip(
        selected_models_labels, selected_models, model_rdm_metrics
    ):
        simulation_targets.append(
            {
                "label": label,
                "neural_tuple": (label, model),
                "neural_metric": metric,
                "origin": "model",
                "model_arrays": selected_models,
                "model_labels": selected_models_labels,
                "model_metrics": model_rdm_metrics,
                "models_include_neural_signal": False,
            }
        )
    total_runs = len(simulation_targets)

    for run_idx, target in enumerate(simulation_targets):
        neural_label = target["label"]
        run_suffix = f"sim_{run_idx:02d}_{slugify_label(neural_label)}"
        neural_signal_sets = [target["neural_tuple"]]
        neural_metrics = [target["neural_metric"]]
        simulation_info = {
            "enabled": True,
            "run_index": run_idx,
            "total_runs": total_runs,
            "neural_source_label": neural_label,
            "neural_source_origin": target["origin"],
            "models_include_neural_signal": target["models_include_neural_signal"],
            "neural_metric": target["neural_metric"],
            "output_dir": str(analysis_output_dir),
        }
        artifacts = execute_drsa_run(
            run_suffix,
            neural_signal_sets,
            neural_metrics,
            target["model_arrays"],
            target["model_labels"],
            target["model_metrics"],
            simulation_info,
        )
        log(
            f"✓ simulation run {run_idx + 1}/{total_runs} completed "
            f"({neural_label}) → {artifacts['analysis_run_id']}"
        )
else:
    artifacts = execute_drsa_run(
        None,
        default_neural_signal_sets,
        default_neural_metrics,
        selected_models,
        selected_models_labels,
        model_rdm_metrics,
        {"enabled": False, "output_dir": str(analysis_output_dir)},
    )
    log(
        f"✓ dRSA run completed → {artifacts['analysis_run_id']} "
        f"(metadata: {artifacts['metadata_path']})"
    )
