import argparse
import hashlib
import json
import os
from pathlib import Path

import mne
import numpy as np
from functions.core_functions import (
    subsampling,
    compute_rsa_matrix_corr,
    compute_rdm_series_from_indices,
    save_subsample_diagnostics,
)
from functions.PCR_alpha import compute_rsa_matrix_PCR
from functions.generic_helpers import ensure_analysis_directories, read_repository_root


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

# CLI + repository root
args = parse_args()

if not args.lock_subsample_to_word_onset and args.word_onset_alignment != "center":
    print(
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

print(f"Analysis name: {analysis_name}")
print(f"Analysis directory: {analysis_root}")
print(f"Subject-level outputs will be written to: {single_subjects_dir}")

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
print(f"subject: {subject}")

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

model_filter_env = os.getenv("DRSA_MODELS")
if model_filter_env:
    model_filter = {
        token.strip().lower() for token in model_filter_env.split(",") if token.strip()
    }
else:
    model_filter = None


def _register_model(path, label, metric):
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

_register_model(envelope_path, "Envelope", "euclidean")
_register_model(voicing_path, "Phoneme Voicing", "euclidean")
_register_model(wordfreq_path, "Word Frequency", "euclidean")
_register_model(glove_path, "GloVe", "correlation")
_register_model(glove_norm_path, "GloVe Norm", "euclidean")

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
    _register_model(gpt_next_path, "GPT Next-Token", "correlation")

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
    _register_model(gpt_surp_path, "GPT Surprisal", "euclidean")


# Report selected models to log
if model_filter_env:
    print(f"DRSA_MODELS filter: {model_filter_env}")
if selected_models_labels:
    print("Selected models (label | features | metric | path):")
    for mdl, lbl, mtr in zip(selected_models, selected_models_labels, model_rdm_metrics):
        feats = int(np.atleast_2d(mdl).shape[0])
        path = model_paths.get(lbl, "<unknown>")
        print(f"  - {lbl} | {feats} | {mtr} | {path}")
else:
    print("Warning: no models selected after filtering and existence checks.")


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

lag_bootstrap_iterations = 1000
lag_bootstrap_confidence = 0.95
lag_bootstrap_random_state = 0


# ===== select data and set RDM metrics =====                    

selected_neural_data = neural_data

# Hard-coded neural subsets (currently three identical MEG datasets).
neural_signal_sets = [
    ("MEG Full 1", selected_neural_data),
    ("MEG Full 2", selected_neural_data),
    ("MEG Full 3", selected_neural_data),
]

neural_signal_labels = [label for label, _ in neural_signal_sets]

selected_models = [np.atleast_2d(model) for model in selected_models]  # ensure 2D arrays. If 1D, add feature axis.

neural_rdm_metric = 'correlation'
# model_rdm_metrics already defined during registration; valid options include:
# euclidean - cosine - hamming - correlation - jaccard

rsa_computation_method = 'correlation'
# accepts: correlation - PCR

# check if lengths match
if not (len(selected_models) == len(selected_models_labels) == len(model_rdm_metrics)):
    raise ValueError(
        f"Lengths do not match: "
        f"models={len(selected_models)}, "
        f"labels={len(selected_models_labels)}, "
        f"metrics={len(model_rdm_metrics)}"
    )


# ===== set parameters =====

n_subsamples = 150
subsampling_iterations = 100

SubSampleDurSec = 5
averaging_diagonal_time_window_sec = 3

resolution = 100 # in Hz - change as needed

tps = selected_neural_data.shape[1]
adtw_in_tps = averaging_diagonal_time_window_sec * resolution
# this is to avoid noisy diagonal averaging in the dRSA matrix edges
# and also, not to use the models outside the -3 +3 window for the PCR.
subsample_tps = SubSampleDurSec * resolution # subsample size in tps - also number of RDMs calculated for each subsample

subsampling_random_state = None

analysis_parameters = {
    "n_subsamples": n_subsamples,
    "subsampling_iterations": subsampling_iterations,
    "subsample_duration_sec": SubSampleDurSec,
    "averaging_diagonal_time_window_sec": averaging_diagonal_time_window_sec,
    "resolution_hz": resolution,
    "tps": tps,
    "averaging_window_tps": adtw_in_tps,
    "subsample_tps": subsample_tps,
    "subsampling_random_state": subsampling_random_state,
    "word_onset_alignment": (
        args.word_onset_alignment if args.lock_subsample_to_word_onset else None
    ),
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
        print(f"Warning: failed to load word onset timestamps from {word_onsets_path}: {exc}")
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
            print(f"Warning: word onset file {word_onsets_path} contains no valid entries.")
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
            print(
                f"Loaded {word_onset_count} word onset timestamps (dropped {word_onset_invalid_dropped} invalid entries)."
            )
        else:
            print(f"Loaded {word_onset_count} word onset timestamps.")
    if args.lock_subsample_to_word_onset:
        print(
            f"Locking subsample windows to {word_onset_candidate_within_bounds} "
            f"{args.word_onset_alignment}-aligned onset start(s) (allow_overlap={args.allow_overlap})."
        )
        if word_onset_candidate_trimmed > 0:
            print(
                f"  Ignored {word_onset_candidate_trimmed} onset(s) that cannot fit a {SubSampleDurSec}-s window at {resolution} Hz."
            )
    elif word_onset_candidate_within_bounds:
        print(
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
analysis_run_id = f"{subject_label}_res{resolution}_{rsa_computation_method}"
subject_results_dir = single_subjects_dir
rsa_matrices_path = subject_results_dir / f"{analysis_run_id}_dRSA_matrices.npy"
metadata_path = subject_results_dir / f"{analysis_run_id}_metadata.json"
plot_path = subject_results_dir / f"{analysis_run_id}_plot.png"
lag_curves_path = subject_results_dir / f"{analysis_run_id}_lag_curves.npy"

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
    print(f"\u2713 loaded cached subsample indices ({subsample_cache_name})")
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
    print(f"\u2713 subsample indices (cached id {subsample_cache_name})")
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
        print(f"Warning: failed to create subsample diagnostic plot: {exc}")

rsa_accumulators = np.zeros(
    (len(neural_signal_sets), len(selected_models), subsample_tps, subsample_tps),
    dtype=np.float32,
)
lag_curves_samples = [[[] for _ in selected_models] for _ in neural_signal_sets]

print("... processing subsamples and computing dRSA")
iterations_completed = 0
for iteration_idx, window_indices in enumerate(subsample_indices):
    iterations_completed = iteration_idx + 1

    neural_rdm_series_per_signal = []
    for neural_label, neural_data_array in neural_signal_sets:
        neural_rdm = compute_rdm_series_from_indices(neural_data_array, window_indices, neural_rdm_metric)
        if not double_precision:
            neural_rdm = neural_rdm.astype(np.float32, copy=False)
        neural_rdm_series_per_signal.append(neural_rdm)

    model_rdm_series = []
    for model_idx, model in enumerate(selected_models):
        model_rdm = compute_rdm_series_from_indices(model, window_indices, model_rdm_metrics[model_idx])
        if not double_precision:
            model_rdm = model_rdm.astype(np.float32, copy=False)
        model_rdm_series.append(model_rdm)

    for neural_idx, neural_rdm in enumerate(neural_rdm_series_per_signal):
        for model_idx, model_rdm in enumerate(model_rdm_series):
            if rsa_computation_method == 'correlation':
                rsa_matrix_it, lag_curve_it = compute_rsa_matrix_corr(
                    neural_rdm,
                    model_rdm,
                    return_lag_curves=True,
                    lag_window=adtw_in_tps,
                )
                rsa_accumulators[neural_idx, model_idx] += rsa_matrix_it
                lag_curves_samples[neural_idx][model_idx].append(lag_curve_it.astype(np.float32, copy=False))
            elif rsa_computation_method == 'PCR':
                # PCR path requires all iterations; accumulate per iteration for later processing
                raise NotImplementedError("Streaming PCR computation is not implemented.")
            else:
                raise ValueError(f"Unsupported rsa_computation_method: {rsa_computation_method}")

if iterations_completed != subsampling_iterations:
    raise RuntimeError(
        f"Expected {subsampling_iterations} iterations, but processed {iterations_completed}."
    )

rsa_matrices = rsa_accumulators / iterations_completed
lag_curves_per_signal_model = [
    [
        np.stack(curves, axis=0) if curves else None
        for curves in lag_curve_list
    ]
    for lag_curve_list in lag_curves_samples
]

lag_curves_array = None
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

print("\u2713 compute dRSA matrices")

# saving dRSA matrices:
if save_rsa_matrices:
    target_dtype = np.float64 if double_precision else np.float32
    rsa_matrices_arr = np.asarray(rsa_matrices, dtype=target_dtype)
    np.save(rsa_matrices_path, rsa_matrices_arr)

lag_bootstrap_settings = {
    "iterations": lag_bootstrap_iterations,
    "confidence": lag_bootstrap_confidence,
    "random_state": lag_bootstrap_random_state,
}

plot_exists = plot_path.exists()
subsample_plot_exists = subsample_plot_path.exists()

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
    "model_rdm_metrics": model_rdm_metrics,
    "double_precision": double_precision,
    "save_rsa_matrices": save_rsa_matrices,
    "save_lag_curves": save_lag_curves,
    "analysis_parameters": analysis_parameters,
    "lag_bootstrap_settings": lag_bootstrap_settings,
    "selected_model_labels": selected_models_labels,
    "neural_signal_labels": neural_signal_labels,
    "analysis_name": analysis_name,
    "word_onsets": {
        "expected_path": str(word_onsets_expected_path),
        "resolved_path": str(word_onsets_path) if word_onsets_path else None,
        "count": word_onset_count if word_onset_seconds is not None else None,
        "seconds_signature": word_onset_seconds_signature,
        "candidate_total": word_onset_candidate_total if word_onset_seconds is not None else None,
        "candidate_within_bounds": word_onset_candidate_within_bounds if word_onset_seconds is not None else None,
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
        "group_level": str(group_level_dir),
        "subject_results_dir": str(subject_results_dir),
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
        "cache_root": str(cache_root),
        "metadata": str(metadata_path),
        "subsample_diagnostics": str(subsample_plot_path) if subsample_plot_exists else None,
        "subsample_diagnostics_target": str(subsample_plot_path),
    },
}

with open(metadata_path, "w") as f:
    json.dump(analysis_metadata, f, indent=2)
