import json
import os
import sys
import hashlib

import numpy as np
import mne
from functions.core_functions import (
    subsampling,
    compute_rsa_matrix_corr,
    compute_rdm_series_from_indices,
)
from functions.PCR_alpha import compute_rsa_matrix_PCR
from functions.generic_helpers import read_repository_root


def compute_mask_signature(mask_array):
    if mask_array is None:
        return "nomask"
    mask_bool = np.asarray(mask_array, dtype=bool)
    packed = np.packbits(mask_bool)
    return hashlib.sha1(packed.tobytes()).hexdigest()

# repository root
repo_root = read_repository_root()

# ===== load data =====
# paths
subject_arg = sys.argv[1]
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


MEG_path = os.path.join(
    repo_root,
    "derivatives",
    "preprocessed",
    subject_label,
    "concatenated",
    f"{subject_label}_concatenated_meg_100Hz.npy",
)
frequency_band = 'unfiltered'

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
}

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
analysis_run_id = f"{subject_label}_res{resolution}_{rsa_computation_method}"
rsa_matrices_path = os.path.join(results_dir, f"{analysis_run_id}_dRSA_matrices.npy")
metadata_path = os.path.join(results_dir, f"{analysis_run_id}_metadata.json")
plot_path = os.path.join(results_dir, f"{analysis_run_id}_plot.png")
lag_curves_path = os.path.join(results_dir, f"{analysis_run_id}_lag_curves.npy")

cache_root = os.path.join(results_dir, "cache")
subsample_cache_dir = os.path.join(cache_root, "subsamples")
os.makedirs(subsample_cache_dir, exist_ok=True)

mask_signature = compute_mask_signature(mask)
subsample_cache_key = json.dumps(
    {
        "tps": tps,
        "subsample_tps": subsample_tps,
        "n_subsamples": n_subsamples,
        "iterations": subsampling_iterations,
        "mask_signature": mask_signature,
        "random_state": subsampling_random_state,
    },
    sort_keys=True,
)
subsample_cache_name = hashlib.sha1(subsample_cache_key.encode("utf-8")).hexdigest()
subsample_cache_path = os.path.join(subsample_cache_dir, f"subsamples_{subsample_cache_name}.npy")

# ===== run dRSA =====

if os.path.exists(subsample_cache_path):
    subsample_indices = np.load(subsample_cache_path)
    print(f"\u2713 loaded cached subsample indices ({subsample_cache_name})")
else:
    subsample_indices = subsampling(
        tps,
        subsample_tps,
        n_subsamples,
        subsampling_iterations,
        mask=mask,
        random_state=subsampling_random_state,
    )
    os.makedirs(cache_root, exist_ok=True)
    np.save(subsample_cache_path, subsample_indices, allow_pickle=False)
    print(f"\u2713 subsample indices (cached id {subsample_cache_name})")

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

analysis_metadata = {
    "subject": subject,
    "subject_label": subject_label,
    "session_label": session_label,
    "task_label": task_label,
    "frequency_band": frequency_band,
    "meg_path": MEG_path,
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
    "subsample_cache": {
        "id": subsample_cache_name,
        "path": subsample_cache_path,
    },
    "rsa_matrix_shape": list(rsa_matrices.shape),
    "lag_curves_shape": (
        list(lag_curves_array.shape) if lag_curves_array is not None else None
    ),
    "outputs": {
        "rsa_matrices": rsa_matrices_path if save_rsa_matrices else None,
        "lag_curves": (
            lag_curves_path if save_lag_curves and lag_curves_array is not None else None
        ),
        "plot": None,
        "plot_target": plot_path,
    },
}

with open(metadata_path, "w") as f:
    json.dump(analysis_metadata, f, indent=2)
