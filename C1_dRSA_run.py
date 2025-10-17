import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import textwrap
import mne
from functions.core_functions import subsampling_iterator, compute_lag_correlation, compute_rsa_matrix_corr, compute_rdm_series_from_indices
from functions.PCR_alpha import compute_rsa_matrix_PCR
from functions.generic_helpers import read_repository_root


def bootstrap_mean_ci(data, n_bootstraps=1000, confidence=0.95, random_state=None):
    """
    Compute a bootstrap confidence interval for the mean across axis=0.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_points)
        Input sample of curves (e.g., lag correlations per iteration).
    n_bootstraps : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level between 0 and 1.
    random_state : int or np.random.Generator, optional
        Reproducibility control.

    Returns
    -------
    lower, upper : ndarray
        Lower and upper bounds of the confidence interval (shape: n_points).
    """
    curves = np.asarray(data)
    if curves.ndim != 2:
        raise ValueError("`data` must be a 2D array (samples, points).")
    if curves.shape[0] == 0:
        raise ValueError("At least one sample is required for bootstrapping.")

    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))
    n_samples, n_points = curves.shape
    indices = rng.integers(0, n_samples, size=(n_bootstraps, n_samples))
    sample_means = curves[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(sample_means, 100 * alpha, axis=0)
    upper = np.percentile(sample_means, 100 * (1 - alpha), axis=0)
    return lower, upper

# repository root
repo_root = read_repository_root()

# ===== load data =====
# paths
subject = int(sys.argv[1])
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
    os.path.join(
        repo_root,
        "derivatives",
        "preprocessed",
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

def _register_model(path, label, metric):
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
_register_model(wordfreq_path, "Word Frequency", "euclidean")
_register_model(glove_path, "GloVe", "correlation")
_register_model(glove_norm_path, "GloVe Norm", "euclidean")


# ===== settings =====

save_rsa_matrices = True
save_plots = True

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

analysis_parameters = {
    "n_subsamples": n_subsamples,
    "subsampling_iterations": subsampling_iterations,
    "subsample_duration_sec": SubSampleDurSec,
    "averaging_diagonal_time_window_sec": averaging_diagonal_time_window_sec,
    "resolution_hz": resolution,
    "tps": tps,
    "averaging_window_tps": adtw_in_tps,
    "subsample_tps": subsample_tps,
}

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
analysis_run_id = f"sub{subject}_res{resolution}_{rsa_computation_method}"
rsa_matrices_path = os.path.join(results_dir, f"{analysis_run_id}_dRSA_matrices.npy")
metadata_path = os.path.join(results_dir, f"{analysis_run_id}_metadata.json")
plot_path = os.path.join(results_dir, f"{analysis_run_id}_plot.png")

# ===== run dRSA =====

subsample_iter = subsampling_iterator(
    tps,
    subsample_tps,
    n_subsamples,
    subsampling_iterations,
    mask=mask,
    random_state=None,
)
print("\u2713 subsample indices")

rsa_accumulators = [np.zeros((subsample_tps, subsample_tps), dtype=np.float32) for _ in selected_models]
lag_curves_samples = [[] for _ in selected_models]

print("... processing subsamples and computing dRSA")
iterations_completed = 0
for iteration_idx, window_indices in enumerate(subsample_iter):
    iterations_completed = iteration_idx + 1

    neural_rdm = compute_rdm_series_from_indices(selected_neural_data, window_indices, neural_rdm_metric)
    if not double_precision:
        neural_rdm = neural_rdm.astype(np.float32, copy=False)

    for model_idx, model in enumerate(selected_models):
        model_rdm = compute_rdm_series_from_indices(model, window_indices, model_rdm_metrics[model_idx])
        if not double_precision:
            model_rdm = model_rdm.astype(np.float32, copy=False)

        if rsa_computation_method == 'correlation':
            rsa_matrix_it, lag_curve_it = compute_rsa_matrix_corr(
                neural_rdm,
                model_rdm,
                return_lag_curves=True,
                lag_window=adtw_in_tps,
            )
            rsa_accumulators[model_idx] += rsa_matrix_it
            lag_curves_samples[model_idx].append(lag_curve_it.astype(np.float32, copy=False))
        elif rsa_computation_method == 'PCR':
            # PCR path requires all iterations; accumulate per iteration for later processing
            raise NotImplementedError("Streaming PCR computation is not implemented.")
        else:
            raise ValueError(f"Unsupported rsa_computation_method: {rsa_computation_method}")

if iterations_completed != subsampling_iterations:
    raise RuntimeError(
        f"Expected {subsampling_iterations} iterations, but processed {iterations_completed}."
    )

rsa_matrices = [accumulator / iterations_completed for accumulator in rsa_accumulators]
lag_curves_per_model = [np.stack(curves, axis=0) if curves else None for curves in lag_curves_samples]
print("\u2713 compute dRSA matrices")

# saving dRSA matrices:
if save_rsa_matrices:
    rsa_matrices_arr = np.array(rsa_matrices, dtype=float) # turn into an array
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
    "save_plots": save_plots,
    "analysis_parameters": analysis_parameters,
    "lag_bootstrap_settings": lag_bootstrap_settings,
    "selected_model_labels": selected_models_labels,
    "outputs": {
        "rsa_matrices": rsa_matrices_path if save_rsa_matrices else None,
        "plot": plot_path if save_plots else None,
    },
}

with open(metadata_path, "w") as f:
    json.dump(analysis_metadata, f, indent=2)


# ===== Compute lag correlations and plot =====

fig, axs = plt.subplots(len(selected_models), 2, figsize=(11, 2.7 * len(selected_models)))

for i, rsa_matrix in enumerate(rsa_matrices):
    model_name = selected_models_labels[i]
    lag_curves = lag_curves_per_model[i] if i < len(lag_curves_per_model) else None

    lags_tp, lag_corr = compute_lag_correlation(rsa_matrix, adtw_in_tps)
    ci_lower = ci_upper = None

    if lag_curves is not None:
        ci_lower, ci_upper = bootstrap_mean_ci(
            lag_curves,
            n_bootstraps=lag_bootstrap_iterations,
            confidence=lag_bootstrap_confidence,
            random_state=lag_bootstrap_random_state,
        )

    lags_sec = lags_tp / resolution

    # RSA matrix plot
    ax_matrix = axs[i, 0] if len(selected_models) > 1 else axs[0]
    im = ax_matrix.imshow(rsa_matrix, cmap='viridis', aspect='auto')
    ax_matrix.set_title(f'dRSA Matrix     Model: {model_name}', fontsize=8)
    ax_matrix.set_xlabel('Model time', fontsize=7)
    ax_matrix.set_ylabel('Neural time', fontsize=7)
    fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

    # Lag correlation plot
    ax_lag = axs[i, 1] if len(selected_models) > 1 else axs[1]
    (lag_line,) = ax_lag.plot(lags_sec, lag_corr)
    if ci_lower is not None and ci_upper is not None:
        ax_lag.fill_between(
            lags_sec,
            ci_lower,
            ci_upper,
            color=lag_line.get_color(),
            alpha=0.25,
        )
    ax_lag.axvline(0, color='gray', linestyle='--')
    ax_lag.axhline(0, color='gray', linestyle='--')
    # ax_lag.set_title(f'Lagged Correlation', fontsize=8)
    ax_lag.set_xlabel('Lag (neural time - model time) [s]', fontsize=7)
    ax_lag.set_ylabel('Correlation', fontsize=7)

    # Mark the peak
    peak_idx = np.argmax(lag_corr)            # index of maximum correlation
    peak_lag = lags_sec[peak_idx]    # lag at maximum
    peak_val = lag_corr[peak_idx]             # maximum correlation value

    ax_lag.plot(peak_lag, peak_val, 'ro', markersize=2)    # red dot at the peak
    ax_lag.annotate(f'Peak: {peak_val:.3f}\nLag: {peak_lag:.3f}s',
                    xy=(peak_lag, peak_val),
                    xytext=(peak_lag + 0.2, peak_val),  # offset text a bit
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=7)

# Clean up layout
parameter_caption = ", ".join(f"{key}={value}" for key, value in analysis_parameters.items())
parameter_caption = "Parameters: " + parameter_caption
parameter_caption = textwrap.fill(parameter_caption, width=110)

plt.tight_layout(rect=[0, 0.18, 1, 0.97])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
fig.text(0.5, 0.05, parameter_caption, ha='center', va='center', fontsize=7)
# plt.suptitle(f'dRSA Subject {subject}', fontsize=10, fontweight='bold')
if save_plots:
    plt.savefig(plot_path, dpi=300)
# plt.show()
