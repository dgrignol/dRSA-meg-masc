import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from functions.core_functions import subsampling, apply_subsampling_indices, compute_lag_correlation, compute_rsa_matrix_corr, compute_rdm_series
from functions.PCR_alpha import compute_rsa_matrix_PCR


# ===== load data =====

subject = int(sys.argv[1]); print(f'subject: {subject}')
frequency_band = 'unfiltered'
region = 'all'

# shape of loaded data should be (features, tps)
neural_data = np.load(f'/path/to/{region}-{frequency_band}/sub{subject}-{frequency_band}-{region}.npy') # change as needed

model1 = np.load('path/to/model1.npy')
model2 = np.load('path/to/model2.npy')


# ===== settings =====

save_rsa_matrices = True
save_plots = True

mask = True
if mask:
    mask = np.squeeze(np.load('path/to/mask.npy'))
else:
    mask = None

double_precision = False


# ===== select data and set RDM metrics =====                    

selected_neural_data = neural_data

selected_models = [model1, model2]

selected_models_labels = ['Model 1 Name', 'Model 2 Name']

neural_rdm_metric = 'correlation'

model_rdm_metrics = ['correlation', 'cosine']
# accepts: euclidean - cosine - hamming - correlation - jaccard

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

n_subsamples = 20
subsampling_iterations = 100

SubSampleDurSec = 5
averaging_diagonal_time_window_sec = 3

resolution = 100

tps = selected_neural_data.shape[1]
adtw_in_tps = averaging_diagonal_time_window_sec * resolution
# this is to avoid noisy diagonal averaging in the dRSA matrix edges
# and also, not to use the models outside the -3 +3 window for the PCR.
subsample_tps = SubSampleDurSec * resolution # subsample size in tps - also number of RDMs calculated for each subsample


# ===== run dRSA =====

# get subsample indices
subsample_indices = subsampling(tps, subsample_tps, n_subsamples, subsampling_iterations, mask=mask, random_state=None)
print("\u2713 subsample indices")

# use the indices to subsample from data
subsampled_neural_data = apply_subsampling_indices(selected_neural_data, subsample_indices)
subsampled_models_features = [apply_subsampling_indices(model, subsample_indices) for model in selected_models]
print("\u2713 extract subsamples from data")

# in all iterations, compute rdm for each time point
neural_rdm_series = compute_rdm_series(subsampled_neural_data, neural_rdm_metric) 
# each time point has a separate n_subsamples * n_subsamples RDM (but flattened)
print("\u2713 compute neural RDM series")
model_rdm_series_list = [compute_rdm_series(subsampled_models_features[i], model_rdm_metrics[i]) for i in range(len(selected_models))]
print("\u2713 compute model RDM series")

if not double_precision:
    # make them float32 for memory efficiency
    neural_rdm_series = neural_rdm_series.astype(np.float32)
    model_rdm_series_list = [model_rdm_series.astype(np.float32) for model_rdm_series in model_rdm_series_list]

# compute rsa matrices for each model
print("... computing dRSA matrices")
if rsa_computation_method == 'correlation':
    rsa_matrices = [compute_rsa_matrix_corr(neural_rdm_series, model_rdm_series_list[i]) for i in range(len(model_rdm_series_list))]
elif rsa_computation_method == 'PCR':
    rsa_matrices = compute_rsa_matrix_PCR(neural_rdm_series, model_rdm_series_list, averaging_diagonal_time_tps=adtw_in_tps)
print("\u2713 compute dRSA matrices")

# saving dRSA matrices:
if save_rsa_matrices:
    rsa_matrices_arr = np.array(rsa_matrices, dtype=float) # turn into an array
    os.makedirs('results', exist_ok=True)
    output_path = f'results/sub{subject}-{frequency_band}_dRSA_matrices-{rsa_computation_method}.npy'
    np.save(output_path, rsa_matrices_arr)

    # save labels for plotting later
    with open("results/selected_model_labels.json", "w") as f:
        json.dump(selected_models_labels, f)


# ===== Compute lag correlations and plot =====

fig, axs = plt.subplots(len(selected_models), 2, figsize=(11, 2.7 * len(selected_models)))

for i, rsa_matrix in enumerate(rsa_matrices):
    model_name = selected_models_labels[i]
    lags, lag_corr = compute_lag_correlation(rsa_matrix, adtw_in_tps)

    lags = lags/resolution

    # RSA matrix plot
    ax_matrix = axs[i, 0] if len(selected_models) > 1 else axs[0]
    im = ax_matrix.imshow(rsa_matrix, cmap='viridis', aspect='auto')
    ax_matrix.set_title(f'dRSA Matrix     Model: {model_name}', fontsize=8)
    ax_matrix.set_xlabel('Model time', fontsize=7)
    ax_matrix.set_ylabel('Neural time', fontsize=7)
    fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

    # Lag correlation plot
    ax_lag = axs[i, 1] if len(selected_models) > 1 else axs[1]
    ax_lag.plot(lags, lag_corr)
    ax_lag.axvline(0, color='gray', linestyle='--')
    ax_lag.axhline(0, color='gray', linestyle='--')
    # ax_lag.set_title(f'Lagged Correlation', fontsize=8)
    ax_lag.set_xlabel('Lag (neural time - model time) [s]', fontsize=7)
    ax_lag.set_ylabel('Correlation', fontsize=7)

    # Mark the peak
    peak_idx = np.argmax(lag_corr)            # index of maximum correlation
    peak_lag = lags[peak_idx]    # lag at maximum
    peak_val = lag_corr[peak_idx]             # maximum correlation value

    ax_lag.plot(peak_lag, peak_val, 'ro', markersize=2)    # red dot at the peak
    ax_lag.annotate(f'Peak: {peak_val:.2f}\nLag: {peak_lag:.2f}s',
                    xy=(peak_lag, peak_val),
                    xytext=(peak_lag + 0.2, peak_val),  # offset text a bit
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=7)

# Clean up layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
# plt.suptitle(f'dRSA Subject {subject}', fontsize=10, fontweight='bold')
if save_plots:
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/sub_{subject}-{frequency_band}_plot-{rsa_computation_method}.png', dpi=300)
# plt.show()
