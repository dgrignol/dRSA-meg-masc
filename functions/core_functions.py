import numpy as np
from scipy.spatial.distance import pdist

def zscore(X, axis=0):
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std

def subsampling(n_timepoints, subsample_tps, n_subsamples, iterations, mask=None, random_state=None):
    """
    Generate subsample window indices for multiple iterations.

    Parameters
    ----------
    n_timepoints : int
        Total number of time points.
    subsample_tps : int
        Window length (in time points) for each subsample.
    n_subsamples : int
        Number of windows to draw per iteration.
    iterations : int
        Number of iterations.
    mask : 1D boolean array, optional
        Length n_timepoints. Windows must lie entirely where mask is True.
        If None, all timepoints are considered valid.
    random_state : int or np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    indices : ndarray, shape (iterations, n_subsamples, subsample_tps)
        Integer indices for each subsample window.
    """
    if subsample_tps <= 0:
        raise ValueError("subsample_tps must be positive.")
    if n_timepoints < subsample_tps:
        raise ValueError("Subsample size is larger than number of time points.")

    # RNG
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))

    # If no mask is given, treat everything as valid
    if mask is None:
        mask = np.ones(n_timepoints, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != n_timepoints:
            raise ValueError("mask length must equal n_timepoints.")

    # Compute valid start positions where a full window fits inside True
    # Convolution: sum over sliding window == subsample_tps means all True.
    # Result length is n_timepoints - subsample_tps + 1
    run_sum = np.convolve(mask.astype(np.int32), np.ones(subsample_tps, dtype=np.int32), mode='valid')
    valid_starts_bool = (run_sum == subsample_tps)
    possible_starts = np.flatnonzero(valid_starts_bool)  # candidates for window starts


    if possible_starts.size == 0:
        raise ValueError("No valid windows fit entirely within the True regions of the mask.")

    if possible_starts.size < n_subsamples:
        raise ValueError(
            f"Not enough valid non-overlapping start positions ({possible_starts.size}) "
            f"to draw {n_subsamples} unique starts per iteration. "
            f"Consider setting replace=True, reducing n_subsamples, or relaxing the mask."
        )

    # exclude overlaps for each iteration
    starts = np.empty((iterations, n_subsamples), dtype=int)
    for it in range(iterations):
        available_starts = possible_starts.copy()
        rng.shuffle(available_starts)
        chosen = []
        while len(chosen) < n_subsamples and available_starts.size > 0:
            start = available_starts[0]
            chosen.append(start)
            # Remove any starts that overlap this one
            available_starts = available_starts[np.abs(available_starts - start) >= subsample_tps]
        if len(chosen) < n_subsamples:
            raise ValueError(
                f"Not enough non-overlapping windows for iteration {it}. "
                f"Needed {n_subsamples}, got {len(chosen)}."
            )
        starts[it] = chosen

    # Build windows
    window = np.arange(subsample_tps, dtype=int)
    indices = starts[:, :, None] + window  # (iterations, n_subsamples, subsample_tps)
    return indices


def apply_subsampling_indices(data, subsample_indices):
    """
    Efficiently applies precomputed subsampling indices using NumPy advanced indexing.

    Parameters:
    - data: (features, timepoints) array
    - subsample_indices: (iterations, n_subsamples, subsample_tps) array

    Returns:
    - subsampled_data: (iterations, n_subsamples, features, subsample_tps) array
    """
    # Reshape data for broadcasting: (1, 1, features, timepoints)
    data_expanded = data[np.newaxis, np.newaxis, :, :]
    
    # Reshape indices for advanced indexing: (iterations, n_subsamples, 1, subsample_tps)
    indices_expanded = subsample_indices[:, :, np.newaxis, :]
    
    # Use advanced indexing to gather all subsamples at once
    subsampled_data = np.take_along_axis(
        data_expanded,
        indices_expanded,
        axis=-1  # Take along the timepoints dimension
    )
    
    return subsampled_data  # (iterations, n_subsamples, features, subsample_tps)


def compute_rdm_series(subsampled_data, metric):
    """
    Computes the time-resolved RDMs for all iterations.

    Parameters:
    - subsampled_data: shape (iterations, n_subsamples, n_features, subsample_tps)
    - metric: distance metric string (e.g., 'correlation', 'cosine', 'euclidean')

    Returns:
    - rdm_series: array of shape (iterations, subsample_tps, rdm_length), e.g. if n_subsamples = 20, 20*19/2 = 190 is the rdm_length
    """

    iterations, n_subsamples, _, subsample_tps = subsampled_data.shape
    rdm_length = n_subsamples * (n_subsamples - 1) // 2
    rdm_series = np.empty((iterations, subsample_tps, rdm_length))

    for it in range(iterations):
        for t in range(subsample_tps):
            # shape: (n_subsamples, n_features)
            data_slice = subsampled_data[it, :, :, t]
            rdm_series[it, t] = pdist(data_slice, metric=metric)

    return rdm_series


def compute_rsa_matrix_corr(neural_rdm_series, model_rdm_series):
    """
    Vectorized computation of RSA correlation matrices.

    Parameters:
    - neural_rdm_series: (iterations, subsample_tps, rdm_len)
    - model_rdm_series: (iterations, subsample_tps, rdm_len)

    Returns:
    - averaged_rsa: (subsample_tps, subsample_tps)
    """

    iterations, subsample_tps, rdm_len = neural_rdm_series.shape

    # Z-score both series across last axis (rdm_len) 
    neural_z = zscore(neural_rdm_series, axis=-1) # shape: (iterations, subsample_tps, rdm_len)
    model_z = zscore(model_rdm_series, axis=-1)

    # Compute dot products: (iterations, subsample_tps, subsample_tps)
    rsa_all = np.matmul(neural_z, model_z.transpose(0, 2, 1)) / rdm_len
    # (iterations, tps, rdm_len) @ (iterations, rdm_len, tps) --> (iterations, tps, tps) - divided by norm to get the values in -1,1 range

    # Average across iterations
    averaged_rsa = rsa_all.mean(axis=0)
    return averaged_rsa


def compute_lag_correlation(rsa_matrix, averaging_diagonal_time_window):
    """
    Computes lag correlation (diagonal average) from RSA matrix,
    restricted to ± averaging_diagonal_time_window (in timepoints).
    """
    n = rsa_matrix.shape[0]
    max_lag = averaging_diagonal_time_window

    lags = np.arange(-max_lag, max_lag + 1)
    lag_corr = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        diag_vals = np.diagonal(rsa_matrix.T, offset=lag) # transpose so that diagonal averaging calculates neural time - model time
        lag_corr[i] = np.nanmean(diag_vals)

    return lags, lag_corr
