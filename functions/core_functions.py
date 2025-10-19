import numpy as np
from scipy.spatial.distance import pdist


def zscore(X, axis=0):
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def _prepare_subsampling(
    n_timepoints,
    subsample_tps,
    n_subsamples,
    mask=None,
    random_state=None,
):
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

    return possible_starts, rng


def subsampling_iterator(
    n_timepoints,
    subsample_tps,
    n_subsamples,
    iterations,
    mask=None,
    random_state=None,
):
    """
    Yield subsample window indices for each iteration without materializing the full array.
    """
    possible_starts, rng = _prepare_subsampling(
        n_timepoints,
        subsample_tps,
        n_subsamples,
        mask=mask,
        random_state=random_state,
    )
    window = np.arange(subsample_tps, dtype=int)

    for it in range(iterations):
        available_starts = possible_starts.copy()
        rng.shuffle(available_starts)
        chosen = np.empty(n_subsamples, dtype=int)
        count = 0
        while count < n_subsamples and available_starts.size > 0:
            start = available_starts[0]
            chosen[count] = start
            count += 1
            # Remove any starts that overlap this one
            available_starts = available_starts[np.abs(available_starts - start) >= subsample_tps]
        if count < n_subsamples:
            raise ValueError(
                f"Not enough non-overlapping windows for iteration {it}. "
                f"Needed {n_subsamples}, got {count}."
            )
        yield chosen[:, None] + window


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
    iterator = subsampling_iterator(
        n_timepoints,
        subsample_tps,
        n_subsamples,
        iterations,
        mask=mask,
        random_state=random_state,
    )
    indices = np.empty((iterations, n_subsamples, subsample_tps), dtype=int)
    for it, window_indices in enumerate(iterator):
        indices[it] = window_indices
    return indices


def apply_subsampling_indices(data, subsample_indices):
    """
    Efficiently applies precomputed subsampling indices using NumPy advanced indexing.

    Parameters:
    - data: (features, timepoints) array
    - subsample_indices: (iterations, n_subsamples, subsample_tps) array
      or (n_subsamples, subsample_tps) for a single iteration.

    Returns:
    - subsampled_data: (iterations, n_subsamples, features, subsample_tps) array
      or (n_subsamples, features, subsample_tps) if a single iteration was provided.
    """
    indices = np.asarray(subsample_indices)
    squeeze_iteration = False
    if indices.ndim == 2:
        indices = indices[np.newaxis, ...]
        squeeze_iteration = True
    elif indices.ndim != 3:
        raise ValueError("subsample_indices must be 2D or 3D.")

    # Reshape data for broadcasting: (1, 1, features, timepoints)
    data_expanded = data[np.newaxis, np.newaxis, :, :]
    
    # Reshape indices for advanced indexing: (iterations, n_subsamples, 1, subsample_tps)
    indices_expanded = indices[:, :, np.newaxis, :]
    
    # Use advanced indexing to gather all subsamples at once
    subsampled_data = np.take_along_axis(
        data_expanded,
        indices_expanded,
        axis=-1  # Take along the timepoints dimension
    )
    if squeeze_iteration:
        subsampled_data = subsampled_data[0]
    return subsampled_data  # (iterations, n_subsamples, features, subsample_tps)


def compute_rdm_series_from_indices(data, subsample_indices, metric):
    """
    Compute time-resolved RDMs directly from the original data using subsampling indices.

    Parameters:
    - data: array of shape (features, timepoints)
    - subsample_indices: array of shape (iterations, n_subsamples, subsample_tps) or
      (n_subsamples, subsample_tps) for a single iteration.
    - metric: distance metric string for scipy.spatial.distance.pdist

    Returns:
    - rdm_series: (iterations, subsample_tps, rdm_length) array or (subsample_tps, rdm_length)
      if a single iteration was provided.
    """
    data_array = np.asarray(data)
    if data_array.ndim != 2:
        raise ValueError("data must be a 2D array (features, timepoints).")

    indices = np.asarray(subsample_indices)
    squeeze_iteration = False
    if indices.ndim == 2:
        indices = indices[np.newaxis, ...]
        squeeze_iteration = True
    elif indices.ndim != 3:
        raise ValueError("subsample_indices must be 2D or 3D.")

    iterations, n_subsamples, subsample_tps = indices.shape
    rdm_length = n_subsamples * (n_subsamples - 1) // 2
    rdm_series = np.empty((iterations, subsample_tps, rdm_length), dtype=float)

    for it in range(iterations):
        idx = indices[it]
        for t in range(subsample_tps):
            time_idx = idx[:, t]
            data_slice = data_array[:, time_idx].T  # (n_subsamples, n_features)
            rdm_series[it, t] = pdist(data_slice, metric=metric)

    if squeeze_iteration:
        return rdm_series[0]
    return rdm_series


def compute_rdm_series(subsampled_data, metric):
    """
    Computes the time-resolved RDMs for all iterations.

    Parameters:
    - subsampled_data: shape (iterations, n_subsamples, n_features, subsample_tps)
      or (n_subsamples, n_features, subsample_tps) for a single iteration
    - metric: distance metric string (e.g., 'correlation', 'cosine', 'euclidean')

    Returns:
    - rdm_series: array of shape (iterations, subsample_tps, rdm_length)
      or (subsample_tps, rdm_length) if a single iteration was provided.
    """
    data = np.asarray(subsampled_data)
    squeeze_iteration = False
    if data.ndim == 3:
        data = data[np.newaxis, ...]
        squeeze_iteration = True
    elif data.ndim != 4:
        raise ValueError("subsampled_data must be 3D or 4D.")

    iterations, n_subsamples, _, subsample_tps = data.shape
    rdm_length = n_subsamples * (n_subsamples - 1) // 2
    rdm_series = np.empty((iterations, subsample_tps, rdm_length), dtype=float)

    for it in range(iterations):
        for t in range(subsample_tps):
            # shape: (n_subsamples, n_features)
            data_slice = data[it, :, :, t]
            rdm_series[it, t] = pdist(data_slice, metric=metric)

    if squeeze_iteration:
        return rdm_series[0]
    return rdm_series


def compute_rsa_matrix_corr(
    neural_rdm_series,
    model_rdm_series,
    return_lag_curves=False,
    lag_window=None,
):
    """
    Compute RSA correlation matrices, optionally returning per-iteration lag curves.

    Parameters:
    - neural_rdm_series: (iterations, subsample_tps, rdm_len)
    - model_rdm_series: (iterations, subsample_tps, rdm_len)
    - return_lag_curves: if True, return lag curves per iteration (requires lag_window)
    - lag_window: int, maximum lag (in time points) to consider when computing lag curves

    Returns:
    - averaged_rsa: (subsample_tps, subsample_tps)
    - lag_curves (optional): (iterations, 2 * lag_window + 1)
    """

    neural = np.asarray(neural_rdm_series)
    model = np.asarray(model_rdm_series)
    squeeze_iteration = False
    if neural.ndim == 2:
        neural = neural[np.newaxis, ...]
        model = model[np.newaxis, ...]
        squeeze_iteration = True
    elif neural.ndim != 3 or model.ndim != 3:
        raise ValueError("RDM series must be 2D or 3D arrays with matching shapes.")
    if neural.shape != model.shape:
        raise ValueError("Neural and model RDM series must have the same shape.")

    iterations, subsample_tps, rdm_len = neural.shape

    result_dtype = np.result_type(neural.dtype, model.dtype)
    accumulated = np.zeros((subsample_tps, subsample_tps), dtype=result_dtype)
    lag_curves = [] if return_lag_curves else None

    if return_lag_curves and lag_window is None:
        raise ValueError("lag_window must be provided when return_lag_curves is True.")

    for it in range(iterations):
        neural_z = zscore(neural[it], axis=-1)
        model_z = zscore(model[it], axis=-1)

        rsa_matrix = np.matmul(neural_z, model_z.T) / rdm_len
        accumulated += rsa_matrix

        if lag_curves is not None:
            _, curve = compute_lag_correlation(rsa_matrix, lag_window)
            lag_curves.append(curve)

    averaged_rsa = accumulated / iterations

    if lag_curves is not None:
        lag_curves = np.stack(lag_curves, axis=0)
        if squeeze_iteration:
            lag_curves = lag_curves[0]
        return averaged_rsa, lag_curves
    if squeeze_iteration:
        return averaged_rsa
    return averaged_rsa


def compute_lag_correlation(rsa_matrix, averaging_diagonal_time_window):
    """
    Computes lag correlation (diagonal average) from RSA matrix,
    restricted to ± averaging_diagonal_time_window (in timepoints).
    Accepts either a single RSA matrix (2D) or a stack over iterations (3D).
    """
    rsa_array = np.asarray(rsa_matrix)
    if rsa_array.ndim not in (2, 3):
        raise ValueError("rsa_matrix must be 2D or 3D (iterations, time, time).")

    max_lag = averaging_diagonal_time_window

    lags = np.arange(-max_lag, max_lag + 1)

    if rsa_array.ndim == 2:
        lag_corr = np.zeros_like(lags, dtype=float)
        for i, lag in enumerate(lags):
            diag_vals = np.diagonal(rsa_array.T, offset=lag)  # transpose so that diagonal averaging calculates neural time - model time
            lag_corr[i] = np.nanmean(diag_vals)
        return lags, lag_corr

    iterations = rsa_array.shape[0]
    lag_corr = np.zeros((iterations, lags.size), dtype=float)

    for it in range(iterations):
        matrix = rsa_array[it]
        for i, lag in enumerate(lags):
            diag_vals = np.diagonal(matrix.T, offset=lag)
            lag_corr[it, i] = np.nanmean(diag_vals)

    return lags, lag_corr
