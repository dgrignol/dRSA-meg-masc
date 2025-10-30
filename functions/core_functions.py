from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.spatial.distance import pdist
from matplotlib.colors import ListedColormap


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
    candidate_starts=None,
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
    if candidate_starts is not None:
        candidate_starts = np.asarray(candidate_starts, dtype=int).ravel()
        if candidate_starts.size == 0:
            raise ValueError("candidate_starts provided but empty after conversion.")
        # Keep only candidates whose entire window fits in the data bounds.
        max_start = n_timepoints - subsample_tps
        candidate_starts = candidate_starts[
            (candidate_starts >= 0) & (candidate_starts <= max_start)
        ]
        if candidate_starts.size == 0:
            raise ValueError("No candidate starts remain within the valid data range.")
        candidate_starts = np.unique(candidate_starts)
        possible_starts = np.intersect1d(possible_starts, candidate_starts, assume_unique=False)
    if possible_starts.size == 0:
        raise ValueError("No valid windows fit entirely within the True regions of the mask.")

    if possible_starts.size < n_subsamples:
        raise ValueError(
            f"Not enough valid start positions ({possible_starts.size}) to draw "
            f"{n_subsamples} unique starts per iteration. Consider reducing n_subsamples "
            "or relaxing the constraints."
        )

    return possible_starts, rng


def subsampling_iterator(
    n_timepoints,
    subsample_tps,
    n_subsamples,
    iterations,
    mask=None,
    random_state=None,
    candidate_starts=None,
    allow_overlap=False,
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
        candidate_starts=candidate_starts,
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
            # Remove any starts that overlap this one unless overlap is allowed
            if allow_overlap:
                available_starts = available_starts[available_starts != start]
            else:
                available_starts = available_starts[np.abs(available_starts - start) >= subsample_tps]
        if count < n_subsamples:
            raise ValueError(
                f"Not enough {'unique' if allow_overlap else 'non-overlapping'} windows for iteration {it}. "
                f"Needed {n_subsamples}, got {count}."
            )
        yield chosen[:, None] + window


def subsampling(
    n_timepoints,
    subsample_tps,
    n_subsamples,
    iterations,
    mask=None,
    random_state=None,
    candidate_starts=None,
    allow_overlap=False,
):
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
    candidate_starts : 1D array-like, optional
        Specific start indices (at the native sampling rate) that windows are allowed to begin at.
        When provided, every subsample start is drawn from this set after intersecting with the mask.
    allow_overlap : bool, optional
        When True, windows are allowed to overlap one another (but starts remain unique).

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
        candidate_starts=candidate_starts,
        allow_overlap=allow_overlap,
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


def save_subsample_diagnostics(
    subsample_indices,
    subsample_cache_path,
    mask_array=None,
    word_onsets_path=None,
    sampling_rate=100.0,
    zoom_window_seconds=None,
):
    """
    Generate a stacked diagnostic figure summarising subsample coverage.

    Parameters
    ----------
    subsample_indices : array-like, shape (iterations, n_subsamples, subsample_tps)
        Indices returned by `subsampling`.
    subsample_cache_path : str or Path
        Path to the `.npy` cache file; the figure is saved alongside it as `.png`.
    mask_array : array-like, optional
        Boolean mask (1D) at the same sampling rate, used to highlight valid regions.
    word_onsets_path : str or Path, optional
        Path to `*_concatenated_word_onsets_sec.npy`; when present, vertical onset
        markers are drawn on the zoomed panels.
    sampling_rate : float, optional
        Sampling rate in Hz (default 100).
    zoom_window_seconds : tuple(float, float), optional
        Start/end (seconds) for the zoomed panels. When omitted, a window is chosen
        around the earliest subsample start.

    Examples
    --------
    >>> save_subsample_diagnostics(
    ...     subsample_indices=subsample_indices,
    ...     subsample_cache_path="results/cache/subsamples/subsamples_abc.npy",
    ...     mask_array=mask,
    ...     word_onsets_path="derivatives/preprocessed/sub-01/concatenated/sub-01_concatenated_word_onsets_sec.npy",
    ...     sampling_rate=100.0,
    ...     zoom_window_seconds=(30.0, 45.0),
    ... )
    """

    subsample_indices = np.asarray(subsample_indices)
    if subsample_indices.ndim != 3:
        raise ValueError("subsample_indices must be 3D (iterations, n_subsamples, subsample_tps).")

    subsample_cache_path = Path(subsample_cache_path)
    png_path = subsample_cache_path.with_suffix(".png")

    iterations, _, subsample_tps = subsample_indices.shape
    max_index = int(subsample_indices.max()) if subsample_indices.size else 0
    t_max = max_index + subsample_tps

    canvas = np.zeros((iterations, t_max), dtype=np.int32)
    starts = subsample_indices[..., 0]
    for it in range(iterations):
        for start in starts[it]:
            start = int(start)
            stop = start + subsample_tps
            canvas[it, start:stop] += 1

    coverage = canvas.sum(axis=0)
    time_support = coverage.size
    if time_support == 0:
        raise ValueError("No subsample coverage available to plot.")

    base_colors = ["white", "#440154", "#31688E", "#35B779", "#FDE725", "#FDAE61"]
    max_overlap = int(canvas.max())
    if max_overlap >= len(base_colors):
        extra = max_overlap + 1 - len(base_colors)
        base_colors.extend([base_colors[-1]] * extra)
    cmap = ListedColormap(base_colors[: max_overlap + 1])
    bounds = np.arange(-0.5, max_overlap + 1.5, 1)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    mask_bool = None
    if mask_array is not None:
        mask_bool = np.asarray(mask_array, dtype=bool).ravel()
        if mask_bool.size != time_support:
            clipped = mask_bool[:time_support] if mask_bool.size >= time_support else None
            if clipped is None:
                raise ValueError(
                    f"Mask length ({mask_bool.size}) does not match subsample time support ({time_support})."
                )
            mask_bool = clipped

    word_onsets = np.empty(0, dtype=float)
    if word_onsets_path is not None:
        onset_path = Path(word_onsets_path)
        if onset_path.exists():
            word_onsets = np.load(onset_path)

    sampling_rate = float(sampling_rate)
    dt = 1.0 / sampling_rate
    total_duration = (time_support - 1) / sampling_rate if time_support > 1 else 0.0
    global_times = np.arange(time_support) / sampling_rate

    if zoom_window_seconds is None:
        earliest_start = starts.min() if starts.size else 0
        window_start_sec = earliest_start / sampling_rate
        window_width_sec = max(5.0, 5 * (subsample_tps / sampling_rate))
        window_end_sec = min(total_duration, window_start_sec + window_width_sec)
        if window_end_sec <= window_start_sec:
            window_end_sec = min(total_duration, window_start_sec + window_width_sec + 1.0)
        zoom_window_seconds = (window_start_sec, window_end_sec)

    window_start_sec, window_end_sec = zoom_window_seconds
    start_sample = max(0, int(np.floor(window_start_sec * sampling_rate)))
    stop_sample = min(time_support, int(np.ceil(window_end_sec * sampling_rate)))
    if stop_sample <= start_sample:
        stop_sample = min(time_support, start_sample + subsample_tps)
    if stop_sample <= start_sample:
        stop_sample = min(time_support, start_sample + 1)

    zoom_samples = np.arange(start_sample, stop_sample)
    zoom_seconds = zoom_samples / sampling_rate
    sub_canvas = canvas[:, start_sample:stop_sample]
    sub_coverage = coverage[start_sample:stop_sample]

    mask_slice = None
    if mask_bool is not None:
        mask_slice = mask_bool[start_sample:stop_sample]

    if word_onsets.size:
        onsets_in_window = word_onsets[(word_onsets >= zoom_seconds[0]) & (word_onsets <= zoom_seconds[-1])]
    else:
        onsets_in_window = np.empty(0, dtype=float)

    def _shade_mask_regions(ax, times, mask, label=None):
        if mask is None:
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not mask.any():
            return
        sample_count = min(times.size, mask.size)
        times = times[:sample_count]
        mask = mask[:sample_count]
        diffs = np.diff(np.concatenate(([False], mask, [False])))
        starts_idx = np.where(diffs == 1)[0]
        ends_idx = np.where(diffs == -1)[0]
        for idx, (s_idx, e_idx) in enumerate(zip(starts_idx, ends_idx)):
            start_t = times[s_idx]
            end_idx = min(e_idx, sample_count)
            end_t = times[end_idx - 1] + dt
            ax.axvspan(
                start_t,
                end_t,
                color="gold",
                alpha=0.12,
                edgecolor=None,
                linewidth=0,
                zorder=3,
                label=label if label and idx == 0 else None,
            )

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        6,
        2,
        width_ratios=[40, 1],
        height_ratios=[3, 1.6, 0.2, 3, 1.6, 0.2],
        hspace=0.35,
        wspace=0.05,
    )

    # Panel A
    axA = fig.add_subplot(gs[0, 0])
    imA = axA.imshow(
        canvas,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(0, total_duration, iterations, 0),
    )
    axA.set_ylabel("Iteration")
    axA.set_title("A. Subsample overlap across all iterations")
    axA.set_xlim(0, total_duration)
    axA.set_xticklabels([])
    _shade_mask_regions(axA, global_times, mask_bool, label="Sentence mask")
    handlesA, labelsA = axA.get_legend_handles_labels()
    if handlesA:
        axA.legend(handlesA, labelsA, loc="upper right", frameon=False)
    cbarA = fig.colorbar(imA, cax=fig.add_subplot(gs[0, 1]), label="Overlap count", boundaries=bounds)
    cbarA.locator = ticker.FixedLocator(range(0, max_overlap + 1))
    cbarA.formatter = ticker.FixedFormatter(range(0, max_overlap + 1))
    cbarA.update_ticks()

    # Panel B
    coverage_seconds = np.arange(time_support) / sampling_rate
    axB = fig.add_subplot(gs[1, 0], sharex=axA)
    axB.plot(coverage_seconds, coverage, color="slateblue", linewidth=0.8)
    axB.set_ylabel("Iterations")
    axB.set_title("B. Coverage across time (100 Hz, full series)")
    axB.set_xlim(0, total_duration)
    axB.grid(alpha=0.2, linewidth=0.5)
    axB.set_xlabel("Time (s)")
    axB.xaxis.set_major_locator(ticker.MaxNLocator(6))
    axB.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    axB.tick_params(axis="x", labelsize=10)
    fig.add_subplot(gs[1, 1]).axis("off")

    fig.add_subplot(gs[2, :]).axis("off")

    # Panel C
    axC = fig.add_subplot(gs[3, 0])
    imC = axC.imshow(
        sub_canvas,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(zoom_seconds[0], zoom_seconds[-1], iterations, 0),
    )
    axC.set_ylabel("Iteration")
    axC.set_title("C. Zoomed subsample windows")
    axC.set_ylim(iterations, 0)
    _shade_mask_regions(axC, zoom_seconds, mask_slice, label="Sentence mask")
    for idx_onset, onset in enumerate(onsets_in_window):
        axC.axvline(onset, color="crimson", linewidth=0.6, alpha=0.7, label="Word onsets" if idx_onset == 0 else None)
    handles, labels = axC.get_legend_handles_labels()
    if handles:
        axC.legend(handles, labels, loc="upper right", frameon=False)
    axC.set_xlim(zoom_seconds[0], zoom_seconds[-1])
    axC.set_xticklabels([])
    cbarC = fig.colorbar(imC, cax=fig.add_subplot(gs[3, 1]), label="Overlap count", boundaries=bounds)
    cbarC.locator = ticker.FixedLocator(range(0, max_overlap + 1))
    cbarC.formatter = ticker.FixedFormatter(range(0, max_overlap + 1))
    cbarC.update_ticks()

    # Panel D
    axD = fig.add_subplot(gs[4, 0], sharex=axC)
    axD.plot(zoom_seconds, sub_coverage, color="slateblue", linewidth=0.9, label="Coverage")
    _shade_mask_regions(axD, zoom_seconds, mask_slice, label="Sentence mask")
    if onsets_in_window.size:
        axD.vlines(
            onsets_in_window,
            ymin=0,
            ymax=sub_coverage.max(initial=0),
            color="crimson",
            linewidth=0.6,
            alpha=0.6,
            label="Word onsets",
        )
    axD.set_xlabel("Time (s)")
    axD.set_ylabel("Iterations")
    axD.set_title("D. Zoomed coverage with annotations")
    axD.legend(loc="upper right", frameon=False)
    axD.grid(alpha=0.2, linewidth=0.5)
    axD.xaxis.set_major_locator(ticker.MaxNLocator(6))
    axD.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    axD.tick_params(axis="x", labelsize=10)
    fig.add_subplot(gs[4, 1]).axis("off")

    fig.add_subplot(gs[5, :]).axis("off")
    fig.suptitle("Subsampling diagnostics summary", y=0.995, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return png_path
