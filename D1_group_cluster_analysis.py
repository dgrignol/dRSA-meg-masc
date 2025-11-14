#!/usr/bin/env python3
"""
Group-level dRSA cluster permutation analysis and summary plotting.

The script aggregates per-subject dRSA outputs produced by ``C1_dRSA_run.py`` and
tests whether the average lag-correlation curve shows significant clusters
following Maris & Oostenveld (2007). It also visualises the average dRSA matrix
for each model alongside the grand-average lag curve (with SEM shading), highlights
significant clusters on the lag axis, and annotates their peak correlation/lag
coordinates. Outputs are stored in ``results/<analysis_name>/group_level`` as both PNG
and PDF, and a cached ``.npz`` bundle makes it possible to regenerate the figures
without re-running the permutation test.

Expected per-subject files (per model) inside ``results/<analysis_name>/single_subjects``:
    sub-{ID}_res100_correlation_dRSA_matrices.npy
    sub-{ID}_res100_correlation_metadata.json

Usage
-----
python D1_group_cluster_analysis.py \
    --subjects 01 02 03 \
    --models Envelope "Word Frequency" "GloVe" "GloVe Norm"

python D1_group_cluster_analysis.py \
    --analysis-name drsa_pilot \
    --subjects $(seq -w 1 10) \
    --models Envelope "Phoneme Voicing" "Word Frequency" "GloVe" "GloVe Norm"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.stats import ttest_rel, t

from functions.generic_helpers import (
    ensure_analysis_directories,
    find_latest_analysis_directory,
    format_log_timestamp,
    read_repository_root,
)


LOGGER = logging.getLogger(__name__)

# Fixed seed for permutation sign choices (exposed in captions/settings for reproducibility)
PERMUTATION_SEED = 0
# Cluster permutation configuration is fixed in this implementation: we run a two-tailed
# test and consider both positive and negative clusters when comparing lag curves/matrices.
CLUSTER_PERMUTATION_TAIL = "two-tailed"
CLUSTER_PERMUTATION_SIGN_SCOPE = "positive_and_negative"
CLUSTER_PERMUTATION_TAIL_DESCRIPTION = "two-tailed (positive and negative clusters)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group-level cluster-permutation analysis on dRSA lag curves."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="List of subject IDs (e.g., 01 02 03).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Ordered list of model labels to analyse (must match metadata labels).",
    )
    parser.add_argument(
        "--analysis-name",
        help=(
            "Named analysis folder under --results-root. When omitted, the most recent analysis "
            "is selected automatically."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Parent directory containing analysis runs (default: results/).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Legacy override for the directory containing per-subject results when older layouts "
            "are still in use."
        ),
    )
    parser.add_argument(
        "--lag-metric",
        type=str,
        default="correlation",
        help="RSA computation method identifier used in filenames (default: correlation).",
    )
    parser.add_argument(
        "--cluster-alpha",
        type=float,
        default=0.01,
        help="Cluster-forming alpha threshold (default: 0.01).",
    )
    parser.add_argument(
        "--permutation-alpha",
        type=float,
        default=0.05,
        help="Cluster-level alpha threshold after permutations (default: 0.05).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=3000,
        help="Number of permutations for the cluster test (default: 5000).",
    )
    parser.add_argument(
        "--force-matrix-clusters",
        action="store_true",
        help=(
            "Run cluster permutation on the full dRSA matrices even when subsamples are not "
            "locked to word onset (default behaviour only tests matrices when locking is enabled)."
        ),
    )
    parser.add_argument(
        "--skip-matrix-clusters",
        action="store_true",
        help=(
            "Skip matrix-level cluster permutation tests even if they would be auto-enabled "
            "(useful for fast smoke runs)."
        ),
    )
    parser.add_argument(
        "--matrix-downsample-factor",
        type=int,
        default=1,
        help=(
            "Downsample factor applied to dRSA matrices before the permutation test "
            "(must be a positive integer; 1 disables downsampling)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the summary figure (PNG will be written here). "
            "Defaults to <analysis>/group_level/group_dRSA_summary.png when using named analyses."
        ),
    )
    parser.add_argument(
        "--summary-cache",
        type=Path,
        default=None,
        help=(
            "Path for storing aggregated group results used to recreate plots without rerunning "
            "the cluster analysis (default: matches --output but with a .npz suffix)."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Skip the cluster analysis and regenerate the plots using a previously saved summary "
            "cache (requires --summary-cache or the default cache file to exist)."
        ),
    )
    parser.add_argument(
        "--simulation-noise",
        action="store_true",
        help=(
            "Use simulation outputs stored under <analysis>/simulations instead of single-subject "
            "results. When enabled, the script filters for synthetic noise runs."
        ),
    )
    parser.add_argument(
        "--simulation-neural-label",
        default="Random Noise 208",
        help=(
            "Simulation neural source label to load when --simulation-noise is set "
            "(e.g., 'Random Noise 208' or 'Random Noise 1')."
        ),
    )
    parser.add_argument(
        "--simulation-origin",
        default=None,
        help=(
            "Override the required simulation origin recorded in metadata "
            "(default: 'synthetic' when --simulation-noise is provided)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level for console logging.",
    )
    return parser.parse_args()


def _label_to_filename_fragment(label: str) -> str:
    fragment = str(label).replace(" ", "_")
    fragment = fragment.replace(os.sep, "_").replace("/", "_").strip("_")
    return fragment or "neural"


def load_subject_data(
    subject: str,
    models: Sequence[str],
    results_dir: Path,
    lag_metric: str,
    *,
    use_simulation_runs: bool = False,
    simulation_label: Optional[str] = None,
    simulation_origin_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load per-subject dRSA matrices and derive the model-specific lag curves.

    Parameters
    ----------
    subject:
        Subject identifier as a two-digit string (e.g., "01").
    models:
        Ordered list of model labels to extract from the subject's data.
    results_dir:
        Folder containing the npy/json outputs produced by ``C1_dRSA_run.py``.
    lag_metric:
        RSA filename prefix indicating the similarity metric (e.g., "correlation").

    Parameters
    ----------
    subject : str
        Two-digit subject identifier (e.g., "01").
    models : Sequence[str]
        Ordered model labels to extract.
    results_dir : Path
        Directory containing subject outputs (single_subjects or simulations).
    lag_metric : str
        Metric tag embedded in filenames (e.g., "correlation").
    use_simulation_runs : bool, optional
        When True, search for simulation outputs (files with `_sim_*` suffix).
    simulation_label : str, optional
        Neural source label recorded in the simulation metadata to select.
    simulation_origin_filter : str, optional
        Require the simulation metadata's `neural_source_origin` to match this string.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The subset of dRSA matrices, their lag curves, metadata, and the lag-axis in TPs.
    """
    subject_id = int(subject)
    prefix_candidates = [
        f"sub-{subject_id:02d}_res100_{lag_metric}",
        f"sub{subject_id}_res100_{lag_metric}",
    ]

    matrices_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    metadata: Optional[Dict[str, Any]] = None

    if use_simulation_runs:
        candidate_pairs: List[Tuple[Path, Dict[str, Any]]] = []
        for prefix in prefix_candidates:
            pattern = f"{prefix}_sim_*_metadata.json"
            for candidate_meta in sorted(results_dir.glob(pattern)):
                try:
                    candidate_metadata = json.loads(candidate_meta.read_text())
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse metadata {candidate_meta}") from exc
                sim_info = candidate_metadata.get("simulation") or {}
                if not sim_info.get("enabled", False):
                    continue
                if simulation_origin_filter and sim_info.get("neural_source_origin") != simulation_origin_filter:
                    continue
                if simulation_label and sim_info.get("neural_source_label") != simulation_label:
                    continue
                candidate_pairs.append((candidate_meta, candidate_metadata))
        if not candidate_pairs:
            label_msg = f" with label '{simulation_label}'" if simulation_label else ""
            origin_msg = (
                f" and origin '{simulation_origin_filter}'" if simulation_origin_filter else ""
            )
            raise FileNotFoundError(
                f"No simulation outputs{label_msg}{origin_msg} found for subject {subject} in {results_dir}."
            )
        if simulation_label is None and len(candidate_pairs) > 1:
            labels_available = ", ".join(
                (pair[1].get("simulation", {}) or {}).get("neural_source_label", pair[0].stem)
                for pair in candidate_pairs
            )
            raise ValueError(
                "Multiple simulation runs found; specify --simulation-neural-label. "
                f"Available labels: {labels_available}"
            )
        meta_path, metadata = candidate_pairs[0]
        base_stem = meta_path.stem.replace("_metadata", "")
        matrices_path = meta_path.with_name(f"{base_stem}_dRSA_matrices.npy")
        if not matrices_path.exists():
            raise FileNotFoundError(
                f"Simulation matrices missing for {meta_path.name}: expected {matrices_path.name}"
            )
    else:
        for prefix in prefix_candidates:
            candidate_matrix = results_dir / f"{prefix}_dRSA_matrices.npy"
            candidate_meta = results_dir / f"{prefix}_metadata.json"
            if candidate_matrix.exists() and candidate_meta.exists():
                matrices_path = candidate_matrix
                meta_path = candidate_meta
                break

    if matrices_path is None or meta_path is None:
        expected = ", ".join(prefix_candidates)
        raise FileNotFoundError(
            f"Missing dRSA outputs for subject {subject} in {results_dir} "
            f"(expected prefixes: {expected})"
        )

    matrices = np.load(matrices_path)
    if metadata is None:
        metadata = json.loads(meta_path.read_text())

    labels = metadata.get("selected_model_labels")
    if labels is None:
        raise KeyError(f"selected_model_labels missing in {meta_path}")

    matrices = np.asarray(matrices)
    if matrices.ndim < 3:
        raise ValueError(
            f"dRSA matrices for subject {subject} must be at least 3D; received shape {matrices.shape}."
        )

    n_models = len(labels)
    neural_labels = metadata.get("neural_signal_labels") or []

    if matrices.shape[1] == n_models:
        n_neural = matrices.shape[0]
        if not neural_labels:
            neural_labels = [f"Neural {idx + 1}" for idx in range(n_neural)]
        elif len(neural_labels) != n_neural:
            if len(neural_labels) == 1 and n_neural > 1:
                base_label = neural_labels[0]
                neural_labels = [f"{base_label} {idx + 1}" for idx in range(n_neural)]
            else:
                raise ValueError(
                    f"neural_signal_labels in {meta_path} has length {len(neural_labels)}, "
                    f"but matrices provide {n_neural} neural signal sets."
                )
    elif matrices.shape[0] == n_models:
        matrices = matrices[None, ...]
        n_neural = 1
        if not neural_labels:
            neural_labels = ["Neural 1"]
        elif len(neural_labels) != 1:
            raise ValueError(
                f"Ambiguous neural_signal_labels for subject {subject}: expected one label, found {neural_labels}."
            )
    else:
        raise ValueError(
            f"Unexpected dRSA matrix shape {matrices.shape} for subject {subject}: "
            f"cannot reconcile with {n_models} model labels."
        )

    metadata["neural_signal_labels"] = neural_labels

    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    model_indices = []
    for model in models:
        if model not in label_to_idx:
            raise KeyError(f"Model '{model}' not found for subject {subject}; available: {labels}")
        model_indices.append(label_to_idx[model])

    matrices_subset = matrices[:, model_indices, ...]

    # Recover analysis settings required to translate lag steps into seconds.
    lag_settings = metadata["analysis_parameters"]
    adtw_in_tps = lag_settings["averaging_window_tps"]

    lag_curves_subject: List[List[np.ndarray]] = []
    lags_tp_axis: Optional[np.ndarray] = None
    for neural_idx in range(matrices_subset.shape[0]):
        neural_curves = []
        for matrix in matrices_subset[neural_idx]:
            lag_curve, lags_tp = compute_lag_curve_from_matrix(matrix, adtw_in_tps)
            if lags_tp_axis is None:
                lags_tp_axis = lags_tp
            elif lags_tp_axis.shape != lags_tp.shape or not np.array_equal(lags_tp_axis, lags_tp):
                raise ValueError(
                    f"Inconsistent lag axis encountered for subject {subject} across neural/model combinations."
                )
            neural_curves.append(lag_curve)
        lag_curves_subject.append(neural_curves)
    lag_curves_subject_arr = np.array(lag_curves_subject)

    if lags_tp_axis is None:
        raise RuntimeError(f"Failed to compute lag axis for subject {subject}.")

    return matrices_subset, lag_curves_subject_arr, metadata, lags_tp_axis


def compute_lag_curve_from_matrix(matrix: np.ndarray, adtw_in_tps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse a dRSA matrix into a lag curve by averaging diagonals across a centred window."""
    from functions.core_functions import compute_lag_correlation

    lags_tp, lag_corr = compute_lag_correlation(matrix, adtw_in_tps)
    return lag_corr, lags_tp


def find_clusters(stat_map: np.ndarray, threshold: float) -> List[np.ndarray]:
    """
    Detect contiguous supra-threshold clusters in a one-dimensional statistic map.

    Returns
    -------
    List[np.ndarray]
        Each entry contains the indices belonging to a positive or negative cluster.
    """
    clusters: List[np.ndarray] = []
    above = stat_map > threshold
    below = stat_map < -threshold

    for mask in (above, below):
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        split_points = np.where(np.diff(idx) > 1)[0] + 1
        clusters_indices = np.split(idx, split_points)
        clusters.extend(clusters_indices)
    return clusters


def permutation_cluster_test(
    data: np.ndarray,
    cluster_alpha: float,
    permutation_alpha: float,
    n_permutations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify time-points with significant group-level effects via sign-permutation.

    The function follows the dependent-samples cluster-permutation procedure
    described by Maris & Oostenveld (2007). It returns a boolean mask indicating
    significant time-points and the t-statistic map for diagnostics.
    """
    n_subjects, n_times = data.shape
    if n_subjects < 2:
        LOGGER.warning(
            "Cluster permutation requires at least two subjects; returning no significant clusters."
        )
        return np.zeros(n_times, dtype=bool), np.zeros(n_times, dtype=np.float32)
    t_vals, _ = ttest_rel(data, np.zeros_like(data), axis=0)
    df = n_subjects - 1

    threshold = abs(t.ppf(1 - cluster_alpha / 2, df))

    clusters = find_clusters(t_vals, threshold)
    cluster_sums = np.array([np.sum(t_vals[cluster]) for cluster in clusters], dtype=float)

    max_sums = np.zeros(n_permutations, dtype=float)
    rng = np.random.default_rng(PERMUTATION_SEED)  # Fixed seed for reproducible thresholds across runs.

    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_subjects)[:, None]
        perm_data = data * signs
        t_perm, _ = ttest_rel(perm_data, np.zeros_like(perm_data), axis=0)
        perm_clusters = find_clusters(t_perm, threshold)
        if perm_clusters:
            max_sums[i] = np.max([abs(np.sum(t_perm[c])) for c in perm_clusters])
        else:
            max_sums[i] = 0.0
        if LOGGER.isEnabledFor(logging.DEBUG):
            iteration = i + 1
            if iteration == 10 or (iteration % 100 == 0):
                LOGGER.debug(
                    "Lag cluster permutation progress: %d/%d iterations completed.",
                    iteration,
                    n_permutations,
                )

    cluster_threshold = np.quantile(max_sums, 1 - permutation_alpha)
    significant = np.zeros(n_times, dtype=bool)

    for cluster, cluster_sum in zip(clusters, cluster_sums):
        if abs(cluster_sum) > cluster_threshold:
            significant[cluster] = True

    return significant, t_vals


def permutation_cluster_test_matrix(
    data: np.ndarray,
    cluster_alpha: float,
    permutation_alpha: float,
    n_permutations: int,
    connectivity: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster permutation test for 2D dRSA matrices using sign-flipping across subjects.

    Parameters
    ----------
    data:
        Array of shape (n_subjects, dim_time, dim_time) containing per-subject dRSA matrices.
    cluster_alpha:
        Point-wise alpha used to form clusters.
    permutation_alpha:
        Alpha threshold applied to the permutation distribution of cluster-level statistics.
    n_permutations:
        Number of permutations used to build the null distribution.
    connectivity:
        Connectivity passed to ``ndimage.generate_binary_structure`` (default: 1 for 4-connectivity).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Boolean mask of significant samples with shape (dim_time, dim_time) and the t-statistic map.
    """
    if data.ndim != 3:
        raise ValueError(
            f"Matrix cluster test expects data with shape (subjects, time, time); received {data.shape}."
        )
    n_subjects = data.shape[0]
    if n_subjects < 2:
        LOGGER.warning(
            "Matrix cluster permutation requires at least two subjects; returning no significant clusters."
        )
        return np.zeros_like(data[0], dtype=bool), np.zeros_like(data[0], dtype=np.float32)

    t_vals, _ = ttest_rel(data, np.zeros_like(data), axis=0)
    df = n_subjects - 1
    threshold = abs(t.ppf(1 - cluster_alpha / 2, df))

    supra_threshold = np.abs(t_vals) > threshold
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled, n_clusters = ndimage.label(supra_threshold, structure=structure)

    cluster_ids = range(1, n_clusters + 1)
    cluster_sums = np.array(
        [np.sum(t_vals[labeled == cluster_id]) for cluster_id in cluster_ids], dtype=float
    )

    max_sums = np.zeros(n_permutations, dtype=float)
    rng = np.random.default_rng(PERMUTATION_SEED)
    zeros_like = np.zeros_like(data)

    for perm_idx in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_subjects)[:, None, None]
        permuted = data * signs
        t_perm, _ = ttest_rel(permuted, zeros_like, axis=0)
        supra_perm = np.abs(t_perm) > threshold
        if np.any(supra_perm):
            labeled_perm, n_perm_clusters = ndimage.label(supra_perm, structure=structure)
            perm_cluster_sums = [
                abs(np.sum(t_perm[labeled_perm == cluster_id]))
                for cluster_id in range(1, n_perm_clusters + 1)
            ]
            max_sums[perm_idx] = max(perm_cluster_sums)
        else:
            max_sums[perm_idx] = 0.0
        if LOGGER.isEnabledFor(logging.DEBUG):
            iteration = perm_idx + 1
            if iteration == 10 or (iteration % 100 == 0):
                LOGGER.debug(
                    "Matrix cluster permutation progress: %d/%d iterations completed.",
                    iteration,
                    n_permutations,
                )

    cluster_threshold = np.quantile(max_sums, 1 - permutation_alpha)
    significant = np.zeros_like(t_vals, dtype=bool)

    for cluster_id, cluster_sum in zip(cluster_ids, cluster_sums):
        if abs(cluster_sum) > cluster_threshold:
            significant[labeled == cluster_id] = True

    return significant, t_vals


def compute_sem(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the standard error of the mean along the requested axis."""
    n = data.shape[axis]
    if n < 2:
        return np.zeros_like(data.take(indices=0, axis=axis), dtype=np.float32)
    return data.std(axis=axis, ddof=1) / np.sqrt(n)


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    """Convert diverse representations of booleans (including strings/ints) to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _extract_lock_to_word_onset(metadata: Dict[str, Any]) -> Optional[bool]:
    """Extract the lock_subsample_to_word_onset flag from subject metadata if available."""
    direct = _coerce_optional_bool(metadata.get("lock_subsample_to_word_onset"))
    if direct is not None:
        return direct
    analysis_parameters = metadata.get("analysis_parameters")
    if isinstance(analysis_parameters, dict):
        nested = _coerce_optional_bool(analysis_parameters.get("lock_subsample_to_word_onset"))
        if nested is not None:
            return nested
    return None


def _downsample_matrix_stack(stack: np.ndarray, factor: int) -> np.ndarray:
    """Downsample square matrices by averaging non-overlapping blocks."""
    if factor <= 1:
        return stack
    if stack.ndim != 3:
        raise ValueError(f"Expected (subjects, time, time) array; received shape {stack.shape}.")
    n_subjects, height, width = stack.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            f"Matrix dimensions {height}x{width} are not divisible by downsample factor {factor}."
        )
    new_h = height // factor
    new_w = width // factor
    reshaped = stack.reshape(n_subjects, new_h, factor, new_w, factor)
    return reshaped.mean(axis=(2, 4))


def _upsample_matrix_mask(mask: np.ndarray, factor: int, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upsample a boolean mask produced on a coarse grid back to the original matrix size."""
    if factor <= 1:
        return mask.astype(bool, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask; received shape {mask.shape}.")
    upsampled = np.kron(mask.astype(bool), np.ones((factor, factor), dtype=bool))
    return upsampled[: target_shape[0], : target_shape[1]]


def _format_analysis_caption(
    analysis_parameters: Optional[Dict[str, Any]],
    extra_entries: Sequence[Tuple[str, Any]],
) -> Optional[str]:
    caption_items: List[Tuple[str, Any]] = []
    if analysis_parameters:
        caption_items.extend(list(analysis_parameters.items()))
    observed_keys = {key for key, _ in caption_items}
    for key, value in extra_entries:
        if key in observed_keys:
            continue
        caption_items.append((key, value))
    if not any(key == "timestamp" for key, _ in caption_items):
        caption_items.append(("timestamp", format_log_timestamp()))
    if not caption_items:
        return None
    caption = ", ".join(f"{key}={value}" for key, value in caption_items)
    caption = "Parameters: " + caption
    return textwrap.fill(caption, width=110)


def create_summary_plot(
    avg_matrices: np.ndarray,
    avg_lag_curves: np.ndarray,
    sem_lag_curves: np.ndarray,
    lags_sec: np.ndarray,
    lag_significance_masks: np.ndarray,
    model_labels: Sequence[str],
    output_path: Path,
    vector_output_path: Optional[Path] = None,
    neural_label: Optional[str] = None,
    analysis_caption: Optional[str] = None,
    matrix_significance_masks: Optional[np.ndarray] = None,
    locked_to_word_onset: bool = False,
    matrix_extent_sec: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Build a figure summarising average dRSA matrices, lag curves, and significant clusters.

    Parameters
    ----------
    avg_matrices:
        Group-average dRSA matrices with shape (n_models, n_tp, n_tp).
    avg_lag_curves:
        Group-average lag curves with shape (n_models, n_times).
    sem_lag_curves:
        Standard error of the mean for each lag curve with the same shape as ``avg_lag_curves``.
    lags_sec:
        Array of lag positions in seconds aligned with the lag curves.
    lag_significance_masks:
        Boolean array flagging significant time-points returned by ``permutation_cluster_test``.
    model_labels:
        Human-readable labels for the models (used in subplot titles and legend).
    output_path:
        Destination path for the rasterised PNG figure (directories created if necessary).
    vector_output_path:
        Optional path for a vector export (e.g., PDF) of the same figure.
    analysis_caption:
        Optional string describing analysis parameters to display beneath the subplots.
    matrix_significance_masks:
        Optional boolean arrays highlighting significant samples within the dRSA matrices.
    locked_to_word_onset:
        Flag indicating whether subsamples were locked to word onset (annotated in the title).
    """
    n_models = len(model_labels)
    fig = plt.figure(figsize=(12, 3 * n_models + 0.75), constrained_layout=True)
    height_ratios = [1.0] * n_models + [0.25]
    grid = fig.add_gridspec(n_models + 1, 2, height_ratios=height_ratios)

# increase the padding between subplots for clarity
#    layout_engine = fig.get_layout_engine()
#    if layout_engine and hasattr(layout_engine, "set"):
#        layout_engine.set(
#            w_pad=4 / 72,
#            h_pad=18 / 72,
#            wspace=0.03,
#            hspace=0.08,
#        )

    axes = np.empty((n_models, 2), dtype=object)
    for idx in range(n_models):
        axes[idx, 0] = fig.add_subplot(grid[idx, 0])
        axes[idx, 1] = fig.add_subplot(grid[idx, 1])

    caption_ax = fig.add_subplot(grid[-1, :])
    caption_ax.axis("off")

    lag_span = float(lags_sec[-1] - lags_sec[0]) if lags_sec.size > 1 else 1.0
    cluster_marker_size = 10
    cluster_peak_marker_size = 2

    for idx, label in enumerate(model_labels):
        ax_matrix = axes[idx, 0]
        im = ax_matrix.imshow(
            avg_matrices[idx],
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=matrix_extent_sec if matrix_extent_sec is not None else None,
        )
        ax_matrix.set_title(f"{label} | Average dRSA")
        ax_matrix.set_xlabel("Model time (s)")
        ax_matrix.set_ylabel("Neural time (s)")
        fig.colorbar(im, ax=ax_matrix, fraction=0.04, pad=0.00005)

        # If using seconds extent, fix axis limits and prep coordinate grids for contours
        x_coords = y_coords = None
        if matrix_extent_sec is not None:
            xmin, xmax, ymin, ymax = matrix_extent_sec
            ax_matrix.set_xlim(xmin, xmax)
            ax_matrix.set_ylim(ymin, ymax)
            ny, nx = avg_matrices[idx].shape
            x_coords = np.linspace(xmin, xmax, nx)
            y_coords = np.linspace(ymin, ymax, ny)

        ax_lag = axes[idx, 1]
        curve = avg_lag_curves[idx]
        sem = sem_lag_curves[idx]
        significant = lag_significance_masks[idx]

        line_color = "#1f77b4"
        fill_color = "#9ecae1"
        ax_lag.plot(lags_sec, curve, color=line_color, label="Mean lag corr")
        ax_lag.fill_between(
            lags_sec,
            curve - sem,
            curve + sem,
            color=fill_color,
            alpha=0.4,
            label="SEM",
        )

        if np.any(significant):
            sig_indices = np.where(significant)[0]
            split_points = np.where(np.diff(sig_indices) > 1)[0] + 1
            clusters = np.split(sig_indices, split_points)
            added_to_legend = False
            for cluster in clusters:
                # Align cluster markers at y=0 to make the time-extent of the cluster explicit.
                legend_label = "Significant cluster" if idx == 0 and not added_to_legend else None
                ax_lag.scatter(
                    lags_sec[cluster],
                    np.zeros_like(cluster, dtype=float),
                    color="red",
                    s=cluster_marker_size,
                    zorder=5,
                    label=legend_label,
                )
                added_to_legend = True

                # Identify and annotate the peak correlation within the cluster.
                cluster_curve = curve[cluster]
                peak_offset = int(np.argmax(np.abs(cluster_curve)))
                peak_index = cluster[peak_offset]
                peak_lag = lags_sec[peak_index]
                peak_val = curve[peak_index]

                peak_marker = ax_lag.scatter(
                    [peak_lag],
                    [peak_val],
                    color="red",
                    edgecolors="white",
                    linewidths=0.5,
                    s=cluster_peak_marker_size,
                    zorder=6,
                )
                peak_marker.set_gid(f"{label}_cluster_peak_{peak_index}")

                curve_span = float(np.ptp(curve)) if np.ptp(curve) > 0 else 1.0
                direction_x = 1 if peak_lag >= 0 else -1
                if np.isclose(peak_lag, 0.0):
                    direction_x = 1
                direction_y = 1 if peak_val >= 0 else -1
                if np.isclose(peak_val, 0.0):
                    direction_y = 1
                text_dx = 0.03 * lag_span * direction_x
                text_dy = 0.05 * curve_span * direction_y

                annotation = ax_lag.annotate(
                    f"Peak: {peak_val:.4f}\nLag: {peak_lag:.2f}s",
                    xy=(peak_lag, peak_val),
                    xytext=(peak_lag + text_dx, peak_val + text_dy),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
                    fontsize=8,
                    color="red",
                    ha="left" if direction_x >= 0 else "right",
                    va="bottom" if direction_y >= 0 else "top",
                )
                annotation.set_gid(f"{label}_cluster_peak_annotation_{peak_index}")

        ax_lag.axvline(0, color="k", linestyle="--", linewidth=0.75)
        ax_lag.axhline(0, color="k", linestyle="--", linewidth=0.75)
        ax_lag.set_title(f"{label} | Lag correlation")
        ax_lag.set_xlabel("Lag (s)")
        ax_lag.set_ylabel("Correlation")
        if idx == 0:
            ax_lag.legend(loc="upper left")

        matrix_mask = None
        if matrix_significance_masks is not None and idx < matrix_significance_masks.shape[0]:
            matrix_mask = matrix_significance_masks[idx]
        if matrix_mask is not None and np.any(matrix_mask):
            if x_coords is not None and y_coords is not None:
                ax_matrix.contour(
                    x_coords,
                    y_coords,
                    matrix_mask.astype(float),
                    levels=[0.5],
                    colors="red",
                    linewidths=0.8,
                    linestyles="-",
                )
            else:
                ax_matrix.contour(
                    matrix_mask.astype(float),
                    levels=[0.5],
                    colors="red",
                    linewidths=0.8,
                    linestyles="-",
                )

    title_parts = ["Group-level dRSA summary"]
    if neural_label:
        title_parts.append(neural_label)
    if locked_to_word_onset:
        title_parts.append("Locked to word onset")
    title = " | ".join(title_parts)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if analysis_caption:
        caption_ax.text(0.5, 0.4, analysis_caption, ha="center", va="center", fontsize=8)
    fig.savefig(output_path, dpi=300)
    if vector_output_path is not None:
        fig.savefig(vector_output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Suppress verbose Matplotlib DEBUG logs (e.g., findfont scoring) while keeping our own DEBUG output
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Resolve repository-relative paths so the script can run from any working directory.
    repo_root = read_repository_root()

    analysis_root: Optional[Path] = None
    group_level_dir: Optional[Path] = None
    results_dir_mode = "analysis"
    analysis_name = args.analysis_name

    simulation_label = args.simulation_neural_label if args.simulation_noise else None
    if args.simulation_noise:
        simulation_origin_filter = args.simulation_origin or "synthetic"
    else:
        simulation_origin_filter = None

    if args.results_dir is not None:
        results_dir = args.results_dir
        if not results_dir.is_absolute():
            results_dir = (repo_root / results_dir).resolve()
        else:
            results_dir = results_dir.resolve()
        results_dir_mode = "legacy"
        LOGGER.info("Using legacy results directory for subject inputs: %s", results_dir)
    else:
        results_root = args.results_root
        if not results_root.is_absolute():
            results_root = (repo_root / results_root).resolve()
        else:
            results_root = results_root.resolve()
        if analysis_name:
            (
                analysis_name,
                analysis_root,
                single_subjects_dir,
                group_level_dir,
            ) = ensure_analysis_directories(results_root, analysis_name)
        else:
            latest_root = find_latest_analysis_directory(results_root)
            if latest_root is None:
                raise FileNotFoundError(
                    f"No analysis directories found under {results_root}. "
                    "Run C1_dRSA_run.py first or provide --analysis-name/--results-dir."
                )
            (
                analysis_name,
                analysis_root,
                single_subjects_dir,
                group_level_dir,
            ) = ensure_analysis_directories(results_root, latest_root.name)
        results_dir = single_subjects_dir
        LOGGER.info("Using analysis '%s' located at %s.", analysis_name, analysis_root)
        if args.simulation_noise:
            simulation_dir = analysis_root / "simulations"
            results_dir = simulation_dir
            LOGGER.info(
                "Simulation-noise mode enabled — sourcing inputs from: %s", results_dir
            )
        else:
            LOGGER.info("Subject-level inputs: %s", results_dir)
        LOGGER.info("Group-level output directory: %s", group_level_dir)

    if args.simulation_noise and results_dir_mode == "legacy":
        LOGGER.warning(
            "--simulation-noise requested with --results-dir; ensure the directory points to simulation outputs."
        )

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    output_path = args.output
    if output_path is None:
        if results_dir_mode == "analysis" and group_level_dir is not None:
            output_path = group_level_dir / "group_dRSA_summary.png"
        else:
            output_path = Path("results") / "group_level" / "group_dRSA_summary.png"
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    # Always emit a vector copy alongside the raster PNG for downstream figure editing.
    vector_output_path = output_path.with_suffix(".pdf")
    summary_cache = args.summary_cache
    if summary_cache is None:
        summary_cache = output_path.with_suffix(".npz")
    if not summary_cache.is_absolute():
        summary_cache = (repo_root / summary_cache).resolve()
    # Ensure parent folders exist before we attempt to save figures or cache files.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_cache.parent.mkdir(parents=True, exist_ok=True)
    vector_output_path.parent.mkdir(parents=True, exist_ok=True)

    def _with_suffix(path: Path, suffix: str) -> Path:
        if not suffix:
            return path
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")

    if args.plot_only:
        # Recreate the plots directly from the cached group-level results.
        cache_candidates: List[Path] = []
        if summary_cache.exists():
            cache_candidates.append(summary_cache)
        else:
            cache_pattern = f"{summary_cache.stem}_*.npz"
            cache_candidates.extend(
                sorted(summary_cache.parent.glob(cache_pattern))
            )
        if not cache_candidates:
            raise FileNotFoundError(
                f"Summary cache not found at {summary_cache}. "
                "Run without --plot-only first."
            )

        def _coerce_np_payload(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                try:
                    value = value.item()
                except ValueError:
                    value = value.tolist()
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                return value[0]
            return value

        def _resolve_output_path(path_value: Optional[str], fallback: Path) -> Path:
            if not path_value:
                return fallback
            candidate = Path(path_value)
            if not candidate.is_absolute():
                candidate = (repo_root / candidate).resolve()
            return candidate

        for cache_path in cache_candidates:
            with np.load(cache_path, allow_pickle=True) as cache_payload:
                avg_matrices = cache_payload["avg_matrices"]
                avg_lag_curves = cache_payload["avg_lag_curves"]
                sem_lag_curves = cache_payload["sem_lag_curves"]
                lags_sec = cache_payload["lags_sec"]
                lag_significance_masks = cache_payload["significance_masks"]
                model_labels = cache_payload["model_labels"].tolist()
                neural_label = cache_payload.get("neural_label", None)
                if isinstance(neural_label, np.ndarray):
                    try:
                        neural_label = neural_label.item()
                    except ValueError:
                        neural_label = neural_label.tolist()
                analysis_caption = cache_payload.get("analysis_caption", None)
                if isinstance(analysis_caption, np.ndarray):
                    try:
                        analysis_caption = analysis_caption.item()
                    except ValueError:
                        analysis_caption = analysis_caption.tolist()
                analysis_settings = _coerce_np_payload(cache_payload.get("analysis_settings"))
                matrix_significance_masks = cache_payload.get("matrix_significance_masks", None)
                if isinstance(matrix_significance_masks, np.ndarray):
                    matrix_significance_masks = matrix_significance_masks.astype(bool)
                else:
                    matrix_significance_masks = np.zeros_like(avg_matrices, dtype=bool)
            output_path_local = output_path
            vector_output_local = vector_output_path
            locked_to_word_onset_cached = False
            if isinstance(analysis_settings, dict):
                output_path_local = _resolve_output_path(
                    analysis_settings.get("output_path"), output_path_local
                )
                vector_output_local = _resolve_output_path(
                    analysis_settings.get("vector_output_path"), vector_output_local
                )
                locked_to_word_onset_cached = bool(
                    analysis_settings.get("locked_to_word_onset", False)
                )
            else:
                suffix = ""
                if cache_path.stem.startswith(summary_cache.stem):
                    suffix = cache_path.stem[len(summary_cache.stem) :]
                if suffix:
                    output_path_local = output_path_local.with_name(
                        f"{output_path_local.stem}{suffix}{output_path_local.suffix}"
                    )
                    vector_output_local = vector_output_local.with_name(
                        f"{vector_output_local.stem}{suffix}{vector_output_local.suffix}"
                    )

            # Attempt to recover matrix extent (seconds) from cached settings
            matrix_extent_sec = None
            if isinstance(analysis_settings, dict):
                ap = analysis_settings.get("analysis_parameters")
                if isinstance(ap, dict):
                    try:
                        res_hz = float(ap.get("resolution_hz"))
                        subs_tps = int(ap.get("subsample_tps"))
                        dur_sec = subs_tps / res_hz if res_hz > 0 else None
                        if dur_sec is not None:
                            matrix_extent_sec = (0.0, float(dur_sec), 0.0, float(dur_sec))
                    except Exception:
                        matrix_extent_sec = None

            create_summary_plot(
                avg_matrices=avg_matrices,
                avg_lag_curves=avg_lag_curves,
                sem_lag_curves=sem_lag_curves,
                lags_sec=lags_sec,
                lag_significance_masks=lag_significance_masks,
                model_labels=model_labels,
                output_path=output_path_local,
                vector_output_path=vector_output_local,
                neural_label=neural_label,
                analysis_caption=analysis_caption,
                matrix_significance_masks=matrix_significance_masks,
                locked_to_word_onset=locked_to_word_onset_cached,
                matrix_extent_sec=matrix_extent_sec,
            )
            LOGGER.info(
                "Regenerated group summary figure (PNG: %s, PDF: %s) using cached results (%s).",
                output_path_local,
                vector_output_local,
                cache_path,
            )
        return 0

    if not args.subjects:
        raise ValueError("No subjects provided. Specify --subjects unless using --plot-only.")
    if not args.models:
        raise ValueError("No models provided. Specify --models unless using --plot-only.")

    matrix_downsample_factor = args.matrix_downsample_factor
    if matrix_downsample_factor < 1:
        raise ValueError("--matrix-downsample-factor must be a positive integer.")
    if matrix_downsample_factor > 1:
        LOGGER.info(
            "Downsampling dRSA matrices by a factor of %d before matrix-level permutation tests.",
            matrix_downsample_factor,
        )

    subjects = [s.lstrip("sub-") for s in args.subjects]
    subject_data = []
    lock_flags: List[Optional[bool]] = []
    lag_axis_reference: Optional[np.ndarray] = None

    for subject in subjects:
        matrices, lag_curves, metadata, lags_tp_axis = load_subject_data(
            subject,
            args.models,
            results_dir,
            args.lag_metric,
            use_simulation_runs=args.simulation_noise,
            simulation_label=simulation_label if args.simulation_noise else None,
            simulation_origin_filter=simulation_origin_filter if args.simulation_noise else None,
        )
        subject_data.append((matrices, lag_curves, metadata))
        if lag_axis_reference is None:
            lag_axis_reference = lags_tp_axis
        elif lag_axis_reference.shape != lags_tp_axis.shape or not np.array_equal(lag_axis_reference, lags_tp_axis):
            raise ValueError(
                "Lag axis mismatch across subjects; ensure all subjects were analysed with the same settings."
            )
        lock_flags.append(_extract_lock_to_word_onset(metadata))

    if lag_axis_reference is None:
        raise RuntimeError("Unable to determine lag axis from subject data.")

    resolved_lock_flags = {flag for flag in lock_flags if flag is not None}
    if len(resolved_lock_flags) > 1:
        raise ValueError(
            "Inconsistent lock_subsample_to_word_onset values across subjects; please verify inputs."
        )
    locked_to_word_onset = resolved_lock_flags.pop() if resolved_lock_flags else False
    if not resolved_lock_flags and any(flag is None for flag in lock_flags):
        LOGGER.warning(
            "Could not determine lock_subsample_to_word_onset from metadata; defaulting to False."
        )
    if locked_to_word_onset:
        LOGGER.info("Detected subsamples locked to word onset — enabling matrix-level cluster testing.")
    elif args.force_matrix_clusters:
        LOGGER.info(
            "Matrix-level cluster testing forced via --force-matrix-clusters despite unlocked subsamples."
        )

    run_matrix_clusters = locked_to_word_onset or args.force_matrix_clusters
    if args.skip_matrix_clusters and run_matrix_clusters:
        LOGGER.info(
            "Skipping matrix-level cluster testing due to --skip-matrix-clusters flag."
        )
        run_matrix_clusters = False

    n_models = len(args.models)
    n_subjects = len(subject_data)

    matrices_stack = np.stack([data[0] for data in subject_data], axis=0)
    lag_curves_stack = np.stack([data[1] for data in subject_data], axis=0)

    metadata_template = subject_data[0][2]
    neural_labels = metadata_template.get("neural_signal_labels") or [
        f"Neural {idx + 1}" for idx in range(matrices_stack.shape[1])
    ]
    if len(neural_labels) != matrices_stack.shape[1]:
        raise ValueError(
            "Mismatch between recorded neural signal labels and data dimensions in subject metadata."
        )

    raw_analysis_parameters = metadata_template.get("analysis_parameters")
    if raw_analysis_parameters is None:
        analysis_parameters_template: Dict[str, Any] = {}
    elif isinstance(raw_analysis_parameters, dict):
        analysis_parameters_template = raw_analysis_parameters
    else:
        try:
            analysis_parameters_template = dict(raw_analysis_parameters)
        except TypeError:
            analysis_parameters_template = {}

    permutation_alpha_caption_value = f"{args.permutation_alpha} ({CLUSTER_PERMUTATION_TAIL_DESCRIPTION})"
    base_caption_entries: List[Tuple[str, Any]] = [
        ("lag_metric", args.lag_metric),
        ("cluster_alpha", args.cluster_alpha),
        ("permutation_alpha", permutation_alpha_caption_value),
        ("n_permutations", args.n_permutations),
        ("permutation_seed", PERMUTATION_SEED),
        ("n_subjects", len(subjects)),
        ("subjects", ", ".join(subjects)),
        ("lock_subsample_to_word_onset", locked_to_word_onset),
        ("matrix_cluster_analysis", run_matrix_clusters),
        ("matrix_downsample_factor", matrix_downsample_factor),
    ]
    if args.simulation_noise:
        base_caption_entries.append(("simulation_neural_label", args.simulation_neural_label))

    resolution = metadata_template["analysis_parameters"]["resolution_hz"]
    lags_sec = lag_axis_reference / resolution
    n_neural = matrices_stack.shape[1]

    for neural_idx in range(n_neural):
        neural_label = neural_labels[neural_idx]
        suffix_fragment = _label_to_filename_fragment(neural_label)
        suffix = f"_{suffix_fragment}"

        matrices_neural = matrices_stack[:, neural_idx, ...]
        lag_curves_neural = lag_curves_stack[:, neural_idx, ...]

        avg_matrices = matrices_neural.mean(axis=0)
        avg_lag_curves = lag_curves_neural.mean(axis=0)
        sem_lag_curves = compute_sem(lag_curves_neural, axis=0)

        lag_significance_masks = []
        for model_idx in range(n_models):
            significant, _ = permutation_cluster_test(
                lag_curves_neural[:, model_idx, :],
                cluster_alpha=args.cluster_alpha,
                permutation_alpha=args.permutation_alpha,
                n_permutations=args.n_permutations,
            )
            lag_significance_masks.append(significant)
        lag_significance_masks = np.array(lag_significance_masks)

        matrix_significance_masks = np.zeros_like(avg_matrices, dtype=bool)
        if run_matrix_clusters:
            for model_idx in range(n_models):
                matrix_stack_model = matrices_neural[:, model_idx, :, :]
                try:
                    matrix_stack_down = _downsample_matrix_stack(
                        matrix_stack_model, matrix_downsample_factor
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to downsample matrices for model '{args.models[model_idx]}' "
                        f"with factor {matrix_downsample_factor}: {exc}"
                    ) from exc
                matrix_significant_coarse, _ = permutation_cluster_test_matrix(
                    matrix_stack_down,
                    cluster_alpha=args.cluster_alpha,
                    permutation_alpha=args.permutation_alpha,
                    n_permutations=args.n_permutations,
                )
                matrix_significant_full = _upsample_matrix_mask(
                    matrix_significant_coarse,
                    matrix_downsample_factor,
                    matrix_stack_model.shape[1:],
                )
                matrix_significance_masks[model_idx] = matrix_significant_full

        output_path_neural = _with_suffix(output_path, suffix)
        vector_output_neural = _with_suffix(vector_output_path, suffix)
        summary_cache_neural = _with_suffix(summary_cache, suffix)

        caption_entries = base_caption_entries + [("neural_label", neural_label)]
        analysis_caption = _format_analysis_caption(analysis_parameters_template, caption_entries)

        create_summary_plot(
            avg_matrices=avg_matrices,
            avg_lag_curves=avg_lag_curves,
            sem_lag_curves=sem_lag_curves,
            lags_sec=lags_sec,
            lag_significance_masks=lag_significance_masks,
            model_labels=args.models,
            output_path=output_path_neural,
            vector_output_path=vector_output_neural,
            neural_label=neural_label,
            analysis_caption=analysis_caption,
            matrix_significance_masks=matrix_significance_masks,
            locked_to_word_onset=locked_to_word_onset,
            matrix_extent_sec=(0.0, float(analysis_parameters_template.get("subsample_tps", matrices_neural.shape[-1]) / resolution),
                               0.0, float(analysis_parameters_template.get("subsample_tps", matrices_neural.shape[-1]) / resolution)),
        )

        analysis_settings: Dict[str, Any] = {
            "cluster_alpha": args.cluster_alpha,
            "permutation_alpha": args.permutation_alpha,
            "n_permutations": args.n_permutations,
            "permutation_seed": PERMUTATION_SEED,
            "permutation_tail": CLUSTER_PERMUTATION_TAIL,
            "permutation_cluster_signs": CLUSTER_PERMUTATION_SIGN_SCOPE,
            "lag_metric": args.lag_metric,
            "output_path": str(output_path_neural),
            "vector_output_path": str(vector_output_neural),
            "neural_label": neural_label,
            "neural_index": neural_idx,
            "locked_to_word_onset": locked_to_word_onset,
            "matrix_cluster_analysis": run_matrix_clusters,
            "matrix_downsample_factor": matrix_downsample_factor,
        }
        analysis_settings["input_dir"] = str(results_dir)
        analysis_settings["simulation_mode"] = args.simulation_noise
        if args.simulation_noise:
            analysis_settings["simulation_neural_label"] = args.simulation_neural_label
            if simulation_origin_filter:
                analysis_settings["simulation_origin"] = simulation_origin_filter
        analysis_settings["analysis_caption"] = analysis_caption
        analysis_settings["analysis_parameters"] = analysis_parameters_template
        if results_dir_mode == "analysis" and analysis_root is not None:
            analysis_settings["analysis_name"] = analysis_name
            analysis_settings["analysis_root"] = str(analysis_root)
            analysis_settings["single_subjects_dir"] = str(single_subjects_dir)
            analysis_settings["simulations_dir"] = str(analysis_root / "simulations")
            if group_level_dir is not None:
                analysis_settings["group_level_dir"] = str(group_level_dir)
        np.savez_compressed(
            summary_cache_neural,
            avg_matrices=avg_matrices,
            avg_lag_curves=avg_lag_curves,
            sem_lag_curves=sem_lag_curves,
            lags_sec=lags_sec,
            significance_masks=lag_significance_masks,
            model_labels=np.array(args.models, dtype=object),
            subjects=np.array(subjects, dtype=object),
            neural_label=np.array(neural_label, dtype=object),
            analysis_settings=np.array(analysis_settings, dtype=object),
            analysis_caption=np.array(analysis_caption, dtype=object),
            matrix_significance_masks=matrix_significance_masks,
        )

        LOGGER.info(
            "Saved group summary for '%s' (PNG: %s, PDF: %s) with cache at %s.",
            neural_label,
            output_path_neural,
            vector_output_neural,
            summary_cache_neural,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
