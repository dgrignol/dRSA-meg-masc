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
from scipy.stats import ttest_rel, t

from functions.generic_helpers import (
    ensure_analysis_directories,
    find_latest_analysis_directory,
    read_repository_root,
)


LOGGER = logging.getLogger(__name__)


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
        default=5000,
        help="Number of permutations for the cluster test (default: 5000).",
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
        "--log-level",
        default="INFO",
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The subset of dRSA matrices for the requested models, their lag curves,
        and the metadata dictionary parsed from disk.
    """
    subject_id = int(subject)
    prefix_candidates = [
        f"sub-{subject_id:02d}_res100_{lag_metric}",
        f"sub{subject_id}_res100_{lag_metric}",
    ]

    matrices_path: Optional[Path] = None
    meta_path: Optional[Path] = None

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
    rng = np.random.default_rng(0)  # Fixed seed for reproducible cluster thresholds across runs.

    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_subjects)[:, None]
        perm_data = data * signs
        t_perm, _ = ttest_rel(perm_data, np.zeros_like(perm_data), axis=0)
        perm_clusters = find_clusters(t_perm, threshold)
        if perm_clusters:
            max_sums[i] = np.max([abs(np.sum(t_perm[c])) for c in perm_clusters])
        else:
            max_sums[i] = 0.0

    cluster_threshold = np.quantile(max_sums, 1 - permutation_alpha)
    significant = np.zeros(n_times, dtype=bool)

    for cluster, cluster_sum in zip(clusters, cluster_sums):
        if abs(cluster_sum) > cluster_threshold:
            significant[cluster] = True

    return significant, t_vals


def compute_sem(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the standard error of the mean along the requested axis."""
    n = data.shape[axis]
    if n < 2:
        return np.zeros_like(data.take(indices=0, axis=axis), dtype=np.float32)
    return data.std(axis=axis, ddof=1) / np.sqrt(n)


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
    significance_masks: np.ndarray,
    model_labels: Sequence[str],
    output_path: Path,
    vector_output_path: Optional[Path] = None,
    neural_label: Optional[str] = None,
    analysis_caption: Optional[str] = None,
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
    significance_masks:
        Boolean array flagging significant time-points returned by ``permutation_cluster_test``.
    model_labels:
        Human-readable labels for the models (used in subplot titles and legend).
    output_path:
        Destination path for the rasterised PNG figure (directories created if necessary).
    vector_output_path:
        Optional path for a vector export (e.g., PDF) of the same figure.
    analysis_caption:
        Optional string describing analysis parameters to display beneath the subplots.
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
        )
        ax_matrix.set_title(f"{label} | Average dRSA")
        ax_matrix.set_xlabel("Model time")
        ax_matrix.set_ylabel("Neural time")
        fig.colorbar(im, ax=ax_matrix, fraction=0.04, pad=0.00005)

        ax_lag = axes[idx, 1]
        curve = avg_lag_curves[idx]
        sem = sem_lag_curves[idx]
        significant = significance_masks[idx]

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

    title = "Group-level dRSA summary"
    if neural_label:
        title = f"{title} | {neural_label}"
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

    # Resolve repository-relative paths so the script can run from any working directory.
    repo_root = read_repository_root()

    analysis_root: Optional[Path] = None
    group_level_dir: Optional[Path] = None
    results_dir_mode = "analysis"
    analysis_name = args.analysis_name

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
        LOGGER.info("Subject-level inputs: %s", results_dir)
        LOGGER.info("Group-level output directory: %s", group_level_dir)

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
                significance_masks = cache_payload["significance_masks"]
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
            output_path_local = output_path
            vector_output_local = vector_output_path
            if isinstance(analysis_settings, dict):
                output_path_local = _resolve_output_path(
                    analysis_settings.get("output_path"), output_path_local
                )
                vector_output_local = _resolve_output_path(
                    analysis_settings.get("vector_output_path"), vector_output_local
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

            create_summary_plot(
                avg_matrices=avg_matrices,
                avg_lag_curves=avg_lag_curves,
                sem_lag_curves=sem_lag_curves,
                lags_sec=lags_sec,
                significance_masks=significance_masks,
                model_labels=model_labels,
                output_path=output_path_local,
                vector_output_path=vector_output_local,
                neural_label=neural_label,
                analysis_caption=analysis_caption,
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

    subjects = [s.lstrip("sub-") for s in args.subjects]
    subject_data = []
    lag_axis_reference: Optional[np.ndarray] = None

    for subject in subjects:
        matrices, lag_curves, metadata, lags_tp_axis = load_subject_data(
            subject,
            args.models,
            results_dir,
            args.lag_metric,
        )
        subject_data.append((matrices, lag_curves, metadata))
        if lag_axis_reference is None:
            lag_axis_reference = lags_tp_axis
        elif lag_axis_reference.shape != lags_tp_axis.shape or not np.array_equal(lag_axis_reference, lags_tp_axis):
            raise ValueError(
                "Lag axis mismatch across subjects; ensure all subjects were analysed with the same settings."
            )

    if lag_axis_reference is None:
        raise RuntimeError("Unable to determine lag axis from subject data.")

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

    base_caption_entries: List[Tuple[str, Any]] = [
        ("lag_metric", args.lag_metric),
        ("cluster_alpha", args.cluster_alpha),
        ("permutation_alpha", args.permutation_alpha),
        ("n_permutations", args.n_permutations),
        ("n_subjects", len(subjects)),
        ("subjects", ", ".join(subjects)),
    ]

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

        significance_masks = []
        for model_idx in range(n_models):
            significant, _ = permutation_cluster_test(
                lag_curves_neural[:, model_idx, :],
                cluster_alpha=args.cluster_alpha,
                permutation_alpha=args.permutation_alpha,
                n_permutations=args.n_permutations,
            )
            significance_masks.append(significant)
        significance_masks = np.array(significance_masks)

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
            significance_masks=significance_masks,
            model_labels=args.models,
            output_path=output_path_neural,
            vector_output_path=vector_output_neural,
            neural_label=neural_label,
            analysis_caption=analysis_caption,
        )

        analysis_settings: Dict[str, Any] = {
            "cluster_alpha": args.cluster_alpha,
            "permutation_alpha": args.permutation_alpha,
            "n_permutations": args.n_permutations,
            "lag_metric": args.lag_metric,
            "output_path": str(output_path_neural),
            "vector_output_path": str(vector_output_neural),
            "neural_label": neural_label,
            "neural_index": neural_idx,
        }
        analysis_settings["analysis_caption"] = analysis_caption
        analysis_settings["analysis_parameters"] = analysis_parameters_template
        if results_dir_mode == "analysis" and analysis_root is not None:
            analysis_settings["analysis_name"] = analysis_name
            analysis_settings["analysis_root"] = str(analysis_root)
            analysis_settings["single_subjects_dir"] = str(results_dir)
            if group_level_dir is not None:
                analysis_settings["group_level_dir"] = str(group_level_dir)
        np.savez_compressed(
            summary_cache_neural,
            avg_matrices=avg_matrices,
            avg_lag_curves=avg_lag_curves,
            sem_lag_curves=sem_lag_curves,
            lags_sec=lags_sec,
            significance_masks=significance_masks,
            model_labels=np.array(args.models, dtype=object),
            subjects=np.array(subjects, dtype=object),
            neural_label=np.array(neural_label, dtype=object),
            analysis_settings=np.array(analysis_settings, dtype=object),
            analysis_caption=np.array(analysis_caption, dtype=object),
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
