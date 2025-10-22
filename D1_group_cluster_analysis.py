#!/usr/bin/env python3
"""
Group-level dRSA cluster permutation analysis and summary plotting.

The script aggregates per-subject dRSA outputs produced by ``C1_dRSA_run.py`` and
tests whether the average lag-correlation curve shows significant clusters
following Maris & Oostenveld (2007). It also visualises the average dRSA matrix
for each model alongside the grand-average lag curve (with SEM shading), highlights
significant clusters on the lag axis, and annotates their peak correlation/lag
coordinates. Outputs are stored in ``results/group_level`` as both PNG and PDF,
and a cached ``.npz`` bundle makes it possible to regenerate the figures without
re-running the permutation test.

Expected per-subject files (per model):
    results/sub-{ID}_res100_correlation_dRSA_matrices.npy
    results/sub-{ID}_res100_correlation_metadata.json

Usage
-----
python D1_group_cluster_analysis.py \
    --subjects 01 02 03 \
    --models Envelope "Word Frequency" "GloVe" "GloVe Norm" \
    --output results/group_level/group_dRSA_summary.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel, t

from functions.generic_helpers import read_repository_root


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
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per-subject results (default: results).",
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
        default=Path("results") / "group_level" / "group_dRSA_summary.png",
        help="Output path for the summary figure (PNG will be written here).",
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

    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    model_indices = []
    for model in models:
        if model not in label_to_idx:
            raise KeyError(f"Model '{model}' not found for subject {subject}; available: {labels}")
        model_indices.append(label_to_idx[model])

    matrices_subset = matrices[model_indices]

    # Recover analysis settings required to translate lag steps into seconds.
    lag_settings = metadata["analysis_parameters"]
    adtw_in_tps = lag_settings["averaging_window_tps"]

    lag_curves_subject = []
    for matrix in matrices_subset:
        lag_curve, _ = compute_lag_curve_from_matrix(matrix, adtw_in_tps)
        lag_curves_subject.append(lag_curve)
    lag_curves_subject = np.array(lag_curves_subject)

    return matrices_subset, lag_curves_subject, metadata


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


def create_summary_plot(
    avg_matrices: np.ndarray,
    avg_lag_curves: np.ndarray,
    sem_lag_curves: np.ndarray,
    lags_sec: np.ndarray,
    significance_masks: np.ndarray,
    model_labels: Sequence[str],
    output_path: Path,
    vector_output_path: Optional[Path] = None,
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
    """
    n_models = len(model_labels)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 3 * n_models), constrained_layout=True)

    if n_models == 1:
        axes = axes[None, :]

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
        fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

        ax_lag = axes[idx, 1]
        curve = avg_lag_curves[idx]
        sem = sem_lag_curves[idx]
        significant = significance_masks[idx]

        ax_lag.plot(lags_sec, curve, color="k", label="Mean lag corr")
        ax_lag.fill_between(lags_sec, curve - sem, curve + sem, color="gray", alpha=0.3, label="SEM")

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

    fig.suptitle("Group-level dRSA summary", fontsize=14, fontweight="bold")
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
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    # Always emit a vector copy alongside the raster PNG for downstream figure editing.
    vector_output_path = output_path.with_suffix(".pdf")
    summary_cache = args.summary_cache
    if summary_cache is None:
        summary_cache = output_path.with_suffix(".npz")
    if not summary_cache.is_absolute():
        summary_cache = repo_root / summary_cache
    # Ensure parent folders exist before we attempt to save figures or cache files.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_cache.parent.mkdir(parents=True, exist_ok=True)
    vector_output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        # Recreate the plots directly from the cached group-level results.
        if not summary_cache.exists():
            raise FileNotFoundError(
                f"Summary cache not found at {summary_cache}. Run without --plot-only first."
            )
        with np.load(summary_cache, allow_pickle=True) as cache_payload:
            avg_matrices = cache_payload["avg_matrices"]
            avg_lag_curves = cache_payload["avg_lag_curves"]
            sem_lag_curves = cache_payload["sem_lag_curves"]
            lags_sec = cache_payload["lags_sec"]
            significance_masks = cache_payload["significance_masks"]
            model_labels = cache_payload["model_labels"].tolist()

        create_summary_plot(
            avg_matrices=avg_matrices,
            avg_lag_curves=avg_lag_curves,
            sem_lag_curves=sem_lag_curves,
            lags_sec=lags_sec,
            significance_masks=significance_masks,
            model_labels=model_labels,
            output_path=output_path,
            vector_output_path=vector_output_path,
        )
        LOGGER.info(
            "Regenerated group summary figure (PNG: %s, PDF: %s) using cached results (%s).",
            output_path,
            vector_output_path,
            summary_cache,
        )
        return 0

    if not args.subjects:
        raise ValueError("No subjects provided. Specify --subjects unless using --plot-only.")
    if not args.models:
        raise ValueError("No models provided. Specify --models unless using --plot-only.")

    subjects = [s.lstrip("sub-") for s in args.subjects]
    subject_data = []

    for subject in subjects:
        # Each entry of subject_data bundles the matrices, lag curves, and metadata per subject.
        matrices, lag_curves, metadata = load_subject_data(
            subject,
            args.models,
            results_dir,
            args.lag_metric,
        )
        subject_data.append((matrices, lag_curves, metadata))

    n_models = len(args.models)
    n_subjects = len(subject_data)

    matrices_stack = np.array([data[0] for data in subject_data])
    lag_curves_stack = np.array([data[1] for data in subject_data])

    adtw_in_tps = subject_data[0][2]["analysis_parameters"]["averaging_window_tps"]
    resolution = subject_data[0][2]["analysis_parameters"]["resolution_hz"]
    _, lag_tp_axis = compute_lag_curve_from_matrix(subject_data[0][0][0], adtw_in_tps)
    lags_sec = lag_tp_axis / resolution

    # Collapse subject stacks into group averages and variability estimates.
    avg_matrices = matrices_stack.mean(axis=0)
    avg_lag_curves = lag_curves_stack.mean(axis=0)
    sem_lag_curves = compute_sem(lag_curves_stack, axis=0)

    # Run the cluster-permutation test for each model's lag curve.
    significance_masks = []
    for model_idx in range(n_models):
        significant, _ = permutation_cluster_test(
            lag_curves_stack[:, model_idx, :],
            cluster_alpha=args.cluster_alpha,
            permutation_alpha=args.permutation_alpha,
            n_permutations=args.n_permutations,
        )
        significance_masks.append(significant)
    significance_masks = np.array(significance_masks)

    create_summary_plot(
        avg_matrices=avg_matrices,
        avg_lag_curves=avg_lag_curves,
        sem_lag_curves=sem_lag_curves,
        lags_sec=lags_sec,
        significance_masks=significance_masks,
        model_labels=args.models,
        output_path=output_path,
        vector_output_path=vector_output_path,
    )
    # Persist all outputs required to regenerate the figure without rerunning the cluster test.
    analysis_settings: Dict[str, Any] = {
        "cluster_alpha": args.cluster_alpha,
        "permutation_alpha": args.permutation_alpha,
        "n_permutations": args.n_permutations,
        "lag_metric": args.lag_metric,
        "output_path": str(output_path),
        "vector_output_path": str(vector_output_path),
    }
    np.savez_compressed(
        summary_cache,
        avg_matrices=avg_matrices,
        avg_lag_curves=avg_lag_curves,
        sem_lag_curves=sem_lag_curves,
        lags_sec=lags_sec,
        significance_masks=significance_masks,
        model_labels=np.array(args.models, dtype=object),
        subjects=np.array(subjects, dtype=object),
        analysis_settings=np.array(analysis_settings, dtype=object),
    )

    LOGGER.info(
        "Saved group summary figure (PNG: %s, PDF: %s) and cached analysis outputs at %s.",
        output_path,
        vector_output_path,
        summary_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
