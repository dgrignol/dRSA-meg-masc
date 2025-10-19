#!/usr/bin/env python3
"""
Group-level dRSA cluster permutation analysis and summary plotting.

The script aggregates per-subject dRSA outputs produced by ``C1_dRSA_run.py`` and
tests whether the average lag-correlation curve shows significant clusters
following Maris & Oostenveld (2007). It also visualises the average dRSA matrix
for each model alongside the grand-average lag curve (with SEM shading) and
highlights significant clusters on the lag axis.

Expected per-subject files (per model):
    results/sub{ID}_res100_correlation_dRSA_matrices.npy
    results/sub{ID}_res100_correlation_metadata.json

Usage
-----
python D1_group_cluster_analysis.py \
    --subjects 01 02 03 \
    --models Envelope "Word Frequency" "GloVe" "GloVe Norm" \
    --output results/group_analysis.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        required=True,
        help="List of subject IDs (e.g., 01 02 03).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
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
        default=Path("results") / "group_dRSA_summary.png",
        help="Output path for the summary figure.",
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
    subject_id = int(subject)
    base = results_dir / f"sub{subject_id}_res100_{lag_metric}"
    matrices_path = Path(f"{base}_dRSA_matrices.npy")
    meta_path = Path(f"{base}_metadata.json")

    if not matrices_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing dRSA outputs for subject {subject} under {base}")

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

   # rely on matrices; compute lag curve as diagonal average relative to centre.
    lag_settings = metadata["analysis_parameters"]
    adtw_in_tps = lag_settings["averaging_window_tps"]

    lag_curves_subject = []
    for matrix in matrices_subset:
        lag_curve, _ = compute_lag_curve_from_matrix(matrix, adtw_in_tps)
        lag_curves_subject.append(lag_curve)
    lag_curves_subject = np.array(lag_curves_subject)

    return matrices_subset, lag_curves_subject, metadata


def compute_lag_curve_from_matrix(matrix: np.ndarray, adtw_in_tps: int) -> Tuple[np.ndarray, np.ndarray]:
    from functions.core_functions import compute_lag_correlation

    lags_tp, lag_corr = compute_lag_correlation(matrix, adtw_in_tps)
    return lag_corr, lags_tp


def find_clusters(stat_map: np.ndarray, threshold: float) -> List[np.ndarray]:
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
    rng = np.random.default_rng(0)

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
) -> None:
    n_models = len(model_labels)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 3 * n_models), constrained_layout=True)

    if n_models == 1:
        axes = axes[None, :]

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
            baseline = (curve - sem).min() - 0.02
            for cluster in clusters:
                ax_lag.hlines(
                    y=baseline,
                    xmin=lags_sec[cluster[0]],
                    xmax=lags_sec[cluster[-1]],
                    color="red",
                    linewidth=3,
                )

        ax_lag.axvline(0, color="k", linestyle="--", linewidth=0.75)
        ax_lag.axhline(0, color="k", linestyle="--", linewidth=0.75)
        ax_lag.set_title(f"{label} | Lag correlation")
        ax_lag.set_xlabel("Lag (s)")
        ax_lag.set_ylabel("Correlation")
        ax_lag.legend(loc="upper right")

    fig.suptitle("Group-level dRSA summary", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    repo_root = read_repository_root()
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir

    subjects = [s.lstrip("sub-") for s in args.subjects]
    subject_data = []

    for subject in subjects:
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

    avg_matrices = matrices_stack.mean(axis=0)
    avg_lag_curves = lag_curves_stack.mean(axis=0)
    sem_lag_curves = compute_sem(lag_curves_stack, axis=0)

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
        output_path=args.output,
    )

    LOGGER.info("Saved group summary figure to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
