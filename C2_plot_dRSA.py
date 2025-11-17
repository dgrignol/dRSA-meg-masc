import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from functions.core_functions import compute_lag_correlation
from functions.generic_helpers import (
    find_latest_analysis_directory,
    format_log_timestamp,
    normalise_analysis_name,
    read_repository_root,
    rebase_path_to_known_root,
)


def bootstrap_mean_ci(data, n_bootstraps=1000, confidence=0.95, random_state=None):
    """Compute bootstrap confidence interval for the mean curve."""
    curves = np.asarray(data)
    if curves.ndim != 2:
        raise ValueError("`data` must be 2D (samples, points).")
    if curves.shape[0] == 0:
        raise ValueError("No samples available for bootstrapping.")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    n_samples, _ = curves.shape
    indices = rng.integers(0, n_samples, size=(n_bootstraps, n_samples))
    sample_means = curves[indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(sample_means, 100 * alpha, axis=0)
    upper = np.percentile(sample_means, 100 * (1 - alpha), axis=0)
    return lower, upper


def generate_plots(
    rsa_matrices,
    lag_curves_array,
    model_labels,
    neural_labels,
    analysis_parameters,
    lag_bootstrap_settings,
    plot_path,
    fixed_ylim: tuple[float, float] | None = None,
):
    """Render per-neural dRSA figures combining RSA matrices and lag curves.

    Parameters
    ----------
    rsa_matrices : array-like
        Nested array shaped (n_neural, n_models, n_tp, n_tp) containing the
        averaged dRSA matrices for every neural/model pair.
    lag_curves_array : array-like or None
        Optional bootstrap samples of lag correlation curves with shape
        (n_neural, n_models, n_iterations, n_lags). When provided, confidence
        intervals are computed per curve.
    model_labels : Sequence[str]
        Ordered labels identifying the computational models.
    neural_labels : Sequence[str]
        Ordered labels describing each neural signal or ROI analysed.
    analysis_parameters : Mapping[str, Any]
        Dictionary persisted by ``C1_dRSA_run.py`` containing analysis metadata
        such as the averaging window (tps) and sampling resolution (Hz).
    lag_bootstrap_settings : Mapping[str, Any]
        Settings used to resample lag curves (iterations, confidence, seed).
    plot_path : str or Path
        Base output path. When multiple neural signals are present, the label is
        appended to the filename before saving.
    fixed_ylim : tuple[float, float], optional
        If provided, force the lag-correlation axes to use this y-range.

    Returns
    -------
    List[str]
        File paths for every generated figure (one entry per neural label).
    """
    adtw_in_tps = analysis_parameters["averaging_window_tps"]
    resolution = analysis_parameters["resolution_hz"]
    caption_timestamp = format_log_timestamp()

    lag_curves_available = lag_curves_array is not None
    if lag_curves_available:
        lag_curves_array = np.asarray(lag_curves_array)

    lag_bootstrap_iterations = lag_bootstrap_settings.get("iterations", 0)
    lag_bootstrap_confidence = lag_bootstrap_settings.get("confidence", 0.95)
    lag_bootstrap_random_state = lag_bootstrap_settings.get("random_state", None)

    rsa_matrices = np.asarray(rsa_matrices)
    n_neural, n_models = rsa_matrices.shape[:2]

    def _sanitize_label(label):
        safe = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in str(label)
        ).strip("_")
        return safe or "neural"

    plot_dir = os.path.dirname(plot_path) or "."
    os.makedirs(plot_dir, exist_ok=True)

    saved_paths = []

    for neural_idx, neural_label in enumerate(neural_labels):
        fig, axs = plt.subplots(
            n_models,
            2,
            figsize=(12, max(3.0, 3.0 * n_models)),
            squeeze=False,
            constrained_layout=True,
        )

        for model_idx, model_name in enumerate(model_labels):
            ax_matrix = axs[model_idx, 0]
            ax_lag = axs[model_idx, 1]

            rsa_matrix = rsa_matrices[neural_idx, model_idx]
            lag_curves = (
                lag_curves_array[neural_idx, model_idx]
                if lag_curves_available
                else None
            )

            lags_tp, lag_corr = compute_lag_correlation(rsa_matrix, adtw_in_tps)
            ci_lower = ci_upper = None

            if lag_curves is not None and lag_bootstrap_iterations:
                ci_lower, ci_upper = bootstrap_mean_ci(
                    lag_curves,
                    n_bootstraps=lag_bootstrap_iterations,
                    confidence=lag_bootstrap_confidence,
                    random_state=lag_bootstrap_random_state,
                )

            lags_sec = lags_tp / resolution
            lag_span = float(lags_sec[-1] - lags_sec[0]) if lags_sec.size > 1 else 1.0
            curve_range = np.ptp(lag_corr)
            curve_span = float(curve_range) if curve_range > 0 else 1.0

            im = ax_matrix.imshow(
                rsa_matrix,
                cmap="viridis",
                aspect="auto",
                origin="lower",
            )
            ax_matrix.set_title(
                f"{model_name}",
                fontsize=10,
                fontweight="bold",
            )
            ax_matrix.set_xlabel("Model time", fontsize=8)
            ax_matrix.set_ylabel("Neural time", fontsize=8)
            fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

            ax_lag.plot(
                lags_sec,
                lag_corr,
                color="black",
                linewidth=1.1,
                label="Lag corr",
            )
            if ci_lower is not None and ci_upper is not None:
                ax_lag.fill_between(
                    lags_sec,
                    ci_lower,
                    ci_upper,
                    color="gray",
                    alpha=0.3,
                    label="CI",
                )
            ax_lag.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax_lag.axhline(0, color="black", linestyle="--", linewidth=0.75)
            ax_lag.set_xlabel("Lag (neural time - model time) [s]", fontsize=8)
            ax_lag.set_ylabel("Correlation", fontsize=8)
            ax_lag.set_title(
                f"{model_name}",
                fontsize=10,
                fontweight="bold",
            )
            if fixed_ylim is not None:
                ax_lag.set_ylim(fixed_ylim)

            peak_idx = np.argmax(lag_corr)
            peak_lag = lags_sec[peak_idx]
            peak_val = lag_corr[peak_idx]

            ax_lag.scatter(
                [peak_lag],
                [peak_val],
                color="red",
                edgecolors="white",
                linewidths=0.5,
                s=12,
                zorder=5,
            )

            direction_x = 1 if peak_lag >= 0 else -1
            if np.isclose(peak_lag, 0.0):
                direction_x = 1
            direction_y = 1 if peak_val >= 0 else -1
            if np.isclose(peak_val, 0.0):
                direction_y = 1

            text_dx = 0.03 * lag_span * direction_x
            text_dy = 0.08 * curve_span * direction_y

            ax_lag.annotate(
                f"Peak: {peak_val:.3f}\nLag: {peak_lag:.3f}s",
                xy=(peak_lag, peak_val),
                xytext=(peak_lag + text_dx, peak_val + text_dy),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
                fontsize=8,
                color="red",
                ha="left" if direction_x >= 0 else "right",
                va="bottom" if direction_y >= 0 else "top",
            )

            if model_idx == 0:
                ax_lag.legend(loc="upper left", fontsize=8)

        parameter_pairs = list(analysis_parameters.items()) if analysis_parameters else []
        parameter_pairs.append(("timestamp", caption_timestamp))
        parameter_caption = ", ".join(f"{key}={value}" for key, value in parameter_pairs)
        parameter_caption = "Parameters: " + parameter_caption
        parameter_caption = textwrap.fill(parameter_caption, width=110)

        fig.suptitle(f"{neural_label} dRSA summary", fontsize=14, fontweight="bold")
        fig.text(0.5, 0.005, parameter_caption, ha="center", va="center", fontsize=8)

        base, ext = os.path.splitext(plot_path)
        ext = ext or ".png"
        if n_neural == 1:
            figure_path = plot_path if plot_path.endswith(ext) else f"{plot_path}{ext}"
        else:
            figure_path = f"{base}_{_sanitize_label(neural_label)}{ext}"

        fig.savefig(figure_path, dpi=300)
        saved_paths.append(figure_path)
        plt.close(fig)
    return saved_paths


def discover_metadata_files(
    results_dir: Path,
    base_run_id: str,
    include_base: bool = True,
    include_simulations: bool = True,
) -> list[Path]:
    """Return metadata files for the requested run id within a directory."""
    results_dir = Path(results_dir)
    metadata_files: list[Path] = []
    if include_base:
        base_metadata = results_dir / f"{base_run_id}_metadata.json"
        if base_metadata.exists():
            metadata_files.append(base_metadata)
    if include_simulations:
        sim_pattern = f"{base_run_id}_sim_*_metadata.json"
        metadata_files.extend(sorted(results_dir.glob(sim_pattern)))
    return metadata_files


def generate_autocorr_summary_plot(
    entries: list[dict],
    plot_path: Path,
) -> Path:
    """Render stacked autocorrelation plots (matrix + lag curve) for simulation runs."""
    if not entries:
        return plot_path

    entries_sorted = sorted(entries, key=lambda item: item.get("order", 0))
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(entries_sorted)
    fig, axs = plt.subplots(
        n_rows,
        2,
        figsize=(12, max(3.0, 3.0 * n_rows)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, entry in enumerate(entries_sorted):
        rsa_matrix = entry["rsa_matrix"]
        analysis_parameters = entry["analysis_parameters"]
        lag_bootstrap_settings = entry["lag_bootstrap_settings"]
        label = entry["label"]

        ax_matrix = axs[row_idx, 0]
        ax_lag = axs[row_idx, 1]

        im = ax_matrix.imshow(rsa_matrix, cmap="viridis", aspect="auto", origin="lower")
        ax_matrix.set_title(label, fontsize=10, fontweight="bold")
        ax_matrix.set_xlabel("Model time", fontsize=8)
        ax_matrix.set_ylabel("Neural time", fontsize=8)
        fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

        adtw_in_tps = analysis_parameters["averaging_window_tps"]
        resolution = analysis_parameters["resolution_hz"]
        lags_tp, lag_corr = compute_lag_correlation(rsa_matrix, adtw_in_tps)
        lags_sec = lags_tp / resolution

        ax_lag.plot(lags_sec, lag_corr, color="black", linewidth=1.1, label="Lag corr")

        lag_samples = entry.get("lag_samples")
        lag_bootstrap_iterations = lag_bootstrap_settings.get("iterations", 0)
        lag_bootstrap_confidence = lag_bootstrap_settings.get("confidence", 0.95)
        lag_bootstrap_random_state = lag_bootstrap_settings.get("random_state", None)
        if lag_samples is not None and lag_bootstrap_iterations:
            ci_lower, ci_upper = bootstrap_mean_ci(
                lag_samples,
                n_bootstraps=lag_bootstrap_iterations,
                confidence=lag_bootstrap_confidence,
                random_state=lag_bootstrap_random_state,
            )
            ax_lag.fill_between(
                lags_sec,
                ci_lower,
                ci_upper,
                color="gray",
                alpha=0.3,
                label="CI",
            )

        ax_lag.axvline(0, color="black", linestyle="--", linewidth=0.75)
        ax_lag.axhline(0, color="black", linestyle="--", linewidth=0.75)
        ax_lag.set_xlabel("Lag (neural - model) [s]", fontsize=8)
        ax_lag.set_ylabel("Correlation", fontsize=8)
        ax_lag.set_title(f"{label}", fontsize=10, fontweight="bold")

        peak_idx = np.argmax(lag_corr)
        peak_lag = lags_sec[peak_idx]
        peak_val = lag_corr[peak_idx]
        ax_lag.scatter(
            [peak_lag],
            [peak_val],
            color="red",
            edgecolors="white",
            linewidths=0.5,
            s=12,
            zorder=5,
        )

        direction_x = 1 if peak_lag >= 0 else -1
        if np.isclose(peak_lag, 0.0):
            direction_x = 1
        direction_y = 1 if peak_val >= 0 else -1
        if np.isclose(peak_val, 0.0):
            direction_y = 1
        lag_span = float(lags_sec[-1] - lags_sec[0]) if lags_sec.size > 1 else 1.0
        curve_span = float(np.ptp(lag_corr)) or 1.0
        text_dx = 0.03 * lag_span * direction_x
        text_dy = 0.08 * curve_span * direction_y
        ax_lag.annotate(
            f"Peak: {peak_val:.3f}\nLag: {peak_lag:.3f}s",
            xy=(peak_lag, peak_val),
            xytext=(peak_lag + text_dx, peak_val + text_dy),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            fontsize=8,
            color="red",
            ha="left" if direction_x >= 0 else "right",
            va="bottom" if direction_y >= 0 else "top",
        )
        if row_idx == 0:
            ax_lag.legend(loc="upper left", fontsize=8)

    parameter_caption = ", ".join(
        f"{key}={value}" for key, value in entries_sorted[0]["analysis_parameters"].items()
    )
    parameter_caption = "Parameters: " + parameter_caption
    parameter_caption = textwrap.fill(parameter_caption, width=110)
    fig.suptitle("Simulation autocorrelations", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, parameter_caption, ha="center", va="center", fontsize=8)
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def plot_run_from_metadata(
    metadata_path: Path,
    results_dir: Path,
    custom_ylim: tuple[float, float] | None = None,
) -> tuple[str, list[str], dict | None]:
    """Load metadata for a single run and generate the corresponding plots.

    When ``custom_ylim`` is provided, a second copy of each plot is created with
    the lag-correlation axis constrained to that range and a descriptive suffix.
    """
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    run_id = metadata.get("analysis_run_id") or metadata_path.stem.replace("_metadata", "")
    outputs = metadata.get("outputs", {})

    def resolve_output_path(
        key: str, fallback_name: str, require_exists: bool = True
    ) -> tuple[Path, list[Path]]:
        stored_path = outputs.get(key)
        attempts: list[Path] = []
        if stored_path:
            candidate = rebase_path_to_known_root(Path(stored_path))
            if not candidate.is_absolute():
                candidate = (results_dir / candidate).resolve()
            attempts.append(candidate)
            if not require_exists or candidate.exists():
                return candidate, attempts
        fallback_path = results_dir / Path(fallback_name)
        attempts.append(fallback_path)
        if not require_exists or fallback_path.exists():
            return fallback_path, attempts
        return fallback_path, attempts

    rsa_default = f"{run_id}_dRSA_matrices.npy"
    rsa_matrices_path, rsa_attempts = resolve_output_path("rsa_matrices", rsa_default)
    if not rsa_matrices_path.exists():
        attempted = ", ".join(str(p) for p in rsa_attempts)
        raise FileNotFoundError(
            f"[{run_id}] RSA matrices file not found after checking: {attempted}"
        )
    rsa_matrices = np.load(rsa_matrices_path)

    lag_default = f"{run_id}_lag_curves.npy"
    lag_curves_path, lag_attempts = resolve_output_path("lag_curves", lag_default)
    lag_curves_array = None
    if lag_curves_path.exists():
        lag_curves_array = np.load(lag_curves_path)
    else:
        if outputs.get("lag_curves"):
            attempted = ", ".join(str(p) for p in lag_attempts)
            print(f"! [{run_id}] lag curves referenced in metadata but missing: {attempted}")
        else:
            print(f"! [{run_id}] lag curves not available; plotting without confidence intervals.")

    plot_default = f"{run_id}_plot.png"
    if outputs.get("plot_target"):
        plot_path, _ = resolve_output_path("plot_target", plot_default, require_exists=False)
    elif outputs.get("plot"):
        plot_path, _ = resolve_output_path("plot", plot_default, require_exists=False)
    else:
        plot_path = results_dir / plot_default

    model_labels = metadata["selected_model_labels"]
    neural_labels = metadata["neural_signal_labels"]
    analysis_parameters = metadata["analysis_parameters"]
    lag_bootstrap_settings = metadata.get("lag_bootstrap_settings", {})

    plot_path_str = str(plot_path)

    def add_suffix(path_str: str, suffix: str) -> str:
        base, ext = os.path.splitext(path_str)
        return f"{base}{suffix}{ext}"

    saved_paths = generate_plots(
        rsa_matrices,
        lag_curves_array,
        model_labels,
        neural_labels,
        analysis_parameters,
        lag_bootstrap_settings,
        plot_path_str,
    )
    if custom_ylim is not None:
        ymin, ymax = custom_ylim
        suffix = f"_ylim_{ymin:g}_{ymax:g}"
        custom_plot_path = add_suffix(plot_path_str, suffix)
        custom_paths = generate_plots(
            rsa_matrices,
            lag_curves_array,
            model_labels,
            neural_labels,
            analysis_parameters,
            lag_bootstrap_settings,
            custom_plot_path,
            fixed_ylim=custom_ylim,
        )
        saved_paths.extend(custom_paths)
    autocorr_entry = None
    simulation_info = metadata.get("simulation") or {}
    if simulation_info.get("enabled"):
        if neural_labels:
            neural_label = neural_labels[0]
            try:
                model_idx = model_labels.index(neural_label)
            except ValueError:
                model_idx = None
            if model_idx is not None:
                rsa_slice = rsa_matrices[0, model_idx]
                lag_samples = (
                    lag_curves_array[0, model_idx]
                    if lag_curves_array is not None
                    else None
                )
                autocorr_entry = {
                    "label": neural_label,
                    "rsa_matrix": rsa_slice,
                    "lag_samples": lag_samples,
                    "analysis_parameters": analysis_parameters,
                    "lag_bootstrap_settings": lag_bootstrap_settings,
                    "output_dir": results_dir,
                    "order": simulation_info.get("run_index", 0),
                }
    return run_id, saved_paths, autocorr_entry


def build_analysis_run_id(subject_label, resolution, rsa_method):
    return f"{subject_label}_res{resolution}_{rsa_method}"


def main():
    parser = argparse.ArgumentParser(description="Plot cached dRSA results.")
    parser.add_argument(
        "subject",
        help="Subject label or integer ID (e.g., 'sub-01' or '1').",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="Sampling resolution used for the analysis (default: 100 Hz).",
    )
    parser.add_argument(
        "--rsa-method",
        default="correlation",
        help="RSA computation method suffix used in the analysis run id.",
    )
    parser.add_argument(
        "--analysis-id",
        help="Explicit analysis run id. Overrides subject/resolution/rsa-method options.",
    )
    parser.add_argument(
        "--analysis-name",
        help=(
            "Named analysis folder under --results-root. "
            "When omitted, the most recent analysis is selected automatically."
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
        help="Legacy override pointing directly to cached dRSA outputs.",
    )
    parser.add_argument(
        "--custom-ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help=(
            "When provided, also save a copy of each plot with the lag y-axis fixed "
            "to the supplied range (min max)."
        ),
    )
    args = parser.parse_args()

    custom_ylim: tuple[float, float] | None = None
    if args.custom_ylim is not None:
        ymin, ymax = args.custom_ylim
        if ymin >= ymax:
            parser.error("--custom-ylim requires YMIN to be smaller than YMAX.")
        custom_ylim = (ymin, ymax)

    if args.subject.startswith("sub-"):
        subject_label = args.subject
    else:
        subject_label = f"sub-{int(args.subject):02d}"

    repo_root = read_repository_root()

    if args.analysis_id:
        analysis_run_id = args.analysis_id
    else:
        analysis_run_id = build_analysis_run_id(
            subject_label,
            args.resolution,
            args.rsa_method,
        )

    analysis_root: Optional[Path] = None
    analysis_name = args.analysis_name

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = (repo_root / results_dir).resolve()
        else:
            results_dir = results_dir.resolve()
        print(f"Using legacy results directory: {results_dir}")
    else:
        results_root = args.results_root
        if not results_root.is_absolute():
            results_root = (repo_root / results_root).resolve()
        else:
            results_root = results_root.resolve()
        if analysis_name:
            normalised_name = normalise_analysis_name(analysis_name)
            analysis_root = (results_root / normalised_name).resolve()
        else:
            latest_root = find_latest_analysis_directory(results_root)
            if latest_root is None:
                raise FileNotFoundError(
                    f"No analysis directories found under {results_root}. "
                    "Run C1_dRSA_run.py first or provide --analysis-name/--results-dir."
                )
            analysis_root = latest_root
            normalised_name = latest_root.name
        if not analysis_root.exists():
            raise FileNotFoundError(
                f"Analysis directory {analysis_root} does not exist. "
                "Ensure C1_dRSA_run.py has finished or adjust --analysis-name."
            )
        analysis_name = normalised_name
        results_dir = (analysis_root / "single_subjects").resolve()
        print(f"Analysis: {analysis_name} ({analysis_root})")
        print(f"Reading subject outputs from {results_dir}")

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    directories: list[tuple[Path, bool, bool]] = []
    if args.results_dir is not None:
        directories.append((results_dir, True, True))
    else:
        directories.append((results_dir, True, False))
        simulation_dir = analysis_root / "simulations"
        if simulation_dir.exists():
            directories.append((simulation_dir, False, True))

    metadata_jobs: list[tuple[Path, Path]] = []
    if args.analysis_id:
        for directory, _, _ in directories:
            candidate = directory / f"{args.analysis_id}_metadata.json"
            if candidate.exists():
                metadata_jobs.append((candidate, directory))
                break
        if not metadata_jobs:
            raise FileNotFoundError(
                f"Metadata file not found for analysis id '{args.analysis_id}' "
                f"in directories: {', '.join(str(d[0]) for d in directories)}"
            )
    else:
        for directory, include_base, include_sim in directories:
            metadata_files = discover_metadata_files(
                directory, analysis_run_id, include_base, include_sim
            )
            metadata_jobs.extend((path, directory) for path in metadata_files)

    if not metadata_jobs:
        raise FileNotFoundError(
            f"No metadata files found for run id '{analysis_run_id}'. "
            "Ensure C1_dRSA_run.py (with or without --simulation) has produced outputs."
        )

    autocorr_entries: list[dict] = []
    for metadata_path, base_dir in metadata_jobs:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        run_id, saved_paths, autocorr_entry = plot_run_from_metadata(
            metadata_path,
            base_dir,
            custom_ylim=custom_ylim,
        )
        if len(saved_paths) == 1:
            print(f"✓ {run_id}: saved dRSA plot to {saved_paths[0]}")
        else:
            print(f"✓ {run_id}: saved {len(saved_paths)} dRSA plots:")
            for path in saved_paths:
                print(f"  - {path}")
        if autocorr_entry is not None:
            autocorr_entries.append(autocorr_entry)

    if autocorr_entries and not args.analysis_id:
        output_dirs = {Path(entry["output_dir"]) for entry in autocorr_entries}
        target_dir = next(iter(output_dirs))
        auto_plot_path = (
            target_dir / f"{analysis_run_id}_sim_autocorrelations_plot.png"
        )
        saved_path = generate_autocorr_summary_plot(autocorr_entries, auto_plot_path)
        print(f"✓ Simulation autocorrelations: saved to {saved_path}")


if __name__ == "__main__":
    main()
