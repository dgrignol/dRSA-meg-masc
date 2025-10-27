import argparse
import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from functions.core_functions import compute_lag_correlation


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
):
    adtw_in_tps = analysis_parameters["averaging_window_tps"]
    resolution = analysis_parameters["resolution_hz"]

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

        parameter_caption = ", ".join(
            f"{key}={value}" for key, value in analysis_parameters.items()
        )
        parameter_caption = "Parameters: " + parameter_caption
        parameter_caption = textwrap.fill(parameter_caption, width=110)

        fig.suptitle(f"{neural_label} dRSA summary", fontsize=14, fontweight="bold")
        fig.text(0.5, 0.02, parameter_caption, ha="center", va="center", fontsize=8)

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
        "--results-dir",
        default="results",
        help="Directory containing cached dRSA outputs (default: results).",
    )
    args = parser.parse_args()

    if args.subject.startswith("sub-"):
        subject_label = args.subject
    else:
        subject_label = f"sub-{int(args.subject):02d}"

    if args.analysis_id:
        analysis_run_id = args.analysis_id
    else:
        analysis_run_id = build_analysis_run_id(
            subject_label,
            args.resolution,
            args.rsa_method,
        )

    results_dir = args.results_dir
    metadata_path = os.path.join(results_dir, f"{analysis_run_id}_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    outputs = metadata.get("outputs", {})
    rsa_matrices_path = outputs.get("rsa_matrices") or os.path.join(
        results_dir, f"{analysis_run_id}_dRSA_matrices.npy"
    )
    if not os.path.exists(rsa_matrices_path):
        raise FileNotFoundError(f"RSA matrices file not found: {rsa_matrices_path}")
    rsa_matrices = np.load(rsa_matrices_path)

    lag_curves_path = outputs.get("lag_curves")
    lag_curves_array = None
    if lag_curves_path and os.path.exists(lag_curves_path):
        lag_curves_array = np.load(lag_curves_path)
    else:
        if lag_curves_path:
            print(
                f"! lag curves file referenced in metadata but missing: {lag_curves_path}"
            )
        else:
            print("! lag curves not available; plotting without confidence intervals.")

    plot_path = (
        outputs.get("plot_target")
        or outputs.get("plot")
        or os.path.join(results_dir, f"{analysis_run_id}_plot.png")
    )

    model_labels = metadata["selected_model_labels"]
    neural_labels = metadata["neural_signal_labels"]
    analysis_parameters = metadata["analysis_parameters"]
    lag_bootstrap_settings = metadata.get("lag_bootstrap_settings", {})

    saved_paths = generate_plots(
        rsa_matrices,
        lag_curves_array,
        model_labels,
        neural_labels,
        analysis_parameters,
        lag_bootstrap_settings,
        plot_path,
    )
    if len(saved_paths) == 1:
        print(f"✓ Saved dRSA plot to {saved_paths[0]}")
    else:
        print("✓ Saved dRSA plots:")
        for path in saved_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
