from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from .core_functions import compute_lag_correlation, compute_rsa_matrix_corr, zscore

RegressionLogger = Callable[[str], None]


@dataclass(slots=True)
class RegressionConfig:
    """Hyperparameters shared across regression methods."""

    method: str = "elasticnet"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    variance_threshold: float = 0.85
    border_threshold: float = 0.1
    plot_borders: bool = True
    border_plot_dir: Path | None = None
    progress_iterations: int = 10
    neural_progress_step: int = 50
    border_thresholds: Sequence[float] | None = None


@dataclass(slots=True)
class RegressionResult:
    """Container for regression outputs tied to one neural signal."""

    betas: np.ndarray
    r2: np.ndarray
    stats: dict


def _normalize_method(method: str) -> str:
    allowed = {"pcr", "ridge", "lasso", "elasticnet"}
    method_lc = method.lower()
    if method_lc not in allowed:
        raise ValueError(f"Unknown regression method '{method}'. Expected one of {sorted(allowed)}.")
    return method_lc


def _slugify(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "model"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def regression_border(
    model_rdm_series_list: Sequence[np.ndarray],
    averaging_diagonal_time_tps: int,
    correlation_thresholds: Sequence[float] | float = 0.1,
    model_labels: Sequence[str] | None = None,
    plot: bool = False,
    plot_dir: Path | None = None,
    logger: RegressionLogger | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Estimate autocorrelation borders for each model RDM series and optionally plot them.
    """

    if plot and plot_dir is None:
        raise ValueError("plot_dir must be provided when plot=True.")
    if plot and plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    borders = np.zeros(len(model_rdm_series_list), dtype=int)
    plot_records: list[dict] = []
    for idx, model in enumerate(model_rdm_series_list):
        if isinstance(correlation_thresholds, Sequence) and not isinstance(
            correlation_thresholds, (str, bytes)
        ):
            threshold = float(correlation_thresholds[idx])
        else:
            threshold = float(correlation_thresholds)
        autocorr = compute_rsa_matrix_corr(model, model)
        lags, lag_corr = compute_lag_correlation(autocorr, averaging_diagonal_time_tps)
        if lag_corr.size == 0:
            borders[idx] = 0
            continue
        mask = np.abs(lag_corr) >= threshold
        if np.any(mask):
            max_lag = int(np.max(np.abs(lags[mask])))
        else:
            max_lag = 0
        borders[idx] = min(max_lag, averaging_diagonal_time_tps)

        if plot and plot_dir is not None:
            label = (
                model_labels[idx]
                if model_labels and idx < len(model_labels)
                else f"Model {idx}"
            )
            plot_path = plot_dir / f"regression_border_{_slugify(label)}.png"
            _plot_regression_border(
                lags,
                lag_corr,
                threshold,
                borders[idx],
                label,
                plot_path,
            )
            data_path = plot_path.with_suffix(".npz")
            np.savez(
                data_path,
                lags=lags,
                lag_corr=lag_corr,
                threshold=threshold,
                border_lag=borders[idx],
            )
            plot_records.append(
                {
                    "label": label,
                    "path": str(plot_path),
                    "border_lag": int(borders[idx]),
                    "threshold": threshold,
                    "data_path": str(data_path),
                }
            )
            if logger:
                logger(
                    f"[regression_border] saved plot for '{label}' at {plot_path}"
                )

    return borders, plot_records


def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "PCR",
    alpha: float | None = None,
    l1_ratio: float | None = None,
    variance_threshold: float = 0.85,
    border_mask: np.ndarray | None = None,
    pca: PCA | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Solve a regression problem for one neural/model pair of time points.
    """

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array (samples × predictors).")
    if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must be a 1D array with the same length as X rows.")

    if border_mask is not None:
        mask = np.asarray(border_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != X_arr.shape[1]:
            raise ValueError("border_mask must match number of predictors.")
        if not np.any(mask):
            raise ValueError("border_mask removed all predictors.")
        X_arr = X_arr[:, mask]

    X_scaled = zscore(X_arr, axis=0)
    y_scaled = zscore(y_arr)
    method_lc = _normalize_method(method)
    stats: dict = {}

    if method_lc == "pcr":
        pca = pca or PCA(svd_solver="full")
        scores = pca.fit_transform(X_scaled)
        explained = np.cumsum(pca.explained_variance_ratio_)
        max_components = min(scores.shape[1], scores.shape[0])
        cutoff = int(np.searchsorted(explained, variance_threshold, side="left") + 1)
        cutoff = max(1, min(cutoff, max_components))
        scores_reduced = scores[:, :cutoff]
        betas_pc, *_ = np.linalg.lstsq(scores_reduced, y_scaled, rcond=None)
        loadings = pca.components_[:cutoff, :]
        coef_full = loadings.T @ betas_pc
        y_hat = scores_reduced @ betas_pc
        stats["n_components"] = cutoff
    else:
        alpha_val = 1.0 if alpha is None else float(alpha)
        if method_lc == "ridge":
            estimator = Ridge(alpha=alpha_val, fit_intercept=False)
        elif method_lc == "lasso":
            estimator = Lasso(alpha=alpha_val, fit_intercept=False, max_iter=5000)
        else:
            l1 = 0.5 if l1_ratio is None else float(l1_ratio)
            estimator = ElasticNet(
                alpha=alpha_val,
                l1_ratio=l1,
                fit_intercept=False,
                max_iter=5000,
            )
            stats["l1_ratio"] = l1
        estimator.fit(X_scaled, y_scaled)
        coef_full = estimator.coef_
        y_hat = estimator.predict(X_scaled)

    residual = y_scaled - y_hat
    ss_res = float(np.dot(residual, residual))
    ss_tot = float(np.dot(y_scaled - y_scaled.mean(), y_scaled - y_scaled.mean()))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    stats["r2"] = r2
    return coef_full, stats


def compute_dRSA_regression(
    neural_rdm_series: np.ndarray,
    model_rdm_series_list: Sequence[np.ndarray],
    averaging_diagonal_time_tps: int,
    config: RegressionConfig,
    logger: RegressionLogger | None = None,
    model_labels: Sequence[str] | None = None,
) -> RegressionResult:
    """
    Compute cross-temporal dRSA matrices using regression-based methods.
    """

    method_lc = _normalize_method(config.method)
    neural = np.asarray(neural_rdm_series)
    if neural.ndim != 3:
        raise ValueError("neural_rdm_series must be (iterations, subsample_tps, rdm_len).")
    iterations, subsample_tps, rdm_len = neural.shape

    models = [np.asarray(model) for model in model_rdm_series_list]
    for idx, model in enumerate(models):
        if model.shape != neural.shape:
            raise ValueError(
                f"Model index {idx} shape {model.shape} does not match neural shape {neural.shape}."
            )

    n_models = len(models)
    window_radius = averaging_diagonal_time_tps
    max_window = 2 * window_radius + 1
    max_predictors = 1 + n_models * max_window
    design_matrix = np.empty((rdm_len, max_predictors), dtype=neural.dtype)
    pca = PCA(svd_solver="full") if method_lc == "pcr" else None

    threshold_values = (
        list(config.border_thresholds)
        if config.border_thresholds is not None
        else [config.border_threshold] * len(models)
    )
    exclusion_windows, border_plots = regression_border(
        models,
        averaging_diagonal_time_tps,
        threshold_values,
        model_labels=model_labels,
        plot=config.plot_borders and config.border_plot_dir is not None,
        plot_dir=config.border_plot_dir if config.plot_borders else None,
        logger=logger,
    )

    betas_sum = np.zeros((n_models, subsample_tps, subsample_tps), dtype=np.float64)
    r2_sum = np.zeros_like(betas_sum)
    pc_counts: list[int] = []

    iteration_log_interval = max(1, config.progress_iterations)
    start_time = time.perf_counter()

    total_steps = iterations * subsample_tps
    processed_steps = 0

    for iteration in range(iterations):
        models_iter = [model[iteration] for model in models]
        neural_iter = neural[iteration]
        iter_start = time.perf_counter()
        neural_step = max(1, config.neural_progress_step)
        for t_neural in range(subsample_tps):
            if logger and t_neural % neural_step == 0:
                logger(
                    f"[{config.method}] iteration {iteration + 1}/{iterations} "
                    f"starting neural {t_neural + 1}/{subsample_tps}"
                )
            y = zscore(neural_iter[t_neural])
            for t_model in range(subsample_tps):
                if abs(t_neural - t_model) > window_radius:
                    continue  # skip cells we never analyse
                start = max(0, t_model - window_radius)
                end = min(subsample_tps, t_model + window_radius + 1)
                window_indices = np.arange(start, end, dtype=int)
                for model_idx in range(n_models):
                    same_indices = _same_model_indices(
                        window_indices, t_model, exclusion_windows[model_idx]
                    )
                    block = _build_design_matrix(
                        design_matrix,
                        models_iter,
                        model_idx,
                        t_model,
                        window_indices,
                        same_indices,
                    )
                    coef, stats = run_regression(
                        block,
                        y,
                        method=method_lc,
                        alpha=config.alpha,
                        l1_ratio=config.l1_ratio,
                        variance_threshold=config.variance_threshold,
                        pca=pca,
                    )
                    betas_sum[model_idx, t_neural, t_model] += coef[0]
                    r2_sum[model_idx, t_neural, t_model] += stats["r2"]
                    if method_lc == "pcr":
                        pc_counts.append(int(stats["n_components"]))

            processed_steps += 1
            if logger and (
                (t_neural + 1) % neural_step == 0 or t_neural == subsample_tps - 1
            ):
                neural_elapsed = time.perf_counter() - iter_start
                eta_str = ""
                if processed_steps > 0 and total_steps:
                    elapsed_total = time.perf_counter() - start_time
                    remaining = total_steps - processed_steps
                    eta_seconds = elapsed_total / processed_steps * remaining
                    eta_str = f" | ETA {format_duration(eta_seconds)}"
                logger(
                    f"[{config.method}] iteration {iteration + 1}/{iterations} "
                    f"neural {t_neural + 1}/{subsample_tps} "
                    f"({neural_elapsed:.1f}s elapsed in iteration){eta_str}"
                )

        if logger and ((iteration + 1) % iteration_log_interval == 0 or iteration == iterations - 1):
            elapsed = time.perf_counter() - start_time
            logger(
                f"[{config.method}] iteration {iteration + 1}/{iterations} "
                f"complete ({elapsed:.1f}s elapsed)."
            )

    betas = (betas_sum / iterations).astype(np.float32, copy=False)
    r2 = (r2_sum / iterations).astype(np.float32, copy=False)

    stats: dict = {
        "method": config.method,
        "iterations": iterations,
        "subsample_tps": subsample_tps,
        "rdm_length": rdm_len,
        "alpha": config.alpha,
        "l1_ratio": config.l1_ratio,
        "variance_threshold": config.variance_threshold,
        "border_threshold": config.border_threshold,
        "border_thresholds": threshold_values,
        "exclusion_windows": exclusion_windows.tolist(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "border_plots": border_plots if border_plots else None,
    }
    if pc_counts:
        stats["pca_components"] = {
            "min": int(np.min(pc_counts)),
            "median": float(np.median(pc_counts)),
            "max": int(np.max(pc_counts)),
        }

    return RegressionResult(betas=betas, r2=r2, stats=stats)


def _build_design_matrix(
    design_matrix: np.ndarray,
    model_iter_list: Sequence[np.ndarray],
    main_model_index: int,
    t_model: int,
    window_indices: np.ndarray,
    same_model_indices: np.ndarray,
) -> np.ndarray:
    """Populate ``design_matrix`` and return the active slice."""

    col = 0
    design_matrix[:, col] = model_iter_list[main_model_index][t_model]
    col += 1

    for idx, model in enumerate(model_iter_list):
        if idx == main_model_index:
            continue
        block = model[window_indices]
        width = block.shape[0]
        if width == 0:
            continue
        design_matrix[:, col : col + width] = block.T
        col += width

    same_block = model_iter_list[main_model_index][same_model_indices]
    width = same_block.shape[0]
    if width:
        design_matrix[:, col : col + width] = same_block.T
        col += width

    return design_matrix[:, :col]


def _same_model_indices(
    window_indices: np.ndarray, center_index: int, exclusion_radius: int
) -> np.ndarray:
    """Select same-model indices within the window but outside the exclusion radius."""

    if exclusion_radius < 0:
        return window_indices
    mask = np.abs(window_indices - center_index) > exclusion_radius
    return window_indices[mask]


def _plot_regression_border(
    lags: np.ndarray,
    lag_corr: np.ndarray,
    threshold: float,
    border_lag: int,
    label: str,
    path: Path,
) -> None:
    """Save a PNG showing the autocorrelation curve with the derived border."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lags, lag_corr, color="tab:blue", linewidth=2, label="Autocorrelation")
    ax.axhline(threshold, color="tab:orange", linestyle="--", label="Threshold")
    ax.axvline(border_lag, color="tab:green", linestyle=":", label="+border")
    ax.axvline(-border_lag, color="tab:red", linestyle=":", label="-border")
    ax.set_title(f"Regression Border | {label}")
    ax.set_xlabel("Lag (time points)")
    ax.set_ylabel("Mean correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _run_smoke_test() -> None:
    """Simple sanity check covering all regression methods."""

    rng = np.random.default_rng(0)
    iterations, subsample_tps, rdm_len = 3, 5, 10
    neural = rng.standard_normal((iterations, subsample_tps, rdm_len))
    models = [
        neural + rng.standard_normal((iterations, subsample_tps, rdm_len)) * 0.1,
        rng.standard_normal((iterations, subsample_tps, rdm_len)),
    ]
    methods = ("pcr", "ridge", "lasso", "elasticnet")
    for method in methods:
        config = RegressionConfig(method=method)
        result = compute_dRSA_regression(
            neural,
            models,
            averaging_diagonal_time_tps=2,
            config=config,
        )
        desc = (
            f"{method:>10} | betas {result.betas.shape} | "
            f"R² mean {np.mean(result.r2):.3f}"
        )
        print(desc)


if __name__ == "__main__":
    _run_smoke_test()
