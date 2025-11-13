"""Legacy PCR helpers kept for backward compatibility."""

from __future__ import annotations

import warnings

from .regression_methods import (
    RegressionConfig,
    RegressionResult,
    compute_dRSA_regression,
    regression_border,
    run_regression,
)

__all__ = [
    "compute_rsa_matrix_PCR",
    "run_regression",
    "regression_border",
]


def compute_rsa_matrix_PCR(
    neural_rdm_series,
    model_rdm_series_list,
    averaging_diagonal_time_tps=300,
    variance_threshold=0.85,
    client=None,
):
    """
    Deprecated shim that now delegates to :mod:`functions.regression_methods`.
    """

    if client is not None:
        warnings.warn(
            "The 'client' argument is unused in the Python implementation.",
            RuntimeWarning,
            stacklevel=2,
        )

    warnings.warn(
        "functions.PCR_alpha.compute_rsa_matrix_PCR is deprecated; "
        "please import functions.regression_methods instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    config = RegressionConfig(method="pcr", variance_threshold=variance_threshold)
    result: RegressionResult = compute_dRSA_regression(
        neural_rdm_series,
        model_rdm_series_list,
        averaging_diagonal_time_tps,
        config,
        logger=None,
    )
    return [result.betas[idx] for idx in range(result.betas.shape[0])]
