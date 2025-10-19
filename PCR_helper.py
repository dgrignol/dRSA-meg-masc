import numpy as np
from sklearn.decomposition import PCA
from core_functions_v7 import zscore, compute_rsa_matrix_corr, compute_lag_correlation

def regression_border(model_rdm_series_list, averaging_diagonal_time_tps, correlation_threshold=0.1):
    '''
    this is to decide what timepoints to exclude of the same model, from regressing out.

    correlation_threshold: the threshold for the lag correlation to be considered significant
    '''

    exclusion_window_for_models = []
    for model in model_rdm_series_list:
        drsa_corr = compute_rsa_matrix_corr(model, model)
        lag, lag_corr = compute_lag_correlation(drsa_corr, averaging_diagonal_time_tps)

        # max of the lag where the lag_corr is above correlation_threshold

        if np.any(lag_corr > correlation_threshold):
            max_lag = int(np.max(lag[lag_corr > correlation_threshold]))
        else:
            max_lag = 0

        max_lag = min(max_lag, averaging_diagonal_time_tps) # ensure it does not exceed the averaging diagonal time window
        exclusion_window_for_models.append(max_lag)

    return exclusion_window_for_models


def compute_rsa_matrix_PCR(neural_rdm_series, model_rdm_series_list,
                           averaging_diagonal_time_tps=300,
                           variance_threshold=0.85,
                           client=None):

    iterations, subsample_tps, rdm_len = neural_rdm_series.shape
    n_models = len(model_rdm_series_list)
    num_other_models = n_models - 1

    # pre-allocations
    rsa_holder_per_iteration_per_model = np.zeros((n_models, iterations, subsample_tps, subsample_tps))
    max_predictors = 1 + n_models * (2 * averaging_diagonal_time_tps + 1)
    X = np.empty((rdm_len, max_predictors), dtype=neural_rdm_series.dtype)  # pre-allocated

    # PCA initialization
    pca = PCA(svd_solver='full')

    # compute exclusion windows once
    exclusion_windows = regression_border(model_rdm_series_list, averaging_diagonal_time_tps)

    for i in range(iterations):
        for t_neural in range(subsample_tps):
            print(f"  [iter {i}] computing t_neural {t_neural}/{subsample_tps}", flush=True)
            
            y = neural_rdm_series[i, t_neural]  # target vector
            y = zscore(y)
            result = np.zeros((n_models, subsample_tps), dtype=neural_rdm_series.dtype)  # local buffer allocation

            for t_model in range(subsample_tps):
                start = max(0, t_model - averaging_diagonal_time_tps)
                end = min(subsample_tps, t_model + averaging_diagonal_time_tps + 1)
                window_len = end - start

                for m_interest in range(n_models):
                    main = model_rdm_series_list[m_interest][i, t_model]

                    # build mask for same-model exclusion
                    mask = np.zeros(subsample_tps, dtype=bool)
                    mask[start:end] = True
                    exc = exclusion_windows[m_interest]
                    mask[max(0, t_model - exc):min(subsample_tps, t_model + exc + 1)] = False

                    # start filling design matrix X
                    X[:, 0] = main
                    col = 1

                    # other models block
                    for m in range(n_models):
                        if m == m_interest:
                            continue
                        block = model_rdm_series_list[m][i, start:end]  # shape (window_len, rdm_len)
                        X[:, col:col+window_len] = block.T
                        col += window_len

                    # same-model block
                    sm = model_rdm_series_list[m_interest][i, mask]  # shape (k, rdm_len)
                    k = sm.shape[0]
                    X[:, col:col+k] = sm.T
                    n_predictors = 1 + (num_other_models * window_len) + k

                    # zscore and PCA
                    X_scaled = zscore(X[:, :n_predictors])
                    X_pca = pca.fit_transform(X_scaled) # get PCA scores (each time point rdm in the new PC space)
                    explained = np.cumsum(pca.explained_variance_ratio_)
                    n_components = int(np.searchsorted(explained, variance_threshold) + 1)
                    X_pca_reduced = X_pca[:, :n_components]
                    loadings_reduced = pca.components_[:n_components, :]

                    # regression
                    betas, *_ = np.linalg.lstsq(X_pca_reduced, y, rcond=None)
                    beta_model_space = loadings_reduced.T @ betas
                    result[m_interest, t_model] = beta_model_space[0]

            # write local buffer to main holder
            rsa_holder_per_iteration_per_model[:, i, t_neural, :] = result

    rsa_matrices = [np.mean(rsa_holder_per_iteration_per_model[m], axis=0) for m in range(n_models)]
    return rsa_matrices


''' 

more efficient like the dask scatter version

'''