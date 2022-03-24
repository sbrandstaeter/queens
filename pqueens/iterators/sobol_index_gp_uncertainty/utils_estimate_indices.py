"""Utils for Sobol index estimation.

All functions below are independent functions so that they can be used for parallel computations
with multiprocessing.

Important: Do not use XArrays in parallel processes as they are very slow!
"""
import numpy as np


def bootstrap(prediction, bootstrap_indices):
    """Bootstrap samples.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_indices (ndarray): bootstrap indices

    Returns:
        current_bootstrap_sample (ndarray): current bootstrap sample
    """
    current_bootstrap_sample = prediction[bootstrap_indices]
    return current_bootstrap_sample


def calculate_indices_first_total_order(
    prediction, bootstrap_idx, input_dim, number_bootstrap_samples, first_order_estimator
):
    """Estimate first and total-order Sobol indices.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_idx (ndarray): bootstrap indices
        input_dim (int): input-parameter index
        number_bootstrap_samples (int): number of bootstrap samples
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_first_order (ndarray): estimates of first-order Sobol index
        estimates_total_order (ndarray): estimates of total-order Sobol index
    """
    estimates_first_order = np.empty(number_bootstrap_samples)
    estimates_total_order = np.empty(number_bootstrap_samples)

    bootstrap_samples = bootstrap(prediction, bootstrap_idx)
    for idx_bootstrap in np.arange(number_bootstrap_samples):
        current_bootstrap_sample = bootstrap_samples[idx_bootstrap, :]
        A, B, AB = _extract_sample_matrices(current_bootstrap_sample, input_dim)
        estimates_first_order[idx_bootstrap] = _estimate_first_order_index(
            A, B, AB, first_order_estimator
        )
        estimates_total_order[idx_bootstrap] = _estimate_total_order_index(A, B, AB)

    return estimates_first_order, estimates_total_order


def calculate_indices_second_order_gp_realizations(
    prediction,
    bootstrap_indices,
    input_dim_i,
    number_bootstrap_samples,
    number_parameters,
    first_order_estimator,
):
    """Estimate first, second and total-order Sobol indices.

    Based on Gaussian process realizations and parallelized over those realizations.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_indices (ndarray): bootstrap indices
        input_dim_i (int): input-parameter index
        number_bootstrap_samples (int): number of bootstrap samples
        number_parameters (int): number of input-space dimensions
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_first_order (ndarray): estimates of first-order Sobol index
        estimates_second_order (ndarray): estimates of second-order Sobol index
        estimates_total_order (ndarray): estimates of total-order Sobol index
    """
    estimates_first_order, estimates_total_order = calculate_indices_first_total_order(
        prediction, bootstrap_indices, input_dim_i, number_bootstrap_samples, first_order_estimator
    )

    estimates_second_order = np.empty(
        (number_bootstrap_samples, number_parameters - (input_dim_i + 1))
    )
    bootstrap_samples = bootstrap(prediction, bootstrap_indices)
    for b in np.arange(number_bootstrap_samples):
        current_bootstrap_sample = bootstrap_samples[b, :]
        idx_loop = 0
        for input_dim_j in range(input_dim_i + 1, number_parameters):
            A, B, ABi, ABj, BAi = _extract_sample_matrices_second_order(
                current_bootstrap_sample, input_dim_i, input_dim_j, number_parameters
            )
            estimates_second_order[b, idx_loop] = _estimate_second_order_index(
                A, ABi, ABj, BAi, B, first_order_estimator
            )
            idx_loop += 1

    return estimates_first_order, estimates_total_order, estimates_second_order


def calculate_indices_second_order_gp_mean(
    prediction, bootstrap_indices, input_dim_i, number_parameters, first_order_estimator
):
    """Estimate first, second and total-order Sobol indices.

    Based on Gaussian process mean and parallelized over bootstrapping samples.

    Args:
        prediction (ndarray): prediction
        bootstrap_indices (ndarray): bootstrap indices
        input_dim_i (int): input-parameter index
        number_parameters (int): number of input-space dimensions
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_first_order (ndarray): estimates of first-order Sobol index
        estimates_second_order (ndarray): estimates of second-order Sobol index
        estimates_total_order (ndarray): estimates of total-order Sobol index
    """
    bootstrap_sample = bootstrap(prediction, bootstrap_indices)
    A, B, AB = _extract_sample_matrices(bootstrap_sample, input_dim_i)
    estimates_first_order = _estimate_first_order_index(A, B, AB, first_order_estimator)
    estimates_total_order = _estimate_total_order_index(A, B, AB)

    estimates_second_order = np.empty(number_parameters - (input_dim_i + 1))
    idx_loop = 0
    for input_dim_j in range(input_dim_i + 1, number_parameters):
        A, B, ABi, ABj, BAi = _extract_sample_matrices_second_order(
            bootstrap_sample, input_dim_i, input_dim_j, number_parameters
        )
        estimates_second_order[idx_loop] = _estimate_second_order_index(
            A, ABi, ABj, BAi, B, first_order_estimator
        )
        idx_loop += 1

    return estimates_first_order, estimates_total_order, estimates_second_order


def calculate_indices_third_order(
    prediction,
    bootstrap_indices,
    number_boostrap_samples,
    number_parameters,
    first_order_estimator,
):
    """Estimate third-order Sobol indices.

    Based on Gaussian process realizations and parallelized over those realizations.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_indices (ndarray): bootstrap indices
        number_boostrap_samples (int): number of bootstrap samples
        number_parameters (int): number of input-space dimensions
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_third_order (ndarray): estimates for third-order Sobol index
    """
    # 1. Estimate first-order Sobol indices
    estimates_first_order = np.empty((number_boostrap_samples, number_parameters))
    estimates_total_order = np.empty((number_boostrap_samples, number_parameters))
    for i in range(number_parameters):
        (
            estimates_first_order[:, i],
            estimates_total_order[:, i],
        ) = calculate_indices_first_total_order(
            prediction, bootstrap_indices, i, number_boostrap_samples, first_order_estimator
        )

    # 2. Estimate second-order Sobol indices
    estimates_second_order = np.empty((number_boostrap_samples, number_parameters))
    bootstrap_samples = bootstrap(prediction, bootstrap_indices)
    for b in np.arange(number_boostrap_samples):
        current_bootstrap_sample = bootstrap_samples[b, :]
        idx_loop = 0
        for idx_i, i in enumerate(range(number_parameters)):
            for j in range(idx_i + 1, number_parameters):
                A, B, ABi, ABj, BAi = _extract_sample_matrices_second_order(
                    current_bootstrap_sample, i, j, number_parameters
                )
                estimates_second_order[b, idx_loop] = _estimate_second_order_index(
                    A, ABi, ABj, BAi, B, first_order_estimator
                )
                idx_loop += 1

    # 3. Estimate closed third-order Sobol indices (includes lower order indices)
    closed_estimates_third_order = np.empty((number_boostrap_samples, 1))
    for b in np.arange(number_boostrap_samples):
        current_bootstrap_sample = bootstrap_samples[b, :]
        A, B, AB_ijk = _extract_sample_matrices_third_order(current_bootstrap_sample)
        closed_estimates_third_order[b, 0] = _estimate_closed_third_order_index(B, AB_ijk)

    # 4. Subtract lower order indices
    estimates_third_order = (
        closed_estimates_third_order.flatten()
        - estimates_second_order.sum(axis=1)
        - estimates_first_order.sum(axis=1)
    )

    return estimates_third_order


def _extract_sample_matrices(prediction, i):
    """Extract sampling matrices A, B, AB from Gaussian process prediction.

    Format:
        [AB_1, AB_2, ..., AB_D, BA_1, ..., BA_D, A, B]

    Args:
        prediction (ndarray): realizations of Gaussian process
        i (int): index to calculate

    Returns:
        A (ndarray): Saltelli A sample matrix
        B (ndarray): Saltelli B sample matrix
        AB (ndarray): Saltelli AB sample matrix
    """
    AB = prediction[:, i]
    A = prediction[:, -2]
    B = prediction[:, -1]

    return A, B, AB


def _extract_sample_matrices_second_order(prediction, i, j, number_parameters):
    """Extract sampling matrices A, B, AB from Gaussian process prediction.

    Format:
        [AB_1, AB_2, ..., AB_D, BA_1, ..., BA_D, A, B]

    Args:
        prediction (ndarray): realizations of Gaussian process
        i (int): index to calculate
        j (int): index to calculate
        number_parameters (int): number of input space dimensions

    Returns:
        A (ndarray): Saltelli A sample matrix
        B (ndarray): Saltelli B sample matrix
        AB (ndarray): Saltelli AB sample matrix
        BA (ndarray): Saltelli AB sample matrix
    """
    ABi = prediction[:, i]
    ABj = prediction[:, j]
    BAi = prediction[:, number_parameters + i]
    A = prediction[:, -2]
    B = prediction[:, -1]

    return A, B, ABi, ABj, BAi


def _extract_sample_matrices_third_order(prediction):
    """Extract sampling matrices A, B, AB from Gaussian process prediction.

    Format:
        [AB_1, AB_2, ..., AB_D, BA_1, ..., BA_D, AB_123, A, B]

    Args:
        prediction (ndarray): realizations of Gaussian process

    Returns:
        A (ndarray): Saltelli A sample matrix
        B (ndarray): Saltelli B sample matrix
        AB_ijk (ndarray): Saltelli AB_ijk sample matrix
    """
    AB_ijk = prediction[:, -3]
    A = prediction[:, -2]
    B = prediction[:, -1]

    return A, B, AB_ijk


def _estimate_first_order_index(A, B, AB, estimator):
    """Compute first-order Sobol indices.

    References for estimators:

    [Janon2014] Janon, Alexandre, Thierry Klein, Agnès Lagnoux, Maëlle Nodet, and Clémentine
    Prieur. ‘Asymptotic Normality and Efficiency of Two Sobol Index Estimators’. ESAIM: Probability
    and Statistics 18 (2014): 342–64. https://doi.org/10.1051/ps/2013040.

    [Gratiet2014] Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach
    for Global Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on
    Uncertainty Quantification 2, no. 1 (1 January 2014): 336–63. https://doi.org/10.1137/130926869.

    [Saltelli2010] Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design
    and Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

    Args:
        A (ndarray): results corresponding to A sample matrix
        B (ndarray): results corresponding B sample matrix
        AB (ndarray): results corresponding to AB sample matrix
        estimator (str): estimator for first-order indices

    Returns:
        first_order (ndarray): first-order Sobol index estimates
    """
    if estimator == 'Janon2014':
        # [Janon2014] Equation (2.5)
        first_order = (np.mean(B * AB) - (0.5 * np.mean(B + AB)) ** 2) / (
            np.mean(B * B) - (0.5 * np.mean(B + AB)) ** 2
        )

    elif estimator == 'Janon2014alt':
        # [Janon2014] Equation (2.8)
        first_order = np.sum(
            (B - 0.5 * (B.mean() + AB.mean())) * (AB - 0.5 * (B.mean() + AB.mean()))
        ) / np.sum(0.5 * (B ** 2 + AB ** 2) - (0.5 * (B.mean() + AB.mean())) ** 2)

    elif estimator == 'Gratiet2014':
        # [Gratiet2014] Equation (4.1)
        first_order = (np.mean(B * AB) - B.mean() * AB.mean()) / (np.mean(B * B) - B.mean() ** 2)

    elif estimator == 'Saltelli2010':
        # [Saltelli2010] also used in SALib library
        first_order = np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)

    else:
        raise ValueError(
            "Unknown first-order estimator. Valid estimators are Janon2014, "
            "Janon2014alt, Gratiet2014 or Saltelli2010."
        )

    return first_order


def _estimate_total_order_index(A, B, AB):
    """Estimate total-order Sobol indices.

    Reference:
    [Saltelli2010] Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design
    and Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

    Args:
        A (ndarray): results corresponding to A sample matrix
        B (ndarray): results corresponding to B sample matrix
        AB (ndarray): results corresponding to AB sample matrix

    Returns:
        total_order (ndarray): total-order Sobol index estimates
    """
    # from SALib library [Saltelli2010]
    total_order = 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)

    return total_order


def _estimate_second_order_index(A, ABj, ABk, BAj, B, first_order_estimator):
    """Estimate second-order Sobol indices.

    [Saltelli2010] Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design
    and Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

    Args:
        A (ndarray): results corresponding to A sample matrix
        ABj (ndarray): results corresponding to AB sample matrix
        ABk (ndarray): results corresponding to AB sample matrix
        BAj (ndarray): results corresponding to BA sample matrix
        B (ndarray): results corresponding to B sample matrix
        first_order_estimator (str): estimator for first-order indices

    Returns:
        second_order (ndarray): second-order Sobol index estimates
    """
    # from SALib library [Saltelli2010]
    Sjk = np.mean(BAj * ABk - A * B, axis=0) / np.var(np.r_[A, B], axis=0)
    Sj = _estimate_first_order_index(A, B, ABj, first_order_estimator)
    Sk = _estimate_first_order_index(A, B, ABk, first_order_estimator)
    second_order = Sjk - Sj - Sk

    return second_order


def _estimate_closed_third_order_index(B, AB_ijk):
    """Estimate closed third-order Sobol indices.

    Estimator based on Equation (4.1) in

    Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach for Global
    Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on Uncertainty
    Quantification 2, no. 1 (1 January 2014): 336–63.
    https://doi.org/10.1137/130926869.

    Args:
        B (ndarray): results corresponding to B
        AB_ijk (ndarray): results corresponding to AB_ijk

    Returns:
        second_order (ndarray): second-order Sobol index estimates
    """
    closed_third_order = (np.mean(B * AB_ijk) - B.mean() * AB_ijk.mean()) / (
        np.mean(B * B) - B.mean() ** 2
    )

    return closed_third_order
