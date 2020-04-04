import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from pqueens.utils.plot_outputs import plot_pdf
from pqueens.utils.plot_outputs import plot_cdf
from pqueens.utils.plot_outputs import plot_failprob
from pqueens.utils.plot_outputs import plot_icdf


def process_ouputs(output_data, output_description, input_data=None):
    """ Process output from QUEENS models

    Args:
        output_data (dict):         Dictionary containing model output
        output_descripion (dict):   Dictionary describing desired output quantities
        input_data (np.array):      Array containing model input

    Returns:
        dict:                       Dictionary with processed results
    """
    processed_results = {}
    try:
        processed_results = do_processing(output_data, output_description)
    except:
        print("Could not process results properly")

    # add the actual raw input and output data
    processed_results["raw_output_data"] = output_data
    if input_data is not None:
        processed_results["input_data"] = input_data

    return processed_results


def do_processing(output_data, output_description):
    """ Do actual processing of output

    Args:
        output_data (dict):         Dictionary containing model output
        output_descripion (dict):   Dictionary describing desired output quantities

    Returns:
        dict:                       Dictionary with processed results
    """
    # do we want confidence intervals
    bayesian = output_description.get('bayesian', False)
    # check if we have the data to support this
    if "post_samples" not in output_data and bayesian is True:
        warnings.warn(
            "Warning: Output data does not contain posterior samples. "
            "Not computing confidence intervals"
        )
        bayesian = False

    # do we want plotting
    plot_results = output_description.get('plot_results', False)

    # result interval
    result_interval = output_description.get('result_interval', None)
    # TODO: we get an error below!
    output_data["mean"] = output_data["mean"].astype(np.float)
    if result_interval is None:
        # estimate interval from results
        result_interval = estimate_result_interval(output_data)

    # get number of support support points
    num_support_points = output_description.get('num_support_points', 100)
    support_points = np.linspace(result_interval[0], result_interval[1], num_support_points)

    mean_mean = estimate_mean(output_data)
    var_mean = estimate_var(output_data)

    processed_results = {}
    processed_results["mean"] = mean_mean
    processed_results["var"] = var_mean

    if output_description.get('cov', False):
        cov_mean = estimate_cov(output_data)
        processed_results["cov"] = cov_mean

    # do we want to estimate all the below (i.e. pdf, cdf, icdf)
    est_all = output_description.get('estimate_all', False)

    # do we want pdf estimation
    est_pdf = output_description.get('estimate_pdf', False)
    if (est_pdf or est_all) is True:
        pdf_estimate = estimate_pdf(output_data, support_points, bayesian)
        if plot_results is True:
            plot_pdf(pdf_estimate, support_points, bayesian)

        processed_results["pdf_estimate"] = pdf_estimate

    # do we want cdf estimation
    est_cdf = output_description.get('estimate_cdf', False)
    if (est_cdf or est_all) is True:
        cdf_estimate = estimate_cdf(output_data, support_points, bayesian)
        if plot_results is True:
            plot_cdf(cdf_estimate, support_points, bayesian)

        processed_results["cdf_estimate"] = cdf_estimate

    # do we want icdf estimation
    est_icdf = output_description.get('estimate_icdf', False)
    if (est_icdf or est_all) is True:
        icdf_estimate = estimate_icdf(output_data, bayesian)

        if plot_results is True:
            plot_icdf(icdf_estimate, bayesian)

        processed_results["icdf_estimate"] = icdf_estimate

    return processed_results


def write_results(processed_results, path_to_file, file_name):
    """ Write results to pickle file

    Args:
        processed_results (dict):  Dictionary with results
        path_to_file (str):        Path to write resutls to
        file_name (str):           Name of resutl file

    """

    pickle_file = pathlib.Path(path_to_file, file_name + ".pickle")

    with open(pickle_file, 'wb') as handle:
        pickle.dump(processed_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def estimate_result_interval(output_data):
    """ Estimate interval of output data

    Estimate interval of output data and add small margins

    Args:
        output_data (dict):       Dictionary with output data

    Returns:
        list:                     Output interval

    """
    samples = output_data["mean"]
    print(samples)
    min_data = np.amin(samples)
    print(min_data)
    max_data = np.amax(samples)

    interval_length = max_data - min_data
    my_min = min_data - interval_length / 6
    my_max = max_data + interval_length / 6

    return [my_min, my_max]


def estimate_mean(output_data):
    """ Estimate mean based on standard unbiased estimator

    Args:
        output_data (dict):       Dictionary with output data

    Returns:
        float                     Unbiased mean estimate

    """
    samples = output_data["mean"]
    return np.mean(samples, axis=0)


def estimate_var(output_data):
    """ Estimate variance based on standard unbiased estimator

    Args:
        output_data (dict):       Dictionary with output data

    Returns:
        float                     Unbiased variance estimate

    """
    samples = output_data["mean"]
    return np.var(samples, ddof=1, axis=0)


def estimate_cov(output_data):
    """ Estimate covariance based on standard unbiased estimator

    Args:
        output_data (dict):       Dictionary with output data

    Returns:
        numpy.array                    Unbiased covariance estimate

    """
    samples = output_data["mean"]

    # we assume that rows represent observations and columns represent variables
    row_variable = False

    cov = np.zeros((samples.shape[1], samples.shape[2], samples.shape[2]))
    for i in range(samples.shape[1]):
        cov[i] = np.cov(samples[:, i, :], rowvar=row_variable)
    return cov


def estimate_cdf(output_data, support_points, bayesian):
    """ Compute estimate of CDF based on provided sampling data

    Args:
        output_data (dict):         Dictionary with output data
        support_points (np.array):  Points where to evaluate cdf
        bayesian (bool):            Compute confindence intervals etc.

    Returns:
        cdf:                        Dictionary with cdf estimates

    """

    cdf = {}
    cdf["x"] = support_points
    if bayesian is False:
        raw_data = output_data["mean"]
        size_data = raw_data.size
        cdf_values = []
        for i in support_points:
            # all the values in raw data less than the ith value in x_values
            temp = raw_data[raw_data <= i]
            # fraction of that value with respect to the size of the x_values
            value = temp.size / size_data
            cdf_values.append(value)
        cdf["mean"] = cdf_values
    else:
        raw_data = output_data["post_samples"]
        size_data = len(support_points)
        num_realizations = raw_data.shape[1]
        cdf_values = np.zeros((num_realizations, len(support_points)))
        for i in range(num_realizations):
            data = raw_data[:, i]
            for j, point in enumerate(support_points):
                # all the values in raw data less than the ith value in x_values
                temp = data[data <= point]
                # fraction of that value with respect to the size of the x_values
                value = temp.size / size_data
                cdf_values[i, j] = value

        cdf["post_samples"] = cdf_values
        # now we compute mean, median cumulative distribution function
        cdf["mean"] = np.mean(cdf_values, axis=0)
        cdf["median"] = np.median(cdf_values, axis=0)
        cdf["q5"] = np.percentile(cdf_values, 5, axis=0)
        cdf["q95"] = np.percentile(cdf_values, 95, axis=0)

    return cdf


def estimate_icdf(output_data, bayesian):
    """ Compute estimate of inverse CDF based on provided sampling data

    Args:
        output_data (dict):         Dictionary with output data
        bayesian (bool):            Compute confindence intervals etc.

    Returns:
        icdf:                        Dictionary with icdf estimates

    """
    my_percentiles = 100 * np.linspace(0 + 1 / 1000, 1 - 1 / 1000, 999)
    icdf = {}
    icdf["x"] = my_percentiles
    if bayesian is False:
        samples = output_data["mean"]
        icdf_values = np.zeros_like(my_percentiles)
        for i, percentile in enumerate(my_percentiles):
            icdf_values[i] = np.percentile(samples, percentile, axis=0)
        icdf["mean"] = icdf_values
    else:
        raw_data = output_data["post_samples"]
        num_realizations = raw_data.shape[1]
        icdf_values = np.zeros((len(my_percentiles), num_realizations))
        for i, point in enumerate(my_percentiles):
            icdf_values[i, :] = np.percentile(raw_data, point, axis=0)

        icdf["post_samples"] = icdf_values
        # now we compute mean, median cumulative distribution function
        icdf["mean"] = np.mean(icdf_values, axis=1)
        icdf["median"] = np.median(icdf_values, axis=1)
        icdf["q5"] = np.percentile(icdf_values, 5, axis=1)
        icdf["q95"] = np.percentile(icdf_values, 95, axis=1)

    return icdf


def estimate_failprob(output_data, failure_thesholds, bayesian):
    """ Compute estimate of failure probability plot based on passed data

    Args:
        output_data (dict):            Dictionary with output data
        failure_thesholds (np.array):  Failure thresholds for which to evaluate
                                       failure probability
        bayesian (bool):               Compute confindence intervals etc.

    Returns:
        icdf:                        Dictionary with icdf estimates

    """
    raise NotImplementedError


def estimate_pdf(output_data, support_points, bayesian):
    """ Compute estimate of PDF based on provided sampling data

    Args:
        output_data (dict):         Dictionary with output data
        support_points (np.array):  Points where to evaluate pdf
        bayesian (bool):            Compute confindence intervals etc.

    Returns:
        pdf:                        Dictionary with pdf estimates

    """
    pdf = {}
    pdf["x"] = support_points
    if bayesian is False:
        samples = output_data["mean"]
        min_samples = np.amin(samples)
        max_samples = np.amax(samples)
        bandwidth = estimate_bandwidth_for_kde(samples, min_samples, max_samples)
        pdf["mean"] = perform_kde(samples, bandwidth, support_points)
    else:
        min_samples = np.amin(support_points)
        max_samples = np.amax(support_points)
        mean_samples = output_data["mean"]
        # estimate kernel bandwidth only once
        bandwidth = estimate_bandwidth_for_kde(mean_samples, min_samples, max_samples)
        raw_data = output_data["post_samples"]
        num_realizations = raw_data.shape[1]
        pdf_values = np.zeros((num_realizations, len(support_points)))
        for i in range(num_realizations):
            data = raw_data[:, i]
            pdf_values[i, :] = perform_kde(data, bandwidth, support_points)

        pdf["post_samples"] = pdf_values
        # now we compute mean, median probability density function
        pdf["mean"] = np.mean(pdf_values, axis=0)
        pdf["median"] = np.median(pdf_values, axis=0)
        pdf["q5"] = np.percentile(pdf_values, 5, axis=0)
        pdf["q95"] = np.percentile(pdf_values, 95, axis=0)

    return pdf


def estimate_bandwidth_for_kde(samples, min_samples, max_samples):
    """ Estimate optimal bandwidth for kde of pdf

    Args:
        samples (np.array):  samples for which to estimate pdf
        min_samples (float): smallest value
        max_samples (float): largest value
    Returns:
        float: estimate for optimal kernel_bandwidth
    """
    kernel_bandwidth = 0
    kernel_bandwidth_upper_bound = (max_samples - min_samples) / 2.0
    kernel_bandwidth_lower_bound = (max_samples - min_samples) / 20.0

    # do 20-fold cross validaton unless we have fewer samples
    num_cv = min(samples.shape[0], 20)
    # cross-validation
    grid = GridSearchCV(
        KernelDensity(),
        {'bandwidth': np.linspace(kernel_bandwidth_lower_bound, kernel_bandwidth_upper_bound, 40)},
        cv=num_cv,
    )

    grid.fit(samples.reshape(-1, 1))
    kernel_bandwidth = grid.best_params_['bandwidth']

    return kernel_bandwidth


def perform_kde(samples, kernel_bandwidth, support_points):
    """ Estimate pdf using kernel density estimation

    Args:
        samples (np.array):         samples for which to estimate pdf
        kernel_bandwidth (float):   kernel width to use in kde
        support_points (np.array):  points where to evaluate pdf
    Returns:
        np.array:                   pdf_estimate at support points
    """

    kde = KernelDensity(kernel='gaussian', bandwidth=kernel_bandwidth).fit(samples.reshape(-1, 1))

    y_density = np.exp(kde.score_samples(support_points.reshape(-1, 1)))
    return y_density
