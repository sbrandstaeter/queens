import numpy as np
import warnings
from pqueens.utils.plot_outputs import plot_pdf
from pqueens.utils.plot_outputs import plot_cdf
from pqueens.utils.plot_outputs import plot_failprob
from pqueens.utils.plot_outputs import plot_icdf

def process_ouputs(output_data, output_description):
    """ Process output from QUEENS models

        Args:
            output_data (dict):         Dictionary containing model output
            output_descripion (dict):   Dictionary describing desired output quantities
    """

    # do we want confindence intervals
    bayesian = output_description.get('bayesian', False)
    # check if we have the data to support this
    if "post_samples" not in output_data and bayesian is True:
        warnings.warn("Warning: Output data does not contain posterior samples. Not computing confidence intervals")
        bayesian = False
    # do we want plotting
    plot_results = output_description.get('plot_results', False)

    # result intervale
    result_interval = output_description.get('result_interval', None)
    if result_interval is None:
        # estimate interval from resutls
        result_interval = estimate_result_interval(output_data)

    # get number of support support points
    num_support_points = output_description.get('num_support_points', 100)
    support_points = np.linspace(result_interval[0], result_interval[1], num_support_points)

    mean_mean = estimate_mean(output_data)
    var_mean = estimate_var(output_data)

    #pdf_mean, pdf_conf = estimate_pdf(output_data,support_points)
    cdf_estimate = estimate_cdf(output_data, support_points, bayesian)
    #icdf_mean, icdf_mean = estimate_icdf(output_data, support_points)
    #f_prob_mean, f_prob_conf = estimate_failprob(output_data, support_points)

    if plot_results is True:
    #    plot_pdf(support_points, pdf_mean, pdf_conf)
        plot_cdf(cdf_estimate, support_points, bayesian)
    #    plot_icdf(support_points, pdf_mean, pdf_conf)
    #    plot_failprob(support_points, pdf_mean, pdf_conf)

def estimate_result_interval(output_data):
    """ Estimate interval of output data

    Estimate interval of output data and add small margins

    Args:
        output_data (dict):       Dictionary with output data

    Returns:
        list:                     Output interval

    """
    samples = output_data["mean"]
    min_data = np.amin(samples)
    max_data = np.amax(samples)

    interval_length = max_data - min_data
    my_min = min_data - interval_length/6
    my_max = max_data + interval_length/6

    return [my_min, my_max]

def estimate_mean(output_data):
    """ Estimate mean based on standard unbiased estimator

    Args:
        output_data (dict):       Dictionary with output data

    Returns:
        float                     Unbiased mean estimate

    """
    samples = output_data["mean"]
    return np.mean(samples)

def estimate_var(output_data):
    """ Estimate variance based on standard unbiased estimator

    Args:
        output_data (dict):       Dictionary with output data

    Returns:
        float                     Unbiased variance estimate

    """
    samples = output_data["mean"]
    return np.var(samples)

def estimate_pdf(output_data, support_points, bayesian):
    """ Compute estimate of PDF based on provided sampling data

    Args:
        output_data (dict):         Dictionary with output data
        support_points (np.array):  Points where to evaluate pdf
        bayesian (bool):            Compute confindence intervals etc.

    Returns:
        pdf:                        Dictionary with pdf estimates

    """
    raise NotImplementedError

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

def estimate_icdf(output_data, support_points, bayesian):
    """ Compute estimate of inverse CDF based on provided sampling data

    Args:
        output_data (dict):         Dictionary with output data
        support_points (np.array):  Points where to evaluate icdf
        bayesian (bool):            Compute confindence intervals etc.

    Returns:
        icdf:                        Dictionary with icdf estimates

    """
    raise NotImplementedError

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
