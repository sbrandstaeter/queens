"""Estimation of the probability density function based on samples from the
distribution.

It uses the kernel density estimation (kde) algorithm.
"""

import pdb

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def estimate_bandwidth_for_kde(samples, min_samples, max_samples):
    """Estimate optimal bandwidth for kde of pdf.

    Args:
        samples (np.array):  samples for which to estimate pdf
        min_samples (float): smallest value
        max_samples (float): largest value

    Returns:
        float: estimate for optimal kernel_bandwidth
    """
    kernel_bandwidth_upper_bound = np.log10((max_samples - min_samples) / 2.0)
    kernel_bandwidth_lower_bound = np.log10((max_samples - min_samples) / 30.0)

    # do 30-fold cross-validation and use all cores available to speed-up the process
    # we use the epanechinkov kernel as it is bounded in contrast to the gaussian kernel
    # we use a log grid to emphasize the smaller bandwidth values
    grid = GridSearchCV(
        KernelDensity(kernel='epanechnikov'),
        {'bandwidth': np.logspace(kernel_bandwidth_lower_bound, kernel_bandwidth_upper_bound, 40)},
        cv=30,
        n_jobs=-1,
    )

    grid.fit(samples.reshape(-1, 1))
    kernel_bandwidth = grid.best_params_['bandwidth']
    print('bandwidth = %s' % kernel_bandwidth)

    return kernel_bandwidth


def estimate_pdf(samples, kernel_bandwidth, support_points=None):
    """Estimate pdf using kernel density estimation.

    Args:
        samples (np.array):         samples for which to estimate pdf
        kernel_bandwidth (float):   kernel width to use in kde
        support_points (np.array):  points where to evaluate pdf
    Returns:
        np.array,np.array:          pdf_estimate at support points
    """
    # make sure that we have at least 2 D column vectors but do not change correct 2D format
    samples = np.atleast_2d(samples).T

    # support points given
    if support_points is None:
        min_samples = np.amin(samples)
        max_samples = np.amax(samples)
        support_points = np.linspace(min_samples, max_samples, 100)
        support_points = np.meshgrid(*[support_points[:, None]] * samples.shape[1])
        points = support_points[0].reshape(-1, 1)
        if len(points.shape) > 1:
            for col in range(1, samples.shape[1]):
                points = np.hstack(
                    (points, support_points[col].reshape(-1, 1))
                )  # reshape matrix to vector with all combinations
        support_points = np.atleast_2d(points)
    else:
        support_points = np.atleast_2d(support_points).T

        # no support points given
    kde = KernelDensity(kernel='gaussian', bandwidth=kernel_bandwidth).fit(samples)

    y_density = np.exp(kde.score_samples(support_points))
    return y_density, support_points
