from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np

# TODO add tests for both methods

def estimate_bandwidth_for_kde(samples,min_samples,max_samples):
    """ Estimate optimal bandwidth for kde of pdf

    Args:
        samples (np.array):  samples for which to estimate pdf
        min_samples (float): smallest value
        max_samples (float): largest value
    Returns:
        float: estimate for optimal kernel_bandwidth
    """
    kernel_bandwidth = 0
    kernel_bandwidth_upper_bound = (max_samples-min_samples)/2.0
    kernel_bandwidth_lower_bound = (max_samples-min_samples)/20.0

     # do 20-fold cross-validation
    grid = GridSearchCV(KernelDensity(),{'bandwidth': \
        np.linspace(kernel_bandwidth_lower_bound,
                    kernel_bandwidth_upper_bound,
                    40)},cv=20)

    grid.fit(samples.reshape(-1,1))
    kernel_bandwidth = grid.best_params_['bandwidth']

    return kernel_bandwidth

def estimate_pdf(samples,kernel_bandwidth,support_points=None):
    """ Estimate pdf using kernel density estimation

    Args:
        samples (np.array):         samples for which to estimate pdf
        kernel_bandwidth (float):   kernel width to use in kde
        support_points (np.array):  points where to evaluate pdf
    Returns:
        np.array,np.array:          pdf_estimate at support points
    """
    if support_points is None:
        min_samples = np.amin(samples)
        max_samples = np.amax(samples)
        support_points =  np.linspace(min_samples, max_samples, 100)

    kde = KernelDensity(kernel='gaussian', bandwidth = \
        kernel_bandwidth).fit(samples.reshape(-1,1))

    y_density = np.exp(kde.score_samples(support_points.reshape(-1, 1)))
    return y_density, support_points
