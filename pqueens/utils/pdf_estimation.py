from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold #LeaveOneOut
import numpy as np
import pdb
# TODO remove file, outdated implementation

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
    kernel_bandwidth_upper_bound = (max_samples-min_samples)/1.0
    kernel_bandwidth_lower_bound = (max_samples-min_samples)/40.0

     # do 20-fold cross-validation
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),{'bandwidth': \
        np.linspace(kernel_bandwidth_lower_bound,
                    kernel_bandwidth_upper_bound,
                    20)},cv=15,return_train_score=False)

    grid.fit(samples.reshape(-1,1))
    kernel_bandwidth = grid.best_params_['bandwidth']
    print('bandwidth = %s' % kernel_bandwidth)

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
        min_samples = -0.5 # np.amin(samples)  #DG: -0.5  #fsi:0.02
        max_samples = 2.0 #np.amax(samples)   #fsi: 0.07
        support_points =  np.linspace(min_samples, max_samples, 500)
        support_points = np.meshgrid(*[support_points[:,None]]*samples.shape[1])
        points = support_points[0].reshape(-1,1)
        if len(points.shape) > 1:
            for col in range(1,samples.shape[1]):
                 points = np.hstack((points, support_points[col].reshape(-1,1))) # reshape matrix to vector with all combinations

    #samples= [sam.T for sam in samples.T]
    #samples = np.meshgrid(*samples)
    kde = KernelDensity(kernel='gaussian', bandwidth = \
        kernel_bandwidth).fit(samples)

    y_density = np.exp(kde.score_samples(points))
    return y_density, points
