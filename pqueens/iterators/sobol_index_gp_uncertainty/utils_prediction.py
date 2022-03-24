"""Utils for Gaussian process prediction."""
import numpy as np


def predict_mean(x_test, regression_approximation):
    """Evaluate the posterior mean of the GP m_GP(X).

    Args:
        x_test (ndarray): input where to make prediction based on Gaussian process
        regression_approximation (RegressionApproximation): holds Gaussian process regression

    Returns:
        output['mean'] (ndarray): mean prediction of the Gaussian process at x_test
    """
    output = regression_approximation.predict(x_test, support='f')

    return output['mean']


def sample_realizations(
    x_test, gp_approximation_gpy, number_gp_realizations, seed_posterior_samples=None
):
    """Sample realizations from the Gaussian process at the points x_test.

    Args:
        x_test (ndarray): input where to make predictions based on Gaussian process
        gp_approximation_gpy (RegressionApproximation): holds Gaussian process regression
        number_gp_realizations (int): number of posterior samples to draw
        seed_posterior_samples (int): seed for posterior samples

    Returns:
        realizations (ndarray): realizations of the Gaussian process
    """
    # We circumvent our implementation in regression approximation classes
    # because it always predicts f and f_samples together.
    # Here, we only draw the posterior samples, no mean and covariance.
    x_test = np.atleast_2d(x_test).reshape((-1, gp_approximation_gpy.model.input_dim))
    x_test = gp_approximation_gpy.scaler_x.transform(x_test)

    if seed_posterior_samples:
        np.random.seed(seed_posterior_samples)

    realizations = gp_approximation_gpy.model.posterior_samples_f(x_test, number_gp_realizations)

    return realizations.squeeze()
