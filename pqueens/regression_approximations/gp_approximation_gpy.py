import numpy as np
import GPy
from .regression_approximation import RegressionApproximation
from IPython.display import display


class GPGPyRegression(RegressionApproximation):
    """ Class for creating GP based regression model based on GPy

    This class constructs a GP regression, currently using a GPy model.
    Currently, a lot of parameters are still hard coded, which will be
    improved in the future

    Attributes:
        X (np.array):               Training inputs
        y (np.array):               Training outputs
        m (Gpy.model):              GPy based Gaussian process model

    """

    @classmethod
    def from_options(cls, approx_options, x_train, y_train):
        """ Create approximation from options dictionary

        Args:
            approx_options (dict): Dictionary with approximation options
            x_train (np.array):    Training inputs
            y_train (np.array):    Training outputs

        Returns:
            gp_approximation_gpy: approximation object
        """
        num_posterior_samples = approx_options.get('num_posterior_samples', None)
        return cls(x_train, y_train, num_posterior_samples)

    def __init__(self, X, y, num_posterior_samples):
        """
        Args:
            approx_options (dict):  Dictionary with model options
            X (np.array):           Training inputs
            y (np.array):           Training outputs
        """
        self.X = X
        self.y = y
        self.num_posterior_samples = num_posterior_samples

        # input dimension
        input_dim = self.X.shape[1]

        # simple GP Model
        lengthscale_0 = 0.1 * abs(np.max(self.X) - np.min(self.X))  # proper initialization of
        # length scale
        variance_0 = abs(np.max(self.y) - np.min(self.y))  # proper initialization of variance

        k_list = [
            GPy.kern.RBF(
                input_dim=1,
                variance=variance_0,
                lengthscale=lengthscale_0,
                ARD=False,
                active_dims=[dim],
            )
            for dim in range(input_dim)
        ]

        k = k_list[0]
        if len(k_list) > 1:
            for k_ele in k_list[1:-1]:
                k += k_ele

        self.m = GPy.models.GPRegression(self.X, self.y, kernel=k, normalizer=True)

    def train(self):
        """ Train the GP by maximizing the likelihood """
        self.m.optimize_restarts(num_restarts=5, max_iters=1000, messages=True)
        display(self.m)

    def predict(self, Xnew, support='y', full_cov=False):
        """
        Predict the posterior distribution at Xnew with respect to the data 'y' or the latent
        function 'f'.

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function f
            support (str): Probabilistic support of random process (default: 'y'). Possible options
                           are 'y' or 'f'. Here, 'f' means the latent function so that the posterior
                           variance of the GP is calculated with respect to f. In contrast 'y'
                           refers to the data itself so that the posterior variance is computed
                           with respect to 'y' (f is integrated out) leading to an extra addition
                           of noise in the posterior variance)
            full_cov (bool): Boolean that specifies whether the entire posterior covariance matrix
                             should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples at Xnew
        """
        if support == 'y':
            output = self.predict_y(Xnew, full_cov=full_cov)
        elif support == 'f':
            output = self.predict_f(Xnew, full_cov=full_cov)
        else:
            raise NotImplementedError(
                f"You choose support={support} but the only valid options " f"are 'y' or 'f'"
            )
        return output

    def predict_y(self, Xnew, full_cov=False):
        """
        Compute the posterior distribution at Xnew with respect to the data 'y'

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function f

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
                           posterior samples of latent function at Xnew
        """
        Xnew = np.atleast_2d(Xnew).reshape((-1, self.m.input_dim))
        output = {}
        output["mean"], output["variance"] = self.m.predict(Xnew, full_cov=full_cov)
        if self.num_posterior_samples is not None:
            output["post_samples"] = self.predict_f_samples(Xnew, self.num_posterior_samples)

        return output

    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at Xnew

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function f

        Returns:
            np.array, np.array: mean and variance of latent function at Xnew
        """
        Xnew = np.atleast_2d(Xnew).reshape((-1, self.m.input_dim))
        output = {}
        output["mean"], output["variance"] = self.m.predict_noiseless(Xnew, full_cov=full_cov)
        if self.num_posterior_samples is not None:
            output["post_samples"] = self.predict_f_samples(Xnew, self.num_posterior_samples)

        return output

    def predict_f_samples(self, Xnew, num_samples):
        """ Produce samples from the posterior latent function Xnew

            Args:
                Xnew (np.array):    Inputs at which to evaluate latent function f
                num_samples (int):  Number of posterior realizations of GP

            Returns:
                np.array, np.array: mean and variance of latent functions at Xnew
        """

        post_samples = self.m.posterior_samples_f(Xnew, num_samples)
        # GPy returns 3d array middle dimension indicates number of ouputs, i.e.
        # it is only != 1 for multioutput processes
        if post_samples.shape[1] != 1:
            raise Exception("GPGPyRegression can not deal with multioutput GPs")
        return np.reshape(post_samples, (Xnew.shape[0], num_samples))
