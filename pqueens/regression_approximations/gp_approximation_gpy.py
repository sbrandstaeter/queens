import numpy as np
import GPy
from . regression_approximation import RegressionApproximation
from IPython.display import display
import pdb
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
        """ Create approxiamtion from options dictionary

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
        dx = np.max(self.X) - np.min(self.X)  # proper initialization of length scale
        dy = np.max(self.y) - np.min(self.y)  # proper initialization of variance
        k1 = GPy.kern.RBF(input_dim=1, variance=0.1*dy, lengthscale=0.25*dx, ARD=False, active_dims=[0])
        if input_dim == 2:
            k2 = GPy.kern.RBF(input_dim=1, variance=0.1*dy, lengthscale=0.25*dx, ARD=False, active_dims=[1])
            k = k1 + k2
        elif input_dim ==3:
            k2 = GPy.kern.RBF(input_dim=1, variance=0.1*dy, lengthscale=0.25*dx, ARD=False, active_dims=[1])
            k3 = GPy.kern.RBF(input_dim=1, variance=0.1*dy, lengthscale=0.25*dx, ARD=False, active_dims=[2])
            k = k1 * k2 * k3
        elif input_dim == 1:
            k = k1
        self.m = GPy.models.GPRegression(self.X, self.y, kernel=k, normalizer=True)

    def train(self):
        """ Train the GP by maximizing the likelihood """

        #self.m[".*Gaussian_noise"].constrain_positive()
        #self.m[".*Gaussian_noise"] = self.m.Y.var()*0.01
        #self.m[".*Gaussian_noise"].fix()
        self.m.optimize(messages=True)#max_iters=500)
        #self.m[".*Gaussian_noise"].unfix()
        self.m.optimize_restarts(num_restarts=10, max_iters=1000, max_f_eval=1000)
        display(self.m)

    def predict(self, Xnew):
        """ Compute latent function at Xnew

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function f

        Returns:
            dict: Dictionary with mean, variance, and possibly
             posterior samples of latent function at Xnew
        """
        output = {}
        mean, variance = self.predict_f(Xnew[:,:,None])
        output['mean'] = mean
        output['variance'] = variance
        if self.num_posterior_samples is not None:
            output['post_samples'] = self.predict_f_samples(Xnew, self.num_posterior_samples)

        return output

    def predict_y(self, Xnew):
        """ Compute latent function at Xnew

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function f

        Returns:
            dict: Dictionary with mean, variance, and possibly
             posterior samples of latent function at Xnew
        """
        output = {}
        mean, variance = self.m.predict(Xnew)
        output['mean'] = mean
        output['variance'] = variance# + self.m.posterior.Gaussian_noise.variance[0]
        if self.num_posterior_samples is not None:
            output['post_samples'] = self.predict_f_samples(Xnew, self.num_posterior_samples)

        return output


    def predict_f(self, Xnew):
        """ Compute the mean and variance of the latent function at Xnew

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function f

        Returns:
            np.array, np.array: mean and varaince of latent function at Xnew
        """
        return self.m.predict_noiseless(Xnew, full_cov=False)


    def predict_f_samples(self, Xnew, num_samples):
        """ Produce samples from the posterior latent funtion Xnew

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
