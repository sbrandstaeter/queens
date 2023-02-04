"""A multi-fidelity Gaussian Process implementation."""

import logging

import GPy
import numpy as np

_logger = logging.getLogger(__name__)


class MF_ICM_GP_Regression(object):
    """Class for creating multi-fidelity GP based emulator.

    This class constructs a multi-fidelity GP emulator, currently using a GPy
    model. Based on this emulator various statistical summaries can be
    computed and returned.

    Attributes:
        Xtrain (list):
            List of arrays of location of design points.
        ytrain (np.array):
            List of arrays of values at design points.
        num_fidelity_levels: TODO_doc
        num_posterior_samples: TODO_doc
        m (Gpy.model):
            GPy based Gaussian process model.
    """

    @classmethod
    def from_options(cls, approx_options, Xtrain, ytrain):
        """Create approximation from options dictionary.

        Args:
            approx_options (dict): Dictionary with approximation options
            Xtrain (np.array):    Training inputs
            ytrain (np.array):    Training outputs

        Returns:
            gp_approximation_gpy: Approximation object
        """
        num_fidelity_levels = len(Xtrain)
        num_posterior_samples = approx_options.get('num_posterior_samples', None)
        return cls(Xtrain, ytrain, num_fidelity_levels, num_posterior_samples)

    def __init__(self, Xtrain, ytrain, num_fidelity_levels, num_posterior_samples):
        """TODO_doc.

        Args:
            Xtrain (list): list of arrays of location of design points
            ytrain (np.array): list of arrays of values at design points
            num_fidelity_levels (int): number of fidelity levels
            num_posterior_samples (int): number of posterior samples for prediction
        """
        # check that X_lofi and X_hifi have the same dimension
        dim_x = Xtrain[0].shape[1]
        if dim_x is not Xtrain[1].shape[1]:
            raise ValueError(
                "Dimension of low fidelity inputs and high fidelity inputs must be the same"
            )

        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.num_fidelity_levels = num_fidelity_levels
        self.num_posterior_samples = num_posterior_samples

        # define icm multi output kernel
        icm = GPy.util.multioutput.ICM(
            input_dim=dim_x, num_outputs=self.num_fidelity_levels, kernel=GPy.kern.RBF(dim_x)
        )

        self.m = GPy.models.GPCoregionalizedRegression(self.Xtrain, self.ytrain, kernel=icm)

    def train(self):
        """Train the GP by maximizing the likelihood."""
        # #For this kernel, B.kappa encodes the variance now.
        self.m['.*rbf.var'].constrain_fixed(1.0)

        self.m[".*Gaussian_noise"] = self.m.Y.var() * 0.01
        self.m[".*Gaussian_noise"].fix()

        self.m.optimize(max_iters=500)

        self.m[".*Gaussian_noise"].unfix()
        self.m[".*Gaussian_noise"].constrain_positive()
        self.m.optimize_restarts(30, optimizer="bfgs", max_iters=1000)

    def predict(self, Xnew):
        """Compute latent function at *Xnew*.

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function 'f'

        Returns:
            dict: Dictionary with mean, variance and possibly
            posterior samples of latent function at *Xnew*
        """
        output = {}
        mean, variance = self.predict_f(Xnew)
        output['mean'] = mean
        output['variance'] = variance
        if self.num_posterior_samples is not None:
            output['post_samples'] = self.predict_f_samples(Xnew, self.num_posterior_samples)
        _logger.info("output type %s", type(output))
        return output

    def predict_f(self, Xnew, level=None):
        """Compute the mean and variance of the latent function at *Xnew*.

        Args:
            Xnew (np.array): Inputs at which to evaluate latent function 'f'
            level (int): Level for which to make prediction

        Returns:
            np.array, np.array: Mean and variance of latent function at *Xnew*
        """
        dim_x = Xnew.shape[1]
        if dim_x is not self.Xtrain[0].shape[1]:
            raise ValueError("Dimension of inputs does not match dimension of emulator")

        if level is None:
            level = self.num_fidelity_levels

        if level > self.num_fidelity_levels:
            raise ValueError(
                f"Cannot access level {self.num_fidelity_levels} since number of levels is {level}"
            )

        # add level dimension to input
        my_samples = np.hstack([Xnew, level - 1 * np.ones((Xnew.shape[0], 1))])

        my_mean, my_var = self.m.predict_noiseless(my_samples)
        # TODO check what happens to the noise

        return my_mean.reshape((-1, 1)), my_var.reshape((-1, 1))

    def predict_f_samples(self, Xnew, num_samples, level=None):
        """Produce samples from the posterior latent function *Xnew*.

        Args:
            Xnew (np.array):    Inputs at which to evaluate latent function 'f'
            num_samples (int):  Number of posterior *field_realizations* of GP
            level (int): Level for which to make prediction


        Returns:
            np.array: Samples of latent function at *Xnew*
        """
        if level is None:
            level = self.num_fidelity_levels

        if level > self.num_fidelity_levels:
            raise ValueError(
                f"Cannot access level {self.num_fidelity_levels} since number of levels is {level}"
            )

        my_samples = np.hstack([Xnew, level - 1 * np.ones((Xnew.shape[0], 1))])

        return self.m.posterior_samples_f(my_samples, num_samples)
