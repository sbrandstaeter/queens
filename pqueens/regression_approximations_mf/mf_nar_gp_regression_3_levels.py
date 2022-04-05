import GPy
import numpy as np


class MF_NAR_GP_Regression_3_Levels(object):
    """Class for creating multi-fidelity nonlinear auto-regressive GP based
    emulator.

        This class constructs a multi fidelity GP emulator, using nonlinear
        information fusion algorihtm described in [1] with three levels.
        Based on this emulator various statistical summarys can be
        computed and returned.


    Attributes:
        num_fidelity_levels (int):      Levels of fidelity (currently hard coded to three)
        X_lofi (np.array):              Training inputs low-fidelity
        X_medfi (np.array):             Training inputs medium-fidelity
        X_hifi (np.array):              Training inputs high-fidelity
        input_dimension (int):          Dimension of training inputs
        active_dimensions (int):
        num_posterior_samples (int):    Number of posterior samples for inference
                                        and prediction
        m1 (GPy.model):                 GPy based Gaussian process model
        m2 (GPy.model):                 GPy based Gaussian process model
        m3 (GPy.model):                 GPy based Gaussian process model
    """

    @classmethod
    def from_options(cls, approx_options, Xtrain, ytrain):
        """Create approxiamtion from options dictionary.

        Args:
            approx_options (dict): Dictionary with approximation options
            x_train (np.array):    Training inputs
            y_train (np.array):    Training outputs

        Returns:
            MF_NAR_GP_Regression: approximation object
        """
        if len(Xtrain) != 3:
            raise ValueError("MF_NAR_GP_Regression_ThreeLevel is only implemented for three levels")

        num_posterior_samples = approx_options.get('num_posterior_samples', 100)
        return cls(Xtrain, ytrain, num_posterior_samples)

    def __init__(self, Xtrain, ytrain, num_posterior_samples):
        """
        Args:
            Xtrain (list):                  List of arrays of location of design points
            ytrain (np.array):              List of arrays of values at desing points
            num_posterior_samples (int):    Number of posterior samples for inference
                                            and prediction
        """

        # check that X_lofi and X_hifi have the same dimension
        dim_x = Xtrain[0].shape[1]
        if dim_x is not Xtrain[1].shape[1] or not Xtrain[2].shape[1]:
            raise Exception("Dimension of inputs must be the same across levels")
        self.num_fidelity_levels = 3

        self.X_lofi = Xtrain[0]
        self.y_lofi = ytrain[0]

        self.X_medfi = Xtrain[1]
        self.y_medfi = ytrain[1]

        self.X_hifi = Xtrain[2]
        self.y_hifi = ytrain[2]

        self.input_dimension = Xtrain[0].shape[1]

        # ensure that X_hifi is subset of X_lofi
        X_hifi_flat_set = set(Xtrain[2].flat)
        X_medfi_flat_set = set(Xtrain[1].flat)
        X_lofi_flat_set = set(Xtrain[0].flat)

        if not X_hifi_flat_set.issubset(X_medfi_flat_set) or not X_medfi_flat_set.issubset(
            X_lofi_flat_set
        ):
            raise Exception("Input sets must be nested")

        self.active_dimensions = np.arange(0, self.input_dimension)
        # active_dimensions = np.arange(0,dim)

        self.active_dimensions = np.arange(0, 2)

        self.num_posterior_samples = num_posterior_samples

        self.m1 = None
        self.m2 = None
        self.m3 = None

    def train(self):
        # train gp on low fidelity data
        k1 = GPy.kern.RBF(self.input_dimension, ARD=True)

        self.m1 = GPy.models.GPRegression(X=self.X_lofi, Y=self.y_lofi, kernel=k1, normalizer=True)
        self.m1[".*Gaussian_noise"] = self.m1.Y.var() * 0.01
        self.m1[".*Gaussian_noise"].fix()
        self.m1.optimize(max_iters=500)
        self.m1[".*Gaussian_noise"].unfix()
        self.m1[".*Gaussian_noise"].constrain_positive()
        self.m1.optimize_restarts(30, optimizer="bfgs", max_iters=1000)

        mu1, _ = self.m1.predict(self.X_medfi)

        # train gp on medium fidelity data
        XX = np.hstack((self.X_medfi, mu1))

        k2 = GPy.kern.RBF(1, active_dims=[self.input_dimension]) * GPy.kern.RBF(
            self.input_dimension, active_dims=self.active_dimensions, ARD=True
        ) + GPy.kern.RBF(self.input_dimension, active_dims=self.active_dimensions, ARD=True)

        self.m2 = GPy.models.GPRegression(X=XX, Y=self.y_medfi, kernel=k2, normalizer=True)

        self.m2[".*Gaussian_noise"] = self.m2.Y.var() * 0.01
        self.m2[".*Gaussian_noise"].fix()
        self.m2.optimize(max_iters=500)
        self.m2[".*Gaussian_noise"].unfix()
        self.m2[".*Gaussian_noise"].constrain_positive()
        self.m2.optimize_restarts(30, optimizer="bfgs", max_iters=1000)

        # Prepare for level 3: sample f_1 at X3
        nsamples = self.num_posterior_samples
        ntest = self.X_hifi.shape[0]
        mu0, C0 = self.m1.predict(self.X_hifi, full_cov=True)
        Z = np.random.multivariate_normal(mu0.flatten(), C0, nsamples)
        tmp_m = np.zeros((nsamples, ntest))
        tmp_v = np.zeros((nsamples, ntest))

        # push samples through f_2
        for i in range(0, nsamples):
            mu, v = self.m2.predict(np.hstack((self.X_hifi, Z[i, :][:, None])))
            tmp_m[i, :] = mu.flatten()
            tmp_v[i, :] = v.flatten()

        # get mean and variance at X3
        mu2 = np.mean(tmp_m, axis=0)
        v2 = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)
        mu2 = mu2[:, None]
        v3 = np.abs(v2[:, None])

        # train gp on high fidelity data
        XX = np.hstack((self.X_hifi, mu2))

        k3 = GPy.kern.RBF(1, active_dims=[self.input_dimension]) * GPy.kern.RBF(
            self.input_dimension, active_dims=self.active_dimensions, ARD=True
        ) + GPy.kern.RBF(self.input_dimension, active_dims=self.active_dimensions, ARD=True)

        self.m3 = GPy.models.GPRegression(X=XX, Y=self.y_hifi, kernel=k3, normalizer=True)

        self.m3[".*Gaussian_noise"] = self.m3.Y.var() * 0.01
        self.m3[".*Gaussian_noise"].fix()
        self.m3.optimize(max_iters=500)
        self.m3[".*Gaussian_noise"].unfix()
        self.m3[".*Gaussian_noise"].constrain_positive()
        self.m3.optimize_restarts(30, optimizer="bfgs", max_iters=1000)

    def predict(self, x_test):
        """Compute latent function at x_test.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function f

        Returns:
            dict: Dictionary with mean, variance, and possibly
                  posterior samples of latent function at x_test
        """
        output = {}
        mean, variance = self.predict_f(x_test)
        output['mean'] = mean
        output['variance'] = variance
        if self.num_posterior_samples is not None:
            output['post_samples'] = self.predict_f_samples(x_test, self.num_posterior_samples)

        return output

    def predict_f(self, x_test):
        """Compute the mean and variance of the latent function at x_test.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function f

        Returns:
            np.array, np.array: mean and varaince of latent function at Xnew
        """

        dim_x = x_test.shape[1]
        num_test_points = x_test.shape[0]
        if dim_x is not self.X_lofi.shape[1]:
            raise Exception("Dimension of inputs does not match dimension of emulator")

        # compute mean and full covariance matrix
        mu1, C1 = self.m1.predict(x_test, full_cov=True)
        # generate nsample samples at x
        Z = np.random.multivariate_normal(mu1.flatten(), C1, self.num_posterior_samples)

        # push samples through f_2 and f_3
        tmp_m = np.zeros((self.num_posterior_samples**2, num_test_points))
        tmp_v = np.zeros((self.num_posterior_samples**2, num_test_points))
        cnt = 0

        for i in range(0, self.num_posterior_samples):
            mu, C = self.m2.predict(np.hstack((x_test, Z[i, :][:, None])), full_cov=True)
            Q = np.random.multivariate_normal(mu.flatten(), C, self.num_posterior_samples)
            for j in range(0, self.num_posterior_samples):
                mu, v = self.m3.predict(np.hstack((x_test, Q[j, :][:, None])))
                tmp_m[cnt, :] = mu.flatten()
                tmp_v[cnt, :] = v.flatten()
                cnt = cnt + 1

        # get f_2 posterior mean and variance at Xtest
        mean_x_test = np.mean(tmp_m, axis=0)
        v3 = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)
        mean_x_test = mean_x_test[:, None]
        var_x_test = np.abs(v3[:, None])
        #
        # for i in range(0, self.num_posterior_samples):
        #     mu, v = self.m2.predict(np.hstack((x_test, Z[i, :][:, None])))
        #     tmp_m[i, :] = mu.flatten()
        #     tmp_v[i, :] = v.flatten()
        #
        # # get posterior mean and variance
        # # TODO check this, this is not so clear to me
        # mean_x_test = np.mean(tmp_m, axis=0)[:, None]
        # var = np.mean(tmp_v, axis=0)[:, None]+ np.var(tmp_m, axis=0)[:, None]
        # var_x_test = np.abs(var)

        # return mu1, var_x_test
        return mean_x_test.reshape((-1, 1)), var_x_test.reshape((-1, 1))

        # return mean_x_test, var_x_test

    def predict_f_samples(self, Xnew, num_samples):
        """Produce samples from the posterior latent funtion Xnew.

        Args:
            Xnew (np.array):    Inputs at which to evaluate latent function f
            num_samples (int):  Number of posterior field_realizations of GP

        Returns:
            np.array: samples of latent function at Xnew
        """
        raise NotImplementedError
