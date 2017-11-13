import numpy as np
import GPy
from  pqueens.designers.monte_carlo_designer import MonteCarloDesigner
from  pqueens.utils import pdf_estimation

# TODO make tests for this class
class MF_NARGP_Emulator(object):
    """ Class for creating multi fidelty GP based emulators

    This class constructs a multi fidelity GP emulator, using nonlinear
    information fusion algorihtm described in [1].
    Based on this emulator various statistical summarys can be
    computed and returned.

    Attributes:
        X_lofi (np.array):
            location of low fidelity design points
        y_lofi (np.array):
            low fidelity function value at desing points
        X_hifi (np.array):
            location of high fidelity design points
        y_hifi (np.array):
            high fidelity function value at desing points
        parameters (dict):
            Dictionary containing information about the input parameters
            such as their distribution

    """
    def __init__(self,X_lofi,X_hifi,y_lofi,y_hifi,parameters):
        """
        Args:
            X_lofi (np.array):
                location of low fidelity design points
            y_lofi (np.array):
                low fidelity function value at desing points
            X_hifi (np.array):
                location of high fidelity design points
            y_hifi (np.array):
                high fidelity function value at desing points
            parameters (dict):
                Dictionary containing information about the input parameters
                such as their distribution
        """
        self.X_lofi = X_lofi
        self.y_lofi = y_lofi

        self.X_hifi = X_hifi
        self.y_hifi = y_hifi

        self.parameters = parameters

        self.input_dimension = X_hifi.shape[1]

        # check that X_lofi and X_hifi have the same dimension
        dim_x = X_hifi.shape[1]
        if dim_x is not X_lofi.shape[1]:
            raise Exception("Dimension of low fidelity inputs and high fidelity inputs must be the same")

        # ensure that X_hifi is subset of X_lofi
        X_hifi_flat_set = set(X_hifi.flat)
        X_lofi_flat_set = set(X_lofi.flat)

        if not X_hifi_flat_set.issubset(X_lofi_flat_set):
            raise Exception("High fidelity inputs are not subset of low fidelity inputs")

        self.active_dimensions = np.arange(0,2)

        # train gp on low fidelity data
        k1 = GPy.kern.RBF(self.input_dimension, ARD = True)
        self.m1 = GPy.models.GPRegression(X=self.X_lofi, Y=self.y_lofi,
                                          kernel=k1)

        self.m1[".*Gaussian_noise"] = self.m1.Y.var()*0.01
        self.m1[".*Gaussian_noise"].fix()

        self.m1.optimize(max_iters = 500)

        self.m1[".*Gaussian_noise"].unfix()
        self.m1[".*Gaussian_noise"].constrain_positive()

        self.m1.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)

        mu1, _ = self.m1.predict(self.X_hifi)

        # train gp on high fidelity data
        XX = np.hstack((self.X_hifi, mu1))

        k2 = GPy.kern.RBF(1, active_dims = [self.input_dimension]) \
             * GPy.kern.RBF(self.input_dimension,
                            active_dims = self.active_dimensions, ARD = True) \
             + GPy.kern.RBF(self.input_dimension,
                            active_dims = self.active_dimensions, ARD = True)

        self.m2 = GPy.models.GPRegression(X=XX, Y=self.y_hifi, kernel=k2)

        self.m2[".*Gaussian_noise"] = self.m2.Y.var()*0.01
        self.m2[".*Gaussian_noise"].fix()

        self.m2.optimize(max_iters = 500)

        self.m2[".*Gaussian_noise"].unfix()
        self.m2[".*Gaussian_noise"].constrain_positive()

        self.m2.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)

        # TODO remove hard coded values and pass via contructor argument instead
        self.num_emulator_samples = 100
        self.num_mc_samples = 500

    def predict(self,x_test):
        """ Compute predictions for x_test

        Args:
            x_test (np.array): test locations at which to predict

        Returns:
            np.array,np.array: mean and variance of posterior
            predictive distribution
        """
        dim_x = x_test.shape[1]
        num_test_points = x_test.shape[0]
        if dim_x is not self.X_lofi.shape[1]:
            raise Exception("Dimension of inputs does not match dimension of emulator")

        # compute mean and full covariance matrix
        mu1, C1 = self.m1.predict(x_test, full_cov=True)
        # generate nsample samples at x
        Z = np.random.multivariate_normal(mu1.flatten(),C1,
                                          self.num_emulator_samples)

        # push samples through level 2
        tmp_m = np.zeros((self.num_emulator_samples,num_test_points))
        tmp_v = np.zeros((self.num_emulator_samples,num_test_points))

        for i in range(0,self.num_emulator_samples):
            mu, v = self.m2.predict(np.hstack((x_test, Z[i,:][:,None])))
            tmp_m[i,:] = mu.flatten()
            tmp_v[i,:] = v.flatten()

        # get posterior mean and variance
        # TODO check this, this is not so clear to me
        mean_x_test = np.mean(tmp_m, axis = 0)[:,None]
        var = np.mean(tmp_v, axis = 0)[:,None]+ np.var(tmp_m, axis = 0)[:,None]
        var_x_test = np.abs(var)

        return mean_x_test, var_x_test

    def compute_mean(self):
        """ Compute mean of emulator output w.r.t. x

        Compute variance of emulator output based on the distributions passed
        via the parameters dict. Since the GP is a Bayesian emulator, we
        can also compute the variance of the mean based on several posterior
        samples of the emulator.

        Returns:
            float, float : mean of the variance, variance of the variance
        """
        mean_mean, mean_var, _, _ = self.compute_mean_and_var()
        return  mean_mean, mean_var

    def compute_var(self):
        """ Compute variance of emulator output w.r.t. x

        Compute variance of emulator output based on the distributions passed
        via the parameters dict. Since the GP is a Bayesian emulator, we
        can also compute the variance of the mean based on several posterior
        samples of the emulator.

        Returns:
            float, float : mean of the variance, variance of the variance
        """
        _, _, var_mean, var_var = self.compute_mean_and_var()
        return var_mean, var_var

    def compute_mean_and_var(self):
        """ Compute mean and variance of emulator output w.r.t. inputs x

        Compute variance and mean of emulator output based on the distributions
        passed via the parameters dict. Since the multi fidelity GP is a
        Bayesian emulator, we can also compute the variance of the mean based
        on several posterior samples of the emulator.

        Returns:
            float, float, float, float: mean of the mean, variance of the mean,
            mean of the variance, variance of the variance
        """
        sample_values_raw = \
            self.create_posterior_samples(self.num_emulator_samples,
                                          self.num_emulator_samples,
                                          self.num_mc_samples)

        # integrate over p_x(X)
        mean_values = np.mean(sample_values_raw, axis=2)

        mean_mean = np.mean(mean_values, (0, 1))
        mean_var  = np.var(mean_values, (0, 1))

        # integrate over p_x(X)
        var_values = np.var(sample_values_raw, axis=2)

        var_mean = np.mean(var_values, (0, 1))
        var_var  = np.var(var_values, (0, 1))

        return mean_mean, mean_var, var_mean, var_var

    def compute_pdf(self):
        """ Compute probability density function of emulator output

        This function computes probability density function of the emulator
        output using kernel density estiation. Confindence bounds on the
        pdf are also provided harnessing the Bayesian nature of the emulator.

        Returns:
            dict,np.array: dict with arrays of pdf values at the locations
            stored in the second output argument
        """

        sample_values_raw = \
            self.create_posterior_samples(self.num_emulator_samples,
                                          self.num_emulator_samples,
                                          self.num_mc_samples)

        min_samples = np.amin(sample_values_raw)
        max_samples = np.amax(sample_values_raw)

        y_plot =  np.linspace(min_samples, max_samples, 100)

        y_density = np.zeros((self.num_emulator_samples,
                              self.num_emulator_samples,len(y_plot)))

        for i in range(self.num_emulator_samples):
            for j in range(self.num_emulator_samples):
                # estimate once optimal kernel bandwidth
                if i == 0 and j == 0:
                    kernel_bandwidth = \
                        pdf_estimation.estimate_bandwidth_for_kde(
                            sample_values_raw[i,j,:].reshape(-1,1),
                            min_samples,max_samples)

                y_density[i,j,:], _  = \
                    pdf_estimation.estimate_pdf(
                        sample_values_raw[i,j,:].reshape(-1,1),
                        kernel_bandwidth,y_plot)

        # compute mean median var and quanitiles of pdf
        pdf_vals ={}
        pdf_vals['mean'] = np.mean(y_density, (0, 1))
        pdf_vals['median'] = np.median(y_density,(0,1))
        pdf_vals['var'] = np.var(y_density,(0,1))

        test = np.percentile(y_density, 2.5,0)
        pdf_vals['quant_low'] = np.percentile(test, 2.5,0)
        test = np.percentile(y_density, 97.5, 0)
        pdf_vals['quant_high'] = np.percentile(test, 97.5, 0)

        return pdf_vals, y_plot

    def compute_cdf(self):
        """ Compute cumulative density function of emulator output

        This function computes the cumulative distribution
        function (cdf) of the emulator. Confindence bounds on the
        cdf are also provided harnessing the Bayesian nature of the emulator.

        Returns:
            dict,np.array: dict with arrays of cdf values at the locations
            stored in the second output argument

        """
        sample_values_raw = self.create_posterior_samples(100,100,500)

        # sort samples in ascending order first
        sample_values_sorted=np.sort(sample_values_raw,axis=2)
        # average over l1 and l2 (axis 0 and 1)
        sample_values = {}
        sample_values['mean'] = np.mean(sample_values_sorted,(0,1))

        # compute cdf
        cdf = np.arange(1, len(sample_values_sorted[0,0,:])+1) / \
            float(len(sample_values_sorted[0,0,:]))

        # percentile function cannot take multiple axis hence we do it in two steps
        # first compute percentile w.r.t. level 1
        temp = np.percentile(sample_values_sorted,2.5, 0)
        # second compute percentile w.r.t. level 2
        sample_values['q_lower_bound'] = np.percentile(temp,2.5, 0)

        # compute median in the same fashion
        temp = np.percentile(sample_values_sorted,50, 0)
        sample_values['median'] = np.percentile(temp,50, 0)

        # compute 97.5 % quantile also using the same approach
        temp = np.percentile(sample_values_sorted,97.5, 0)
        sample_values['q_upper_bound']= np.percentile(temp,97.5, 0)

        return cdf,sample_values

    def compute_failure_probability_function(self):
        """ Compute failure probabiliy function of emulator output

        This function computed the quantile function of the emulator output
        and includes a confidence regions using the Bayesian nature of the
        emulator.

        Returns:
            dict,np.array: dict wiht arrays of quantiles values and an
            array with corresponding quantiles
        """
        my_quantiles = np.linspace(0,100,1000)
        my_evals = self.create_posterior_samples(self.num_emulator_samples,
                                                 self.num_emulator_samples,
                                                 self.num_mc_samples)
        my_quantile = {}
        y_quantile = np.percentile(my_evals,my_quantiles,axis=2)
        print("y_quantile shape{}".format(y_quantile.shape))

        temp = np.percentile(y_quantile,2.5, 2)
        my_quantile['q_lower_bound'] = np.percentile(temp, 2.5, 1)

        temp = np.percentile(y_quantile, 97.5, 2)
        my_quantile['q_upper_bound'] = np.percentile(temp, 97.5, 1)

        temp = np.percentile(y_quantile, 50, 2)
        my_quantile['median'] = np.percentile(temp, 50, 1)

        my_quantile['mean'] = np.mean(y_quantile, (2, 1))

        return my_quantile, 1-my_quantiles/100.0

    def compute_quantile_function(self):
        """ Compute quantile function of emulator output

        This function computed the quantile function of the emulator output
        and includes a confidence regions using the Bayesian nature of the
        emulator.

        Returns:
            dict,np.array: dict wiht arrays of quantiles values and an
            array with corresponding quantiles
        """
        my_quantiles = np.linspace(0,100,1000)
        my_evals = self.create_posterior_samples(self.num_emulator_samples,
                                                 self.num_emulator_samples,
                                                 self.num_mc_samples)
        my_quantile = {}
        y_quantile = np.percentile(my_evals,my_quantiles,axis=2)
        temp = np.percentile(y_quantile,2.5, 2)
        my_quantile['q_lower_bound'] = np.percentile(temp, 2.5, 1)

        temp = np.percentile(y_quantile, 97.5, 2)
        my_quantile['q_upper_bound'] = np.percentile(temp, 97.5, 1)

        temp = np.percentile(y_quantile, 50, 2)
        my_quantile['median'] = np.percentile(temp, 50, 1)

        my_quantile['mean'] = np.mean(y_quantile, (0, 1))

        return my_quantile, my_quantiles


    def create_posterior_samples(self,num_realizations_l1,
                                 num_realizations_l2,num_input_samples):
        """ Generate samples from posterior distribution

        Args:
            num_realizations_l1 (float): number of posterior realizations of
            level 1 GP
            num_realizations_l2 (float): number of posterior realizations of
            level 2 GP
            num_input_samples (float): number of samples in input dimension

        Returns:
            np.array: samples from posterior predictive distribution

        """

        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           num_input_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        # compute mean and full covariance matrix
        mu1, C1 = self.m1.predict(my_mc_samples, full_cov=True)

        # generate num_emulator_samples samples of level 1 at my_mc_samples
        Z = np.random.multivariate_normal(mu1.flatten(),
                                          C1,num_realizations_l1)

        # init array to store samples
        my_samples= np.zeros((num_realizations_l1,
                              num_realizations_l2,
                              num_input_samples))

        # loop over all samples of f1
        for i in range(0,num_realizations_l1):
            mu2, C2 = self.m2.predict(np.hstack((my_mc_samples,
                                                 Z[i,:][:,None])),
                                                 full_cov=True)

            # generate num_realizations_l2 of f2 based on f1(i)
            my_samples[i,:,:] = \
                np.random.multivariate_normal(mu2.flatten(),
                                              C2,num_realizations_l2)

        return my_samples
