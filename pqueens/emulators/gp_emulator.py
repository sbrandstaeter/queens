import numpy as np
import GPy
from  pqueens.designers.monte_carlo_designer import MonteCarloDesigner
from  pqueens.utils import pdf_estimation

class GPEmulator(object):
    """ Class for creating GP based emulators

    This class constructs a GP emulator, currently using a GPy model. Based on
    this emulator various statistical summarys can be computed and returned.

    Attributes:
        X (np.array):
            location of design points
        y (np.array):
            function value at desing points
        parameters (dict):
            Dictionary containing information about the input parameters
            such as their distribution
        m (Gpy.model):
            GPy based Gaussian process model
        num_emulator_samples (int):
            Number of posterior samples to generate
        num_mc_samples (int):
            Number of MC samples to use for evaluation
        y_plot (np.array):
            Y-values to use for returning pdf, cdf, and failure probability

    """
    def __init__(self,X,y,parameters):
        """
        Args:
            X (np.array):
                location of design points
            y (np.array):
                function value at desing points
            parameters (dict):
                Dictionary containing information about the input parameters
                such as their distribution
        """
        self.X = X
        self.y = y
        self.parameters = parameters
         # create simple GP Model
        self.m = GPy.models.GPRegression(self.X, self.y)
        self.m[".*Gaussian_noise"] = self.m.Y.var()*0.01
        self.m[".*Gaussian_noise"].fix()
        self.m.optimize(max_iters = 500)

        self.m[".*Gaussian_noise"].unfix()
        self.m[".*Gaussian_noise"].constrain_positive()

        self.m.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)
        # set the lengthscale to be something sensible (defaults to 1)
        #self.m.kern.lengthscale = 1.



        self.num_emulator_samples = 100
        self.num_mc_samples = 2000

    def predict(self,x_test):
        """ Compute predictions for x_test

        Args:
            x_test (np.array): test locations at which to predict

        Returns:
            np.array,np.array: mean and variance of posterior
            predictive distribution
        """
        mean_x_test, var_x_test = self.m.predict_noiseless(x_test)

        return mean_x_test, var_x_test

    def compute_mean(self):
        """ Compute mean of emulator output w.r.t. x

        Compute mean of emulator output based on the distributions passed
        via the parameters dict. Since the GP is a Bayesian emulator, we
        can also compute the variance of the mean based on several posterior
        samples of the emulator.

        Returns:
            float, float: mean of the mean, variance of the mean

        """
        mean_mean, var_mean, _, _ = self.compute_mean_and_var()
        return mean_mean, var_mean


    def compute_var(self):
        """ Compute variance of emulator output w.r.t. x

        Compute variance of emulator output based on the distributions passed
        via the parameters dict. Since the GP is a Bayesian emulator, we
        can also compute the variance of the mean based on several posterior
        samples of the emulator.

        Returns:
            float, float : mean of the variance, variance of the variance
        """
        _, _, mean_variance, var_variance = self.compute_mean_and_var()
        return mean_variance, var_variance

    def compute_mean_and_var(self):
        """ Compute mean and variance of emulator output w.r.t. inputs x

        Compute variance and mean of emulator output based on the distributions
        passed via the parameters dict. Since the GP is a Bayesian emulator, we
        can also compute the variance of the mean based on several posterior
        samples of the emulator.

        Returns:
            float, float, float, float: mean of the mean, variance of the mean,
            mean of the variance, variance of the variance
        """
        my_evals = self.create_posterior_samples(self.num_emulator_samples,
                                                 self.num_mc_samples)


        my_means = np.mean(my_evals,0)
        # compute mean of mean
        mean_mean = np.mean(my_means)
        # compute variance of mean
        var_mean = np.var(my_means)

        my_variances = np.var(my_evals,0)

        # compute mean of variance
        mean_variance = np.mean(my_variances)
        # compute variance of variance
        var_variance = np.var(mean_variance)

        return mean_mean, var_mean, mean_variance, var_variance

    def compute_pdf(self):
        """ Compute probability density function of emulator output

        This function computes probability density function of the emulator
        output using kernel density estiation. Confindence bounds on the
        pdf are also provided harnessing the Bayesian nature of the emulator.

        Returns:
            dict,np.array: dict with arrays of pdf values at the locations
            stored in the second output argument
        """
        my_evals = self.create_posterior_samples(self.num_emulator_samples,
                                                 self.num_mc_samples)

        min_evals = np.amin(my_evals)
        max_evals = np.amax(my_evals)

        y_plot =  np.linspace(np.amin(my_evals), np.max(my_evals), 100)
        y_density = np.zeros((self.num_emulator_samples,len(y_plot)))

        for i in range(self.num_emulator_samples):
            # estimate optimal kernel bandwidth once
            if i == 0:
                kernel_bandwidth = \
                    pdf_estimation.estimate_bandwidth_for_kde(my_evals[i,:].reshape(-1,1),
                                                              min_evals, max_evals)
            y_density[i,:], _  = \
                pdf_estimation.estimate_pdf(my_evals[i,:].reshape(-1,1),
                                            kernel_bandwidth,y_plot)

        # compute mean median var and quanitiles of pdf
        my_pdf ={}
        my_pdf['mean'] = np.mean(y_density, 0)
        my_pdf['median'] = np.median(y_density, 0)
        my_pdf['var'] = np.var(y_density, 0)
        my_pdf['quant_low'] = np.percentile(y_density, 2.5, 0)
        my_pdf['quant_high'] = np.percentile(y_density, 97.5, 0)

        return my_pdf, y_plot

    def compute_cdf(self):
        """ Compute cumulative density function of emulator output

        Returns:
            np.array,np.array: cdf for values in second array

        """
        sample_values_raw = self.create_posterior_samples(self.num_emulator_samples,
                                                          self.num_mc_samples)

        # sort samples in ascending order first
        sample_values_sorted=np.sort(sample_values_raw,axis=1)
        # average over realiazations
        sample_values = {}
        sample_values['mean'] = np.mean(sample_values_sorted,0)

        # compute cdf
        cdf = np.arange(1, len(sample_values_sorted[0,:])+1) / \
            float(len(sample_values_sorted[0,:]))

        # first compute percentile
        sample_values['q_lower_bound'] = np.percentile(sample_values_sorted,
                                                       2.5, 0)
        # compute median in the same fashion
        sample_values['median'] = np.percentile(sample_values_sorted, 50, 0)

        # compute 97.5 % quantile
        sample_values['q_upper_bound'] = np.percentile(sample_values_sorted,
                                                       97.5, 0)
        return cdf,sample_values

    def compute_failure_probability_function(self):
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
                                                 self.num_mc_samples)
        my_quantile = {}
        y_quantile = np.percentile(my_evals,my_quantiles,axis=1)
        my_quantile['q_lower_bound'] = np.percentile(y_quantile, 2.5, 1)
        my_quantile['q_upper_bound'] = np.percentile(y_quantile, 97.5, 1)
        my_quantile['median'] = np.percentile(y_quantile, 50, 1)
        my_quantile['mean'] = np.mean(y_quantile, 1)

        return my_quantile, 1-my_quantiles/100.0

    def compute_quantile_function(self):
        """ Compute quantile function of emulator output

        Returns (np.array,np.array): array with quantile values and array with
        corresponding quantiles
        """
        my_quantiles = np.linspace(0,100,1000)
        my_evals = self.create_posterior_samples(self.num_emulator_samples,
                                                 self.num_mc_samples)

        my_quantile = {}
        y_quantile = np.percentile(my_evals,my_quantiles,axis=0)
        my_quantile['q_lower_bound'] = np.percentile(y_quantile, 2.5, 1)
        my_quantile['q_upper_bound'] = np.percentile(y_quantile, 97.5, 1)
        my_quantile['median'] = np.percentile(y_quantile, 50, 1)
        my_quantile['mean'] = np.mean(y_quantile, 1)

        return y_quantile, my_quantiles

    def create_posterior_samples(self,num_realizations, num_input_samples):
        """ Generate samples from posterior distribution

        Args:
            num_realizations (float): number of posterior realizations of GP
            num_input_samples (float): number of samples in input dimension

        Returns:
            np.array: samples from posterior predictive distribution

        """
        mc_sampler = MonteCarloDesigner(self.parameters, 43,
                                        num_input_samples)

        mc_samples = mc_sampler.get_all_samples()

        sample_values = self.m.posterior_samples_f(mc_samples,
                                                   num_realizations)


        return np.transpose(sample_values)
