import numpy as np
import GPy
from sklearn.neighbors import KernelDensity
from  pqueens.designers.monte_carlo_designer import MonteCarloDesigner

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

        # set the lengthscale to be something sensible (defaults to 1)
        self.m.kern.lengthscale = 1.
        # fit the GP
        self.m.optimize('bfgs', max_iters=200)

        # TODO remove hard coded values and pass via contructor argument instead
        self.num_emulator_samples = 500
        self.num_mc_samples = 500
        #self.y_plot =  np.linspace(np.amin(self.y), np.max(self.y), 100)


    def compute_mean(self):
        """ Compute mean of emulator output

            Compute mean of emulator output based on the distributions passed
            vie the parameters dict. Since the GP is a Bayesian emulator, we
            can also compute the variance of the mean based on several posterior
            samples of the emulator.

        Returns (float,float): Mean of the mean and variance of the mean based
                               on emulator uncertainty

        """
        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           self.num_mc_samples)
        my_mc_samples = my_mc_sampler.get_all_samples()

        my_means = self.m.posterior_samples_f(my_mc_samples,
                                              self.num_emulator_samples)
        my_mean = np.mean(my_means,0)
        # compute mean of mean
        mean_mean = np.mean(my_mean)
        # compute variance of mean
        var_mean = np.var(my_mean)
        return mean_mean, var_mean


    def compute_var(self):
        """ Compute variance of emulator output """
        # TODO implement
        raise NotImplementedError

    def compute_quantile(self,quantile):
        """ Compute quantile of emulator output """
        # TODO implement
        raise NotImplementedError


    def compute_quantile_function(self):
        """ Compute quantile function of emulator output

        Returns (np.array,np.array): array with quantile values and array with
                                     corresponding quantiles
        """
        my_quantiles = np.linspace(0,100,1000)
        my_mc_sampler = MonteCarloDesigner(self.parameters, 43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()
        my_evals = self.m.posterior_samples_f(my_mc_samples,
                                              self.num_emulator_samples)

        y_quantile = np.percentile(my_evals,my_quantiles)

        return y_quantile, my_quantiles

    def compute_failure_probability_function(self):
        """ Compute quantile function of emulator output

        This function computed the quantile function of the emulator output
        and includes a confidence regions using the Bayesian nature of the
        emulator.

        Returns (dict,np.array): dict wiht arrays of quantiles values and an
                                 array with corresponding quantiles
        """
        my_quantiles = np.linspace(0,100,1000)
        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        my_evals = self.m.posterior_samples_f(my_mc_samples,
                                              self.num_emulator_samples)
        my_quantile = {}
        y_quantile = np.percentile(my_evals,my_quantiles,axis=0)
        my_quantile['quant_low'] = np.percentile(y_quantile, 2.5, 1)
        my_quantile['quant_high'] = np.percentile(y_quantile, 97.5, 1)
        my_quantile['mean'] = np.mean(y_quantile, 1)

        return my_quantile, 1-my_quantiles/100.0

    def compute_pdf(self):
        """ Compute probability density function of emulator output

        This function computes probability density function of the emulator
        output using kernel density estiation. Confindence bounds on the
        pdf are also provided harnessing the Bayesian nature of the emulator.

        Returns (dict,np.array): dict with arrays of pdf values at the locations
                                stored in the second output argument
        """

        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()
        my_evals = self.m.posterior_samples_f(my_mc_samples,
                                              self.num_emulator_samples)

        min_evals = np.amin(my_evals)
        max_evals = np.amax(my_evals)

        y_plot =  np.linspace(np.amin(my_evals), np.max(my_evals), 100)
        y_density = np.zeros((self.num_emulator_samples,len(y_plot)))
        kernel_bandwidth = 0
        kernel_bandwidth_upper_bound = (max_evals-min_evals)/2.0
        kernel_bandwidth_lower_bound = (max_evals-min_evals)/20.0
        for i in range(self.num_emulator_samples):
            # estimate optimal kernel bandwidth using grid search
            # do this only once
            if i == 0:
                from sklearn.grid_search import GridSearchCV
                 # 20-fold cross-validation
                grid = GridSearchCV(KernelDensity(),{'bandwidth': \
                    np.linspace(kernel_bandwidth_lower_bound,
                                kernel_bandwidth_upper_bound,
                                40)},cv=20)

                grid.fit(my_evals[:,i].reshape(-1,1))
                kernel_bandwidth = grid.best_params_['bandwidth']
                #print("kernel_bandwidth{}".format(kernel_bandwidth))
            kde = KernelDensity(kernel='gaussian', bandwidth = \
                kernel_bandwidth).fit(my_evals[:,i].reshape(-1,1))

            y_density[i,:] = np.exp(kde.score_samples(y_plot.reshape(-1,1)))

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

        Returns (np.array,np.array):
             cdf for values in second array

        """
        # TODO provide error bounds based on posterior samples
        my_mc_sampler = MonteCarloDesigner(self.parameters, 43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()
        #my_evals = self.m.posterior_samples_f(my_mc_samples,self.num_emulator_samples)
        my_evals = self.m.posterior_samples_f(my_mc_samples,1)
        my_evals.sort(axis=0)
        ys = np.arange(1, len(my_evals)+1)/float(len(my_evals))
        return ys, my_evals

    def compute_posterior_samples(self,X_test):
        """ Compute samples of posterior for X_test """
        my_evals = self.m.posterior_samples_f(X_test,self.num_emulator_samples)
        return my_evals
