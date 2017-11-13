import numpy as np
import GPy
from sklearn.neighbors import KernelDensity
from  pqueens.designers.monte_carlo_designer import MonteCarloDesigner

class MF_ICM_GPEmulator(object):
    """ Class for creating GP based emulators

    This class constructs a multi fidelity GP emulator, currently using a GPy
    model. Based on this emulator various statistical summarys can be
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
        m (Gpy.model):
            GPy based Gaussian process model
        num_emulator_samples (int):
            Number of posterior samples to generate
        num_mc_samples (int):
            Number of MC samples to use for evaluation
        y_plot (np.array):
            Y-values to use for returning pdf, cdf, and failure probability

    """
    def __init__(self,X_lofi,X_hifi,y_lofi,y_hifi,num_fidelity_levels,parameters):
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
            num_fidelity_levels (int):
                number of fidelity levels
            parameters (dict):
                Dictionary containing information about the input parameters
                such as their distribution
        """
        # TODO extend this to an arbitrary number of levels

        self.X_lofi = X_lofi
        self.y_lofi = y_lofi

        self.X_hifi = X_hifi
        self.y_hifi = y_hifi

        self.num_fidelity_levels = num_fidelity_levels
        self.parameters = parameters

        # check that X_lofi and X_hifi have the same dimension
        dim_x = X_hifi.shape[1]
        if dim_x is not X_lofi.shape[1]:
            raise Exception("Dimension of low fidelity inputs and high fidelity inputs must be the same")

        # define icm multi output kernel
        icm = GPy.util.multioutput.ICM(input_dim=dim_x,
                                       num_outputs=self.num_fidelity_levels,
                                       kernel=GPy.kern.RBF(dim_x))

        self.m = GPy.models.GPCoregionalizedRegression([X_lofi,X_hifi],
                                                       [y_lofi,y_hifi],
                                                       kernel=icm)

        #For this kernel, B.kappa encodes the variance now.
        self.m['.*rbf.var'].constrain_fixed(1.)

        self.m[".*Gaussian_noise"] = self.m.Y.var()*0.01
        self.m[".*Gaussian_noise"].fix()

        self.m.optimize(max_iters = 500)

        self.m[".*Gaussian_noise"].unfix()
        self.m[".*Gaussian_noise"].constrain_positive()
        self.m.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)

        # TODO remove hard coded values and pass via contructor argument instead
        self.num_emulator_samples = 100
        self.num_mc_samples = 500

    def predict(self,x,level):
        """ Compute prediction

        Args:
            x (np.array): location at which to predict
            level (int): levels which to include
        Returns:
            np.array: value of emulator at x
        """
        dim_x = x.shape[1]
        if dim_x is not self.X_lofi.shape[1]:
            raise Exception("Dimension of inputs does not match dimension of emulator")

        if level > self.num_fidelity_levels-1:
            raise Exception("Cannot access level {} since number of levels is {}"\
            .format(self.num_fidelity_levels,level))

        # add level dimension
        my_samples = np.hstack([x, level * np.ones((x.shape[0], 1))])

        my_mean, my_var = self.m.predict_noiseless(my_samples)
        # now idea how and why this works or is needed, see
        # http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb
        noise_dict = {'output_index':my_samples[:,dim_x:].astype(int)}
        my_quantiles_low, my_quantiles_high = self.m.predict_quantiles(my_samples,Y_metadata=noise_dict)

        return my_mean.reshape((-1,1)), my_var.reshape((-1,1)), \
               my_quantiles_low.reshape((-1,1)), my_quantiles_high.reshape((-1,1))

    def compute_mean(self,level):
        """ Compute mean of emulator output

            Compute mean of emulator output based on the distributions passed
            vie the parameters dict. Since the GP is a Bayesian emulator, we
            can also compute the variance of the mean based on several posterior
            samples of the emulator.

        Args:
         level (int): level to compute mean

        Returns (float,float): Mean of the mean and variance of the mean based
                               on emulator uncertainty

        """
        if level > self.num_fidelity_levels-1:
            raise Exception("Cannot access level {} since number of levels is {}"\
            .format(self.num_fidelity_levels,level))

        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        # add level dimension
        my_mc_samples = np.hstack([my_mc_samples,level *
                                   np.ones((self.num_mc_samples, 1))])

        my_means = self.m.posterior_samples_f(my_mc_samples,
                                               self.num_emulator_samples)
        my_mean = np.mean(my_means,0)
        # compute mean of mean
        mean_mean = np.mean(my_mean)
        # compute variance of mean
        var_mean = np.var(my_mean)
        return mean_mean, var_mean


    def compute_var(self,level):
        """ Compute variance of emulator output

            Compute variance of emulator output based on the distributions passed
            via the parameters dict. Since the GP is a Bayesian emulator, we
            can also compute the variance of the mean based on several posterior
            samples of the emulator.

        Args:
         level (int):
            level to compute mean

        Returns:
            float,float:
                Mean of the variance and variance of the
                variance mean based on emulator uncertainty

        """

        if level > self.num_fidelity_levels-1:
            raise Exception("Cannot access level {} since number of levels is {}"\
            .format(self.num_fidelity_levels,level))

        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        # add level dimension
        my_mc_samples = np.hstack([my_mc_samples,level *
                                   np.ones((self.num_mc_samples, 1))])

        my_means = self.m.posterior_samples_f(my_mc_samples,
                                              self.num_emulator_samples)
        my_variances = np.var(my_means,0)

        # compute mean of variance
        mean_variance = np.mean(my_variances)

        # compute variance of variance
        var_variance = np.var(mean_variance)
        return mean_variance, var_variance

    def compute_mean_and_var(self,level):
        pass
        # TODO implement


    def compute_pdf(self,level):
        """ Compute probability density function of emulator output

        This function computes probability density function of the emulator
        output using kernel density estiation. Confindence bounds on the
        pdf are also provided harnessing the Bayesian nature of the emulator.

        Args:
         level (int): level to compute pdf

        Returns:
            dict,np.array: dict with arrays of pdf values at the locations
                                stored in the second output argument
        """
        # TODO generate samples via posterior samples funcion
        if level > self.num_fidelity_levels-1:
            raise Exception("Cannot access level {} since number of levels is {}"\
            .format(self.num_fidelity_levels,level))

        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        # add level dimension
        my_mc_samples = np.hstack([my_mc_samples,level *
                                   np.ones((self.num_mc_samples, 1))])

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

    def compute_cdf(self,level):
        """ Compute cumulative density function of emulator output

        Args:
         level (int): level to compute cdf

        Returns (np.array,np.array):
             cdf for values in second array

        """

        if level > self.num_fidelity_levels-1:
            raise Exception("Cannot access level {} since number of levels is {}"\
            .format(self.num_fidelity_levels,level))
        # TODO generate samples via posterior samples funcion
        # TODO provide error bounds based on posterior samples
        my_mc_sampler = MonteCarloDesigner(self.parameters, 43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        # add level dimension
        my_mc_samples = np.hstack([my_mc_samples,level *
                                   np.ones((self.num_mc_samples, 1))])

        #my_evals = self.m.posterior_samples_f(my_mc_samples,self.num_emulator_samples)
        my_evals = self.m.posterior_samples_f(my_mc_samples,1)
        my_evals.sort(axis=0)
        ys = np.arange(1, len(my_evals)+1)/float(len(my_evals))
        return ys, my_evals


    def compute_failure_probability_function(self,level):
        """ Compute quantile function of emulator output

        This function computed the quantile function of the emulator output
        and includes a confidence regions using the Bayesian nature of the
        emulator.

        Args:
         level (int): level to compute failure probability at

        Returns (dict,np.array): dict wiht arrays of quantiles values and an
                                 array with corresponding quantiles
        """
        # TODO generate samples via posterior samples funcion

        if level > self.num_fidelity_levels-1:
            raise Exception("Cannot access level {} since number of levels is {}"\
            .format(self.num_fidelity_levels,level))

        my_quantiles = np.linspace(0,100,1000)
        my_mc_sampler = MonteCarloDesigner(self.parameters,43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        # add level dimension
        my_mc_samples = np.hstack([my_mc_samples,level *
                                   np.ones((self.num_mc_samples, 1))])

        my_evals = self.m.posterior_samples_f(my_mc_samples,
                                              self.num_emulator_samples)
        my_quantile = {}
        y_quantile = np.percentile(my_evals,my_quantiles,axis=0)
        my_quantile['quant_low'] = np.percentile(y_quantile, 2.5, 1)
        my_quantile['quant_high'] = np.percentile(y_quantile, 97.5, 1)
        my_quantile['mean'] = np.mean(y_quantile, 1)

        return my_quantile, 1-my_quantiles/100.0


    def compute_quantile_function(self,level):
        """ Compute quantile function of emulator output

        Args:
         level (int): level to compute failure probability at

        Returns:
            np.array,np.array: array with quantile values and array with
                                     corresponding quantiles
        """
        # TODO generate samples via posterior samples funcion
        if level > self.num_fidelity_levels-1:
            raise Exception("Cannot access level {} since number of levels is {}"\
            .format(self.num_fidelity_levels,level))

        my_quantiles = np.linspace(0,100,1000)
        my_mc_sampler = MonteCarloDesigner(self.parameters, 43,
                                           self.num_mc_samples)

        my_mc_samples = my_mc_sampler.get_all_samples()

        # add level dimension
        my_mc_samples = np.hstack([my_mc_samples,level *
                                   np.ones((self.num_mc_samples, 1))])

        my_evals = self.m.posterior_samples_f(my_mc_samples,
                                              self.num_emulator_samples)

        y_quantile = np.percentile(my_evals,my_quantiles)

        return y_quantile, my_quantiles

    def create_posterior_samples(self,num_realizations_l1,
                                     num_realizations_l2,num_x_samples):
        pass
        # TODO implement
