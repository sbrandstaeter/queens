import time
import autograd.numpy as np
import pprint
from autograd import grad
from scipy.stats import multivariate_normal as mvn

from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.process_outputs import write_results
from pqueens.database.mongodb import MongoDB
from pqueens.utils import mcmc_utils
import pqueens.visualization.variational_inference_visualization as vis


class BBVIIterator(Iterator):
    """
    Black box Bayesian variational inference (BBVI) iterator for Bayesian inverse problems.
    BBVI does not require model gradients and can hence be used with any simulation 
    model and without the need for adjoints implementations.
    The algorithm is based on [1].
    The approach is based on a prior mean-field family q. Transformations to unconstrained spaces,
    or to non-Gaussian distributions as well as a reparameterization trick for the covariance
    structure are partly based on [2]. The implementation at hand supports automated
    differentiation using autograd to quickly exchange model densities without the need to
    implement extensive code.

    References:
        [1]: Ranganath, Rajesh, Sean Gerrish, and David M. Blei. "Black Box Variational Inference."
             Proceedings of the Seventeenth International Conference on Artificial Intelligence
             and Statistics. 2014.
        [2]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Args:
        global_settings (dict): Global settings of the QUEENS simulations
        model (obj): Underlying simulation model on which the inverse analysis is conducted
        result_description (dict): Settings for storing and visualizing the results
        db (obj): QUEENS database object
        experiment_name (str): Name of the QUEENS simulation
        min_requ_relative_change_variational_params (float): Minimum required relative change in
                                                             the variational parameters in order
                                                             to reach convergence
        variational_family (str): Density type for variatonal approximation (before transformation)
        variational_approximation_type (str): String determineing mean field or full rank
                                              approximation for the variational distribution
        learning_rate (float): Learning rate of the ADAMS stochastic decent optimizer
        n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration to
                                  estimate the involved expectations)
        variational_transformation (str): String encoding the transformation that will be applied to
                                          the variational density
        random_seed (int): Seed for the random number generators
        max_feval (int): Maximum number of simulation runs for this analysis

    Attributes:
        result_description (dict): Settings for storing and visualizing the results
        db (obj): QUEENS database object
        experiment_name (str): Name of the QUEENS simulation
        variational_samples (np.array): Matrix containing samples/current batch of the
                                        variational distribution
        variational_params (np.array): (Latent) Parameters / Hyperparameters of the variational
                                       distribution and the likelihood model.
        variational_family (str): Type of distribution used in the variational approach (before
                                  transformation)
        variational_distribution_obj (obj): Instance of the variational distribution
        noisy_gradient_ELBO (np.array): Gradient Monte-Carlo Estimate of the ELBO
        f_mat (np.array): Control variate matrix (each sample corresponds to one row, dimensions
                          for Rao-Blackwellization per column)
        h_mat (np.array): Control variate matrix (each sample corresponds to one row, dimensions
                          for Rao-Blackwellization per column)
        control_variate_scales_vec (np.array): Scaling factors for the control variates
        relative_change_variational_params (float): Relative change of the variational
                                                    parameters for the last iterations
                                                    of BBVI
        min_requ_relative_change_variational_params (float): Minimum required relative change in
                                                             the variational parameters in order
                                                             to reach convergence
        variational_approximation_type (str): String determining full rank or mean field
                                              approximation for the variational distribution
        learning_rate (float): Learning rate of the ADAMS stochastic decent optimizer
        n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration to
                                  estimate the involved expectations)
        variational_transformation (str): String encoding the transformation that will be applied to
                                          the variational density
        iter (int): Number of iterations for the BBVI algorithm
        log_likelihood_vec (np.array): Log-likelihood evaluation per sample-point as vector
        elbo_rb_vec (np.array): Vector with Rao-Blackwellized (component-wise per variational
                                parameter) entries of the ELBO
        grad_elbo_cv (np.array): Control-variate estimate of the ELBO gradient for the
                                 variational parameters
        grad_log_posterior (np.array): Gradient (for variational parameters) of the log-posterior
                                       distribution
        grad_log_variational (np.array): Gradient (for variational parameters) of the variational
                                         distribution
        log_variational_mat (np.array): Matrix containing different Rao-Blackwell dimensions row
                                        for each random input variable (not per parameter of the
                                        variational distribution). The matrix columns hold the
                                        realizations for each sample.
        elbo (float): Evidence Lower Bound
        log_posterior_unnormalized (np.array): Unnormalized logarithmic posterior (prior x
                                               likelihood) per sample
        v_param_adams (np.array): Parameter in the ADAMS stochastic optimizer
        m_param_adams (np.array): Parameter in the ADAMS stochastic optimizer
        random_seed (int): Seed for the random number generators
        prior_obj_list (list): List containing objects of prior models for the random input. The
                               list is ordered in the same way as the random input definition in
                               the input file
        max_feval (int): Maximum number of simulation runs for this analysis
        min_relative_elbo_change (float): Threshold for the relative change in the ELBO before
                                          convergence is assumed
        num_variables (int): Actual number of model input variables that should be calibrated
        grad_variational_obj (obj): Autograd object for the variational gradient w.r.t
                                    variational parameters

    Returns:
        bbvi_obj (obj): Instance of the BBVIIterator

    """

    def __init__(
        self,
        global_settings,
        model,
        result_description,
        db,
        experiment_name,
        min_requ_relative_change_variational_params,
        variational_family,
        variational_approximation_type,
        learning_rate,
        n_samples_per_iter,
        variational_transformation,
        random_seed,
        max_feval,
        num_variables,
    ):
        super().__init__(model, global_settings)

        self.result_description = result_description
        self.db = db
        self.experiment_name = experiment_name
        self.variational_samples = None
        self.variational_params = []
        self.variational_family = variational_family
        self.variational_distribution_obj = None
        self.noisy_gradient_ELBO = None
        self.f_mat = None
        self.h_mat = None
        self.control_variate_scales_vec = None
        self.relative_change_variational_params = []
        self.min_requ_relative_change_variational_params = (
            min_requ_relative_change_variational_params
        )
        self.variational_approximation_type = variational_approximation_type
        self.learning_rate = learning_rate
        self.n_samples_per_iter = n_samples_per_iter
        self.variational_transformation = variational_transformation
        self.iter = 0
        self.log_likelihood_vec = None
        self.elbo_rb_vec = None
        self.grad_elbo_cv = None
        self.grad_variational_obj = None
        self.grad_log_posterior = None
        self.grad_log_variational = None
        self.log_variational_mat = None
        self.elbo = [1, 1, 1]
        self.log_posterior_unnormalized = None
        self.v_param_adams = None  # stochastic_ascent_adam optimizer
        self.m_param_adams = None  # stochastic_ascent_adam optimizer
        self.random_seed = random_seed
        self.prior_obj_list = []
        self.max_feval = max_feval
        self.num_variables = num_variables

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """
        Create black box variational inference iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator (obj): BBVI object
        """

        if iterator_name is None:
            method_options = config['method']['method_options']
        else:
            method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = Model.from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        db = MongoDB.from_config_create_database(config)
        experiment_name = config['global_settings']['experiment_name']
        relative_change_variational_params = method_options.get(
            'min_relative_change_variational_params', 0.01
        )
        variational_family = method_options.get("variational_family")
        variational_approximation_type = method_options.get("variational_approximation_type")
        learning_rate = method_options.get("learning_rate")
        n_samples_per_iter = method_options.get("n_samples_per_iter")
        variational_transformation = method_options.get("variational_transformation")
        random_seed = method_options.get("random_seed")
        max_feval = method_options.get("max_feval")
        num_variables = len(model.variables[0].variables)

        vis.from_config_create(config)

        # initialize objective function
        return cls(
            global_settings=global_settings,
            model=model,
            result_description=result_description,
            db=db,
            experiment_name=experiment_name,
            min_requ_relative_change_variational_params=relative_change_variational_params,
            variational_family=variational_family,
            variational_approximation_type=variational_approximation_type,
            learning_rate=learning_rate,
            n_samples_per_iter=n_samples_per_iter,
            variational_transformation=variational_transformation,
            random_seed=random_seed,
            max_feval=max_feval,
            num_variables=num_variables,
        )

    def core_run(self):
        """
        Core run for black-box variational inference
        """
        print('Starting black box Bayesian variational inference...')
        start = time.time()
        # --------------------------------------------------------------------
        # -------- here comes the bbvi algorithm -----------------------------
        while (self._check_convergence()) and (
            (self.iter * self.n_samples_per_iter) <= self.max_feval
        ):
            self._catch_non_converging_simulations()
            variational_params = np.array(self.variational_params)
            self.calculate_grad_log_variational_rb(variational_params)
            self.calculate_grad_elbo_cv()
            self._update_variational_params()
            self._calculate_elbo()
            self._verbose_output()
            self.iter += 1
        # --------- end of bvvi algorithm ------------------------------------
        # --------------------------------------------------------------------
        end = time.time()

        print(f"Finished sucessfully! :-)")
        print(f"Black box variational inference took {end-start} seconds.")
        print(f"---------------------------------------------------------")
        print(f"Cost of the analysis: {self.iter*self.n_samples_per_iter} simulation runs")
        print(f"Final ELBO: {self.elbo[-1]}")
        print(f"---------------------------------------------------------")
        print(f"Post run: Finishing, saving and cleaning...")

    def _check_convergence(self):
        """
        Check the convergence criterion for the BBVI iterator.

        Returns:
            convergence_bool (bool): True if not yet converged. False if converged

        """
        if self.iter > 5:
            convergence_bool = (
                np.any(
                    np.mean(self.relative_change_variational_params[-5:], axis=0)
                    > self.min_requ_relative_change_variational_params
                )
            ) and not np.any(np.isnan(self.relative_change_variational_params[-1]))
        else:
            convergence_bool = True
        return convergence_bool

    def _catch_non_converging_simulations(self):
        """
        Reset variational parameters in case of non-converging simulation runs

        Returns:
            None

        """
        if len(self.relative_change_variational_params) > 0:
            if np.any(np.isnan(self.relative_change_variational_params[-1])):
                self.variational_params = list(self.variational_params_array[:, -2])
                self.variational_distribution_obj.update_distribution_params(
                    self.variational_params
                )

    def initialize_run(self):
        """
        Initialize BBVI object with experimental data and the models for

        - the underlying likelihood model
        - the variational distribution model
        - the underlying prior model

        Returns:
            None

        """
        print("Initialize Optimization run.")
        self._initialize_variational_distribution()
        self._initialize_prior_model()

    def post_run(self):
        """
        Write results and potentially visualize them using the visualization module.

        Returns:
            None
        """
        if self.result_description["write_results"]:
            result_dict = self._prepare_result_description()
            write_results(
                result_dict,
                self.global_settings["output_dir"],
                self.global_settings["experiment_name"],
            )
        vis.vi_visualization_instance.save_plots()

    def eval_model(self):
        """
        Evaluate model for the sample batch

        Returns:
           result_dict (dict): Dictionary containing model response for sample batch

        """
        result_dict = self.model.evaluate()
        return result_dict

    def _initialize_prior_model(self):
        """
        Initialize the prior model of the inverse problem form the problem description.

        Returns:
            None

        """
        # Read design of prior
        input_dict = self.model.get_parameter()
        # get random variables
        random_variables = input_dict.get('random_variables', None)
        # get random fields
        random_fields = input_dict.get('random_fields', None)

        if random_fields:
            raise NotImplementedError(
                'Variational inference for random fields is not yet implemented! Abort...'
            )
        # for each rv, evaluate prior all corresponding dims in current sample batch (column of
        # mat corresponds to all realization for this variable)
        for dim, rv_options in enumerate(random_variables.values()):
            if rv_options['distribution'] == 'normal' and rv_options['size'] == 1:
                rv_options['distribution_parameter'][1] = 1
            self.prior_obj_list.append(mcmc_utils.create_proposal_distribution(rv_options))

    def eval_log_likelihood(self, sample_batch):
        """
        Calculate the log-likelihood of the observation data. Evaluation of the likelihood model
        for all inputs of the sample batch will trigger the actual forward simulation
        (can be executed in parallel as batch-sequential procedure)

        Args:
            sample_batch (np.array): Sample-batch with samples row-wise

        Returns:
            log_likelihood (np.array): Vector of the log-likelihood function for all input
                                       samples of the current batch

        """
        # The first samples belong to simulation input
        # get simulation output (run actual forward problem)--> data is saved to DB
        sample_batch = self._transform_samples(sample_batch)

        self.model.update_model_from_sample_batch(sample_batch)
        log_likelihood = self.eval_model()

        return log_likelihood

    def get_log_prior(self, sample_batch):
        """
        Construct and evaluate the log prior of the model for current sample batch.

        Args:
            sample_batch (np.array): Sample batch for which the model should be evaluated

        Returns:
            log_prior (np.array): log-prior vector evaluated for current sample batch

        """
        sample_batch_transf = self._transform_samples(sample_batch)
        log_prior_vec = np.zeros((self.n_samples_per_iter, 1))
        for dim, prior_distr in enumerate(self.prior_obj_list):
            log_prior_vec = log_prior_vec + prior_distr.logpdf(sample_batch_transf[:, dim]).reshape(
                -1, 1
            )
        return log_prior_vec

    def get_log_posterior_unnormalized(self, sample_batch):
        """
        Calculate the unnormalized log posterior joint for all samples in batch

        Args:
            sample_batch (np.array): Sample batch for which the model should be evaluated

        Returns:
            unnormalized_log_posterior (np.array): Values of unnormalized log posterior
                                                   distribution at positions of sample batch

        """
        log_prior = self.get_log_prior(sample_batch)

        # potential transformation of sample takes place in the log_likelihood function
        log_likelihood = self.eval_log_likelihood(sample_batch)
        log_posterior_unnormalized = (log_likelihood + log_prior).T
        return log_posterior_unnormalized

    def _estimate_control_variate_scalings(self):
        """
        Calculate the control variate scaling parameters for each entry of the
        variational gradient

        Returns:
            None

        """
        # iterate and stack results per sample
        f_mat = self.grad_log_variational_mat[:, 0 : self.n_samples_per_iter] * (
            self.log_posterior_unnormalized
            - self.log_variational_mat[:, 0 : self.n_samples_per_iter]
        )

        dim = len(self.variational_params)

        cov_sum = 0
        var_sum = 0

        h_mat = self.grad_log_variational_mat[:, 0 : self.n_samples_per_iter]
        for f_dim, h_dim in zip(f_mat, h_mat):
            cov_sum += np.cov(f_dim, h_dim)[0, 1]
            var_sum += np.var(h_dim)

        self.control_variate_scales_vec = np.ones((dim, 1)) * cov_sum / var_sum

    def _verbose_output(self):
        """
        Give some informative outputs during the BBVI iterations

        Returns:
            None

        """
        if self.iter % 10 == 0:
            mean_change = np.mean(np.abs(self.relative_change_variational_params[-1]), axis=0) * 100
            print("------------------------------------------------------------------------")
            print(f"Iteration {self.iter + 1} of BBVI algorithm")
            print(f"So far {self.iter * self.n_samples_per_iter} simulation runs")
            print(f"The elbo is: {self.elbo[-1]:.2f}")
            print(
                f"Mean absolute percentage change of all variational parameters: "
                f""
                f"{mean_change:.2f} %"
            )
            print(f"Values of variational parameters: \n")
            pprint.pprint(self.variational_params)
            print("------------------------------------------------------------------------")

    def stochastic_ascent_adam(self, gradient_estimate_x, x_vec, b1=0.915, b2=0.999, eps=10 ** -8):
        """
        Stochastic gradient ascent algorithm ADAM.
        Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
        It's basically RMSprop with momentum and some correction terms.

        Args:
            gradient_estimate_x (np.array): (Noisy) Monte-Carlo estimate of a gradient
            x_vec (np.array): Current position in the latent design space
            b1 (float): The exponential decay rate for the first moment estimates
            b2 (float): The exponential decay rate for the second-moment estimates
            eps (float): Is a very small number to prevent any division by zero in the
                         implementation

        Returns:
            x_vec_new (lst): Updated vector/point in the latent design space

        """
        x_vec = x_vec.reshape(-1, 1)
        g = gradient_estimate_x
        FIM = self._get_FIM()
        g = np.linalg.solve(FIM, g)
        self.m_param_adams = (1 - b1) * g + b1 * self.m_param_adams  # First  moment estimate.
        self.v_param_adams = (1 - b2) * (
            g ** 2
        ) + b2 * self.v_param_adams  # Second moment estimate.
        mhat = self.m_param_adams / (1 - b1 ** (self.iter + 1))  # Bias correction.
        vhat = self.v_param_adams / (1 - b2 ** (self.iter + 1))
        x_vec_new = x_vec + self.learning_rate * mhat / (np.sqrt(vhat) + eps)  # update
        # TODO we should change the data type to np.array
        return list(x_vec_new)

    def _update_variational_params(self):
        """
        Update the variational parameters of the variational distribution based on learning rate
        rho_to and the noisy ELBO gradients

        Returns:
            None

        """
        self.variational_params_array = np.hstack(
            (self.variational_params_array, np.array(self.variational_params).reshape(-1, 1))
        )
        old_variational_params = np.array(self.variational_params)

        # Use Adam for stochastic optimization
        variational_params = self.stochastic_ascent_adam(
            self.grad_elbo_cv, np.array(self.variational_params)
        )

        self.variational_params = list(variational_params)
        self._get_percentage_change_params(old_variational_params)

        # TODO: Not sure if below is necessary as we hand over params explicitly now
        self.variational_distribution_obj.update_distribution_params(self.variational_params)

        cov = np.array(self.variational_params[self.num_variables : 2 * self.num_variables])
        mean = np.array(self.variational_params[: self.num_variables])

        # visualize the parameter updates
        vis.vi_visualization_instance.plot_gaussian_pdfs_params(
            mean, cov, self.variational_transformation
        )

    def _get_percentage_change_params(self, old_variational_params):
        """
        Calculate L2 norm of the percentage change of the variational parameters

        Args:
            old_variational_params (np.array): Array of variational parameters

        Returns:
            None

        """
        rel_distance_vec = np.divide(
            (np.array(self.variational_params).flatten() - old_variational_params.flatten()),
            old_variational_params.flatten(),
        )
        if len(self.relative_change_variational_params) > 0:
            self.relative_change_variational_params.append(rel_distance_vec)
        else:
            self.relative_change_variational_params = [rel_distance_vec]

        if np.any(np.isnan(self.relative_change_variational_params[-1])):
            self.relative_change_variational_params.append(1)  # dummy value to redo iteration

    def _initialize_variational_distribution(self):
        """
        Initialize the variational distribution with parameter values from the prior distribution.

        Returns:
            None

        """
        # Read design of prior
        input_dict = self.model.get_parameter()
        # get random variables
        random_variables = input_dict.get('random_variables', None)
        # get random fields
        random_fields = input_dict.get('random_fields', None)
        if random_fields:
            raise NotImplementedError(
                'Variational inference for random fields is not yet implemented! Abort...'
            )

        # Initialize the variational parameters based on prior distribution and input config
        self._initialize_variational_params(random_variables)
        if self.variational_family == "normal":
            if self.variational_approximation_type == 'mean_field':
                mean_list = self.variational_params[: self.num_variables]
                std_list = self.variational_params[self.num_variables : 2 * self.num_variables]
                distribution_options = {
                    "distribution": "mean_field_normal",
                    "distribution_parameter": {"mean": mean_list, "standard_deviation": std_list},
                }

                self.variational_distribution_obj = mcmc_utils.create_proposal_distribution(
                    distribution_options
                )

            elif self.variational_approximation_type == 'full_rank':
                # TODO: here we could introduce a sparse approx for the full rank
                raise NotImplementedError(
                    f"A variational full rank approximation is not implemented, " f"yet! Abort..."
                )
            else:
                raise ValueError(
                    f"Your choice for the variational approximation was: "
                    f"{self.variational_approximation_type}. This is not a supported "
                    f"approximation type! Abort..."
                )
        else:
            raise NotImplementedError(
                f"Your input for the variational family was "
                f"{self.variational_family}. This is not an appropriate "
                f"choice! Abort..."
            )

    def _initialize_variational_params(self, random_variables):
        """
        The variational distribution is a Gaussian, that might be transformed in a second step
        such that the actual variational distribution is non-Gaussian. Here we determine suitable
        mean and standard deviation parameters, such the transformed variational distribution
        matches the first and second moment of the prior distributions.

        Args:
            random_variables (dict): Dictionary containing the prior probabilistic description of
            the input variables

        Returns:
            None

        """
        # Get the first and second moments of the prior distributions (here no likelihood model!)
        mean_list_prior = []
        std_list_prior = []
        for rv_options in random_variables.values():
            params = rv_options["distribution_parameter"]
            if rv_options["distribution"] == "normal":
                mean_list_prior.append(params[0])
                std_list_prior.append(params[1])
            elif rv_options["distribution"] == "lognormal":
                mean_list_prior.append(np.exp(params[0] + (params[1] ** 2) / 2))
                std_list_prior.append(mean_list_prior[-1] * np.sqrt(np.exp(params[1] ** 2) - 1))
            elif rv_options["distribution"] == "uniform":
                mean_list_prior.append((params[1] - params[0]) / 2)
                std_list_prior.append((params[1] - params[0]) / np.sqrt(12))
            else:
                distr_type = rv_options["distribution"]
                raise NotImplementedError(
                    f"Your prior type {distr_type} is not supported in"
                    f" the variational approach! Abort..."
                )

        # Set the mean and std-deviation params of the variational distr such that the
        # transformed distribution would match the moments of the prior
        if self.variational_transformation == 'exp':
            mean_list_variational = [
                np.log(E ** 2 / np.sqrt(E ** 2 + S ** 2))
                for E, S in zip(mean_list_prior, std_list_prior)
            ]
            std_list_variational = [
                np.sqrt(np.log(1 + S ** 2 / E ** 2))
                for E, S in zip(mean_list_prior, std_list_prior)
            ]
        elif self.variational_transformation is None:
            mean_list_variational = mean_list_prior
            std_list_variational = std_list_prior
        else:
            raise ValueError(
                f"The transformation type {self.variational_transformation} for the "
                f"variational density is unknown! Abort..."
            )

        # Transform the std lists with log as we take later the exp to map covariance in pos space
        cov_list_variational = np.log(np.array(std_list_variational) ** 2) / 2
        self.variational_params = mean_list_variational + list(cov_list_variational)
        self.variational_params_array = np.empty((len(self.variational_params), 0))

        # Some intermediate initialization for stochastic_ascent_adam optimizer
        self.m_param_adams = np.zeros((len(self.variational_params), 1))
        self.v_param_adams = np.zeros((len(self.variational_params), 1))

    def _transform_samples(self, x_mat):
        """
        Transform samples of the variational distribution according to the specified transformation
        mapping

        Args:
            x_mat (np,.array): Samples of the variational distribution

        Returns:
            x_mat_trans (np.array): Transformed samples of variational distribution

        """
        if self.variational_transformation == 'exp':
            x_mat_trans = np.exp(x_mat)
        elif self.variational_transformation is None:
            x_mat_trans = x_mat
        else:
            raise ValueError(
                f"The transformation type {self.variational_transformation} for the "
                f"variational density is unknown! Abort..."
            )

        return x_mat_trans

    def _prepare_result_description(self):
        """
        Transform the results back the correct space and summarize results in a dictionary that
        will be stored as a pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis

        """
        if self.variational_family == "normal":
            if self.variational_approximation_type == 'mean_field':
                mean = np.array(self.variational_params[: self.num_variables])
                var = np.array(self.variational_params[self.num_variables : 2 * self.num_variables])

                if self.variational_transformation == 'exp':
                    # TODO Update below to new MLE likelihood scheme
                    mu = np.log(mean ** 2 / np.sqrt(var + mean ** 2))
                    sigma = np.sqrt(2 * np.exp(var))
                    sigma = np.sqrt(np.log(1 + var / mean ** 2))
                    distr_type = "lognormal"

                elif self.variational_transformation is None:
                    mu = mean
                    # transform variances
                    sigma = np.sqrt(np.exp(2 * var))
                    distr_type = "normal"

                else:
                    raise ValueError(
                        f"The transformation type {self.variational_transformation} for the "
                        f"variational density is unknown! Abort..."
                    )

                result_description = {
                    "variational_distr": {
                        "type": distr_type,
                        "mu": mu,
                        "sigma": sigma,
                        # "noise_std": self.model.noise_var,  # TODO check if releveant
                    },
                    # "noise_var_mle": self.model.historic_mle_noise_var,  # TODO check if relevant
                    # "noise_var_fit": self.model.historic_fit_noise_var,  # TODO check if relevant
                    "elbo": self.elbo,
                    "iterations": self.iter,
                    "batch_size": self.n_samples_per_iter,
                    "learning_rate": self.learning_rate,
                    "number_of_sim": self.n_samples_per_iter * self.iter,
                }

            elif self.variational_approximation_type == 'full_rank':
                raise NotImplementedError(
                    f"A variational full rank approximation is not implemented, yet! Abort..."
                )
            else:
                raise ValueError(
                    f"Your choice for the variational approximation was: "
                    f"{self.variational_approximation_type}. This is not a supported "
                    f"approximation type! Abort..."
                )
        else:
            raise NotImplementedError(
                f"Your input for the variational family was "
                f"{self.variational_family}. This is not an appropriate "
                f"choice! Abort..."
            )
        return result_description

    def calculate_log_variational_rb(
        self, variational_params, num_likelihood_params, variational_sample
    ):
        """
        Per row are the different Rao-Blackwell (rb) dimensions per random variable (not per
        latent parameter!)
        Per column are the realizations for each sample.

        Args:
            variational_params (np.array): Array of the (latent) parameters for the variational
                                           distribution
            variational_sample (np.array): One sample of the sample batch
            num_likelihood_params (int): Number of additional parameters for the likelihood model

        Returns:
            log_variational_rb (np.array): Vector with logarithmic RB dimensions per random input

        """
        log_variational_rb = self.variational_distribution_obj.log_pdf_rb(
            variational_params, variational_sample
        )
        log_variational_rb = np.vstack((log_variational_rb, np.zeros((num_likelihood_params, 1))))
        return log_variational_rb

    def calculate_log_variational(self, variational_params, x_sample):
        """
        Calculate the log of the variational distribution that will
        later be used to calculate the Rao-Backwellized gradient of the variational distribution.
        Basically this expression is used in the automated differentiation.

        Args:
            variational_params (np.array): Array of (latent) parameters of the variational
                                           distribution
            x_sample (np.array): Sample point in the (latent) design space at which the
                                 variational distribution (or its log) should be evaluated at

        Returns:
            log_variational (np.array): Standard log of variational distribution

        """
        # RB-part from variational distribution
        log_variational = self.variational_distribution_obj.log_pdf_rb_for_grad(
            variational_params, x_sample
        )
        return log_variational

    def calculate_grad_log_variational_rb(self, variational_params):
        """
        Calculate the gradient of the Rao-Blackwellized log_variational expression for each
        dimension of the Rao-Blackwell vector of the using automated differentiation.
        Evaluate the gradient expression at each sample of the sample batch.
        These results will be used in the subsequent control variates expression for the ELBO
        gradient. (See equation (9) in [1]: The gradient expression here corresponds to the
        gradient expression within the expectation operator).

        Args:
            variational_params (np.array): Array with (latent) parameters of the variational
                                           distribution

        Returns:
            None

        """
        # TODO here we hardcode the difference between likelihood variables and variational vars
        #  and assume a mean field approach (2 params per var)--> this should be generalized
        variational_params_without_likelihood = variational_params[0 : 2 * self.num_variables]
        num_likelihood_params = len(variational_params) - len(variational_params_without_likelihood)

        self.grad_log_variational_mat = np.empty((variational_params.size, 0))

        #  Automated differentiation of the Rao-Blackwellized log variational expression
        self.grad_variational_obj = grad(self.calculate_log_variational)

        #  Generate samples from the variational distribution
        #  First we generate an extended sample batch to approx the expectation
        #  for terms not dependent on the model. A subset of size n_samples_per_iter is then
        #  selected for the likelihood expression
        num_extended_samples = 100  # TODO: This is hard coded atm but could be pulled out
        sample_batch_extended = self.variational_distribution_obj.draw(
            variational_params_without_likelihood, num_extended_samples
        )
        self.log_posterior_unnormalized = self.get_log_posterior_unnormalized(
            sample_batch_extended[0 : self.n_samples_per_iter]
        )

        #  Evaluate the grad_variational for each sample in the sample batch
        self.evaluate_variational_distribution_for_batch(
            sample_batch_extended, variational_params_without_likelihood, num_likelihood_params,
        )

        self._filter_failed_simulations()

    def evaluate_variational_distribution_for_batch(
        self, sample_batch_extended, variational_params_without_likelihood, num_likelihood_params,
    ):
        """
        Evaluate expressions that only depend on the variational density for the extended sample
        batch

        Args:
            sample_batch_extended (np.array): The extended sample batch that we use for
                                              Monte-Carlo estimation of variational expectations
            variational_params_without_likelihood (np.array): Parameters of the variational density
            num_likelihood_params (int): Number of parameters in the likelihood model

        Returns:
            grad_log_variational_mat (np.array): Matrix containing the gradient w.r.t.
                                                 variational parameters of the different
                                                 Rao-Blackwell dimensions (row for parameter
                                                 gradient). The matrix columns hold the
                                                 realizations for each sample of the
                                                 extended sample batch
            log_variational_rb (np.array): Matrix containing the log of the variational
                                                 distribution with different Rao-Blackwell
                                                 dimensions per row (for each random input variable
                                                 not per parameter of the variational
                                                 distribution). The matrix columns hold the
                                                 realizations for each sample of the extended
                                                 sample batch

        """
        num_variational_params = variational_params_without_likelihood.size
        grad_log_variational_mat = np.empty((num_variational_params, 0))
        log_variational_rb = np.empty((num_variational_params, 0))
        for sample in sample_batch_extended:
            # stacked within fun: here we add also additional dimension for the likelihood params
            log_variational_rb = np.hstack(
                (
                    log_variational_rb,
                    self.calculate_log_variational_rb(
                        variational_params_without_likelihood, num_likelihood_params, sample
                    ),
                )
            )
            # calculate the gradient of the variational distribution (without zero entries for
            # likelihood parameters
            grad_variational_arg = self.grad_variational_obj(
                variational_params_without_likelihood.squeeze(), sample
            ).reshape(-1, 1)
            # add zero entries for the likelihood parameters here
            grad_variational_arg = np.vstack(
                (grad_variational_arg, np.zeros((num_likelihood_params, 1)))
            )
            # stacked here
            grad_log_variational_mat = np.hstack((grad_log_variational_mat, grad_variational_arg))

        if self.log_variational_mat is None:
            self.log_variational_mat = log_variational_rb
        else:
            self.log_variational_mat = np.hstack((self.log_variational_mat, log_variational_rb))

        if self.grad_log_variational_mat is None:
            self.grad_log_variational_mat = grad_log_variational_mat
        else:
            self.grad_log_variational_mat = np.hstack(
                (self.grad_log_variational_mat, grad_log_variational_mat)
            )

    def _filter_failed_simulations(self):
        """
        Filter samples and expressions that are associated with failed simulations

        Returns:
            None

        """
        self.log_variational_mat = self.log_variational_mat.T[0 : self.n_samples_per_iter, :][
            ~np.isnan(self.log_posterior_unnormalized).any(axis=0)
        ].T
        self.grad_log_variational_mat = self.grad_log_variational_mat.T[
            0 : self.n_samples_per_iter, :
        ][~np.isnan(self.log_posterior_unnormalized).any(axis=0)].T
        self.log_posterior_unnormalized = self.log_posterior_unnormalized.T[
            ~np.isnan(self.log_posterior_unnormalized).any(axis=0)
        ].T

    def calculate_grad_elbo_cv(self):
        """
        Calculate the gradient of the ELBO using control variates and the previous
        Rao-Blackwellization of the log-variational gradient.

        Returns:
            None

        """
        self._estimate_control_variate_scalings()

        self.grad_elbo_cv = np.mean(
            self.grad_log_variational_mat[:, 0 : self.n_samples_per_iter]
            * self.log_posterior_unnormalized,
            axis=1,
        ).reshape(-1, 1) + np.mean(
            -self.grad_log_variational_mat * self.log_variational_mat
            - self.control_variate_scales_vec * self.grad_log_variational_mat,
            axis=1,
        ).reshape(
            -1, 1
        )

    def _calculate_elbo(self):
        """
        Calculate the ELBO of the current variational approximation

        Returns:
            None

        """
        mu = np.array(self.variational_params[: self.num_variables])
        cov = np.diag(
            np.exp(
                2 * np.array(self.variational_params[self.num_variables : 2 * self.num_variables])
            ).flatten()
        )
        elbo = mvn.entropy(mu.flatten(), cov) + np.mean(self.log_posterior_unnormalized)
        self.elbo.append(elbo)

        # clear internal variables
        self.log_variational_mat = None
        self.log_posterior_unnormalized = None
        self.grad_log_variational_mat = None

        # some plotting and output
        iterations = np.arange(self.iter + 1)
        vis.vi_visualization_instance.plot_convergence(
            iterations,
            self.variational_params_array,
            self.elbo,
            self.relative_change_variational_params,
        )

    def _get_FIM(self):
        """
        Calculate the Fisher information matrix of the current variational approximation

        Returns:
            fisher (np.array): fisher information matrix of the variational distribution

        """
        # TODO generalize this function for non Gaussians using MC
        # TODO generalize damping
        # Damping for the FIM in order to avoid a slowdown in the first few iterations (is
        # useful when the variance of the variational distribution is already small at the start
        # of the optimization)
        damping_lower_bound = 1e-2
        if self.iter > 50:
            damping_coefficient = damping_lower_bound * np.exp(-(self.iter - 50) / 50)
        else:
            damping_coefficient = damping_lower_bound
        fisher_diag = (
            np.exp(-2 * np.array(self.variational_params[self.num_variables :]))
            + damping_coefficient
        )
        fisher_diag = np.append(fisher_diag.flatten(), 2 * np.ones(self.num_variables))
        fisher = np.diag(fisher_diag)
        return fisher
