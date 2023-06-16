"""Multi-fidelity Gaussian likelihood model."""

import logging

import numpy as np

import pqueens.visualization.bmfia_visualization as qvis
from pqueens.distributions.mean_field_normal import MeanFieldNormalDistribution
from pqueens.interfaces import from_config_create_interface
from pqueens.iterators import from_config_create_iterator
from pqueens.models.likelihood_models.likelihood_model import LikelihoodModel
from pqueens.utils.ascii_art import print_bmfia_acceleration

_logger = logging.getLogger(__name__)


class BMFGaussianModel(LikelihoodModel):
    """Multi fidelity likelihood function.

    Multi-fidelity likelihood of the Bayesian multi-fidelity inverse
    analysis scheme [1, 2].

    Attributes:
        mf_interface (obj): QUEENS multi-fidelity interface
        bmfia_subiterator (obj): Subiterator to select the training data of the
                                probabilistic regression model
        normal_distribution (obj): Mean field normal distribution object
        noise_var (np.array): Noise variance of the observations
        likelihood_counter (int): Internal counter for the likelihood evaluation
        num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                                      dependency in refinement step
        likelihood_evals_for_refinement (lst):  List with necessary number of likelihood
                                                evaluations before the refinement step is
                                                conducted

    Returns:
        Instance of BMFGaussianModel. This is a multi-fidelity version of the
        Gaussian noise likelihood model.


    References:
        [1] Nitzler, J.; Biehler, J.; Fehn, N.; Koutsourelakis, P.-S. and Wall, W.A. (2020),
            "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
            Propagation in Complex Physical Simulations", arXiv:2001.02892

        [2] Nitzler J.; Wall, W. A. and Koutsourelakis P.-S. (2021),
            "An efficient Bayesian multi-fidelity framework for the solution of high-dimensional
            inverse problems in computationally demanding simulations", unpublished internal report
    """

    def __init__(
        self,
        model_name,
        forward_model,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
        mf_interface,
        bmfia_subiterator,
        mean_field_normal,
        noise_var,
        num_refinement_samples,
        likelihood_evals_for_refinement,
    ):
        """Instantiate the multi-fidelity likelihood class.

        Args:
            model_name (str): Name of the likelihood model in the config file
            forward_model (obj): Forward model to iterate; here: the low fidelity model
            coords_mat (np.array): Matrix with coordinate values for observations
            time_vec (np.array): Vector with time-stamps of observations
            y_obs (np.array): Matrix / vector of observations
            output_label (str): Label / name of the output / QoI in the experimental data file
            coord_labels (str): Labels / names of the coordinates in the experimental data file
            mf_interface (obj): QUEENS multi-fidelity interface
            bmfia_subiterator (obj): Subiterator to select the training data of the
                                    probabilistic regression model
            mean_field_normal (obj): Mean field normal distribution object
            noise_var (np.array): Noise variance of the observations
            num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                                          dependency in refinement step
            likelihood_evals_for_refinement (lst): List with necessary number of likelihood
                                                   evaluations before the refinement step is
                                                   conducted
        """
        super().__init__(
            model_name,
            forward_model,
            coords_mat,
            time_vec,
            y_obs,
            output_label,
            coord_labels,
        )

        self.mf_interface = mf_interface
        self.bmfia_subiterator = bmfia_subiterator
        self.min_log_lik_mf = None
        self.normal_distribution = mean_field_normal
        self.noise_var = noise_var
        self.likelihood_counter = 1
        self.num_refinement_samples = num_refinement_samples
        self.likelihood_evals_for_refinement = likelihood_evals_for_refinement

    @classmethod
    def from_config_create_model(
        cls,
        model_name,
        config,
    ):
        """Configure multi-fidelity likelihood class from problem description.

        Returns:
            BMFGaussianModel (obj): A BMFGaussianModel object
        """
        (
            forward_model,
            coords_mat,
            time_vec,
            y_obs,
            output_label,
            coord_labels,
        ) = super().get_base_attributes_from_config(model_name, config)

        # get model options
        model_options = config[model_name]

        # ---------- create multi-fidelity interface --------------------------------------------
        bmfia_interface_name = model_options.get("interface_name")
        if config[bmfia_interface_name]["type"] != "bmfia":
            raise ValueError("The interface type must be 'bmfia' for BMFGaussianModel!")
        bmfia_interface = from_config_create_interface(bmfia_interface_name, config)

        # ----------------------- create subordinate bmfia iterator ------------------------------
        bmfia_iterator_name = model_options["mf_subiterator_name"]
        approx_name = model_options["mf_approx_name"]
        bmfia_subiterator = from_config_create_iterator(config, bmfia_iterator_name)

        # ----------------------- initialize the mean field normal distribution ------------------
        noise_variance = np.array(model_options.get("noise_value"))
        dimension = y_obs.size

        num_refinement_samples = model_options.get("num_refinement_samples")
        likelihood_evals_for_refinement = model_options.get("likelihood_evals_for_refinement")

        # build distribution with dummy values; parameters might change during runtime
        mean_field_normal = MeanFieldNormalDistribution(
            mean=y_obs, variance=noise_variance, dimension=dimension
        )

        # ---------------------- initialize some model settings/train surrogates -----------------
        BMFGaussianModel.initialize_bmfia_iterator(coords_mat, time_vec, y_obs, bmfia_subiterator)
        BMFGaussianModel._build_approximation(
            bmfia_subiterator,
            bmfia_interface,
            config,
            approx_name,
            coord_labels,
            time_vec,
            coords_mat,
        )

        # ----------------------- create visualization object(s) ---------------------------------
        plotting_options = config[model_name].get("plotting_options", {})
        qvis.from_config_create(plotting_options)

        return cls(
            model_name,
            forward_model,
            coords_mat,
            time_vec,
            y_obs,
            output_label,
            coord_labels,
            bmfia_interface,
            bmfia_subiterator,
            mean_field_normal,
            noise_variance,
            num_refinement_samples,
            likelihood_evals_for_refinement,
        )

    def evaluate(self, samples):
        """Evaluate multi-fidelity likelihood.

        Evaluation with current set of variables
        which are an attribute of the underlying low-fidelity simulation model.

        Args:
            samples (np.ndarray): Evaluated samples

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        # reshape the model output according to the number of coordinates
        num_coordinates = self.coords_mat.shape[0]
        num_samples = samples.shape[0]

        # we explicitly cut the array at the variable size as within one batch several chains
        # e.g., in MCMC might be calculated; we only want the last chain here
        forward_model_output = self.forward_model.evaluate(samples)['mean'].reshape(
            -1, num_coordinates
        )[:num_samples, :]

        mf_log_likelihood = self.evaluate_from_output(samples, forward_model_output)
        self.response = {
            'forward_model_output': forward_model_output,
            'mf_log_likelihood': mf_log_likelihood,
        }

        return mf_log_likelihood

    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        partial_grad = self.partial_grad_evaluate(samples, self.response['forward_model_output'])
        upstream_gradient = upstream_gradient * partial_grad
        gradient = self.forward_model.grad(samples, upstream_gradient)
        return gradient

    def evaluate_from_output(self, samples, forward_model_output):
        """Evaluate multi-fidelity likelihood from forward model output.

        Args:
            samples (np.ndarray): Samples to evaluate
            forward_model_output (np.ndarray): Forward model output

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        if self._adaptivity_trigger():
            raise NotImplementedError("Adaptivity is not yet implemented for BMFGaussianModel!")

        # evaluate the modified multi-fidelity likelihood expression with LF model response
        mf_log_likelihood = self._evaluate_mf_likelihood(samples, forward_model_output)
        self.likelihood_counter += 1
        return mf_log_likelihood

    def partial_grad_evaluate(self, forward_model_input, forward_model_output):
        """Implement the partial derivative of the evaluate method.

        The partial derivative w.r.t. the output of the sub-model is for example
        required to calculate gradients of the current model w.r.t. to the sample
        input.

        Args:
            forward_model_input (np.array): Sample inputs of the model run (here not required).
            forward_model_output (np.array): Output of the underlying sub- or forward model
                                             for the current batch of sample inputs.

        Returns:
            grad_out (np.array): Evaluated partial derivative of the evaluation function
                                 w.r.t. the output of the underlying sub-model.
        """
        # construct LF feature matrix
        z_mat = self.bmfia_subiterator.set_feature_strategy(
            forward_model_output,
            forward_model_input,
            self.coords_mat[: forward_model_output.shape[0]],
        )

        # Get the response matrices of the multi-fidelity mapping
        m_f_mat, var_y_mat, grad_m_f_mat, grad_var_y_mat = self.mf_interface.evaluate_and_gradient(
            z_mat
        )

        if grad_m_f_mat.ndim == 3:
            grad_m_f_mat = grad_m_f_mat[:, :, 0]  # extract only derivative w.r.t. to LF output
            grad_var_y_mat = grad_var_y_mat[:, :, 0]  # extract only derivative w.r.t. to LF output

        assert np.array_equal(
            m_f_mat.shape[1], np.atleast_2d(self.y_obs).shape[1]
        ), "Column dimension of the probab. regression output and y_obs do not agree!"

        # here we iterate over samples meaning we
        # iterate here over all surrogates simultaneously such that
        # the new m is a vector of all e.g. first entries in all surrogates
        log_lik_mf_lst = []
        grad_log_lik_lst = []
        for m_f_vec, variance_vec, grad_m_f, grad_var_y in zip(
            m_f_mat, var_y_mat, grad_m_f_mat, grad_var_y_mat, strict=True
        ):
            self.normal_distribution.update_variance(
                variance_vec.flatten() + self.noise_var.flatten()
            )
            log_lik_mf_lst.append(self.normal_distribution.logpdf(m_f_vec.reshape(1, -1)))
            grad_log_lik_lst.append(
                self.grad_log_pdf_d_ylf(m_f_vec, grad_m_f, grad_var_y).flatten()
            )

        log_lik_mf_output = np.array(log_lik_mf_lst).reshape(-1, 1)
        grad_out = np.array(grad_log_lik_lst)

        if self.min_log_lik_mf is None:
            self.min_log_lik_mf = np.min(log_lik_mf_output)

        return grad_out

    def _adaptivity_trigger(self):
        """Triggers adaptive refinement for the m_f_likelihood."""
        if self.likelihood_evals_for_refinement:
            if self.likelihood_counter in self.likelihood_evals_for_refinement:
                return True
        else:
            return False

    def _refine_mf_likelihood(self, additional_x_train, additional_y_lf_train=None):
        """Refine multi-fidelity likelihood.

        Args:
            additional_x_train (np.array): New input training points.
            additional_y_lf_train (np.array, optional): New output training points.
                                                        Defaults to None.
        """
        z_train, y_hf_train = self.bmfia_subiterator.expand_training_data(
            additional_x_train, additional_y_lf_train=additional_y_lf_train
        )
        _logger.info('Start updating the probabilistic model...')
        self.mf_interface.build_approximation(z_train, y_hf_train)
        _logger.info("---------------------------------------------------------------------")
        _logger.info('Probabilistic model was updated successfully!')
        _logger.info("---------------------------------------------------------------------")

    def _evaluate_mf_likelihood(self, x_batch, y_lf_mat):
        """Evaluate the Bayesian multi-fidelity likelihood as described in [1].

        Args:
            x_batch (np.array): Input batch matrix; rows correspond to one input vector;
                                different dimensions along columns

            y_lf_mat (np.array): Response matrix of the low-fidelity model; Row-wise corresponding
                                 to rows in x_batch input batch matrix. Different coordinate
                                 locations along the columns

        Returns:
            log_lik_mf_output (tuple): Tuple with vector of log-likelihood values
                                       per model input and potentially the gradient
                                       of the model w.r.t. its inputs


        References:
            [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
                "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
                Propagation in Complex Physical Simulations", arXiv:2001.02892
        """
        # construct LF feature matrix
        z_mat = self.bmfia_subiterator.set_feature_strategy(
            y_lf_mat, x_batch, self.coords_mat[: y_lf_mat.shape[0]]
        )
        # Get the response matrices of the multi-fidelity mapping
        m_f_mat, var_y_mat = self.mf_interface.evaluate(z_mat)
        assert np.array_equal(
            m_f_mat.shape[1], np.atleast_2d(self.y_obs).shape[1]
        ), "Column dimension of the probab. regression output and y_obs do not agree! Abort..."

        # iterate here over all surrogates simultaneously such that the
        # new m is a vector of all, e.g., first entries in all surrogates
        log_lik_mf_lst = []
        for m_f_vec, variance_vec in zip(m_f_mat, var_y_mat, strict=True):
            self.normal_distribution.update_variance(
                variance_vec.flatten() + self.noise_var.flatten()
            )
            log_lik_mf_lst.append(self.normal_distribution.logpdf(m_f_vec.reshape(1, -1)))
        log_lik_mf_output = np.array(log_lik_mf_lst).reshape(-1, 1)

        if self.min_log_lik_mf is None:
            self.min_log_lik_mf = np.min(log_lik_mf_output)

        return log_lik_mf_output

    def grad_log_pdf_d_ylf(self, m_f_vec, grad_m_f_dy, grad_var_y_dy):
        """Calculate the gradient of the logpdf w.r.t. to the LF model output.

        The gradient is calculated from the individual partial derivatives
        and then composed in this method.

        Args:
            m_f_vec (np.array): mean vector of the probabilistic surrogate evaluated at sample
                                points
            grad_m_f_dy (np.array): gradient of the mean function/vector of the probabilistic
                                 regression model w.r.t. the regression model's input
            grad_var_y_dy (np.array): gradient of the variance function/vector of the probabilistic
                                   regression model w.r.t. the regression model's input

        Returns:
            d_log_lik_d_z (np.array): gradient of the logpdf w.r.t. y_lf
        """
        d_log_lik_d_m_f = self.normal_distribution.grad_logpdf(m_f_vec).reshape(1, -1)
        d_log_lik_d_var = self.normal_distribution.grad_logpdf_var(m_f_vec).reshape(1, -1)

        d_log_lik_d_y = d_log_lik_d_m_f * grad_m_f_dy.reshape(
            1, -1
        ) + d_log_lik_d_var * grad_var_y_dy.reshape(1, -1)

        return d_log_lik_d_y

    @staticmethod
    def initialize_bmfia_iterator(coords_mat, time_vec, y_obs, bmfia_subiterator):
        """Initialize the bmfia iterator.

        Args:
            coords_mat (np.array): Coordinates of the experimental data.
            time_vec (np.array): Time vector of the experimental data.
            y_obs (np.array): Experimental data observations at coordinates
            bmfia_subiterator (bmfia_subiterator): BMFIA subiterator object.
        """
        _logger.info("---------------------------------------------------------------------")
        _logger.info("Speed-up through Bayesian multi-fidelity inverse analysis (BMFIA)!")
        _logger.info("---------------------------------------------------------------------")
        print_bmfia_acceleration()

        bmfia_subiterator.coords_experimental_data = coords_mat
        bmfia_subiterator.time_vec = time_vec
        bmfia_subiterator.y_obs = y_obs

    @staticmethod
    def _build_approximation(
        bmfia_subiterator,
        bmfia_interface,
        config,
        approx_name,
        coord_labels,
        time_vec,
        coords_mat,
    ):
        """Construct the probabilistic surrogate / mapping.

        Surrogate is calculated based on the provided training-data and
        optimize the hyper-parameters by maximizing the data's evidence
        or its lower bound (ELBO).

        Args:
            bmfia_subiterator (bmfia_subiterator): BMFIA subiterator object.
            bmfia_interface (bmfia_interface): BMFIA interface object.
            config (dict): Configuration dictionary.
            approx_name (str): Name of the approximation for probabilistic mapping.
            coord_labels (list): List of coordinate labels.
            time_vec (np.array): Time vector of the experimental data.
            coords_mat (np.array): (Spatial) Coordinates of the experimental data.
        """
        # Start the bmfia (sub)iterator to create the training data for the probabilistic mapping
        z_train, y_hf_train = bmfia_subiterator.core_run()
        # ----- train regression model on the data ----------------------------------------
        bmfia_interface.build_approximation(
            z_train, y_hf_train, config, approx_name, coord_labels, time_vec, coords_mat
        )
        _logger.info("---------------------------------------------------------------------")
        _logger.info('Probabilistic model was built successfully!')
        _logger.info("---------------------------------------------------------------------")
