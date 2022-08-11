"""Multi-fidelity Gaussian likelihood model."""

import logging

import numpy as np

import pqueens.visualization.bmfia_visualization as qvis
from pqueens.interfaces.bmfia_interface import BmfiaInterface
from pqueens.iterators import from_config_create_iterator
from pqueens.utils.ascii_art import print_bmfia_acceleration

from .likelihood_model import LikelihoodModel

_logger = logging.getLogger(__name__)


class BMFGaussianModel(LikelihoodModel):
    """Multi fidelity likelihood function.

    Multi-fidelity likelihood of the Bayesian multi-fidelity inverse
    analysis scheme [1, 2].

    Args:
        model_name (str): Name of the likelihood model in the config file
        model_parameters (np.array): Parameters of the inverse problem
        nugget_noise_var (float): Lower bound for the likelihood noise
        forward_model (obj): Forward model to iterate; here: the low fidelity model
        coords_mat (np.array): Matrix with coordinate values for observations
        time_vec (np.array): Vector with time-stamps of observations
        y_obs_vec (np.array): Matrix / vector of observations
        likelihood_noise_type (str): Type likelihood noise computation
        fixed_likelihood_noise_value (float): Prescribed value for the likelihood noise
        output_label (str): Label / name of the output / QoI in the experimental data file
        coord_labels (str): Labels / names of the coordinates in the experimental data file
        settings_probab_mapping (dict): Dictionary with problem setup for the
                                        probabilistic regression model used in
                                        the likelihood formulation
        mf_interface (obj): QUEENS multi-fidelity interface
        bmfia_subiterator (obj): Subiterator to select the training data of the
                                 probabilistic regression model
        noise_upper_bound (float): Upper bound for the likelihood noise
        x_train (np.array): Input of the simulation model used to generate training outputs
                            for probabilistic regression model
        y_hf_train (np.array): High-fidelity training output data for probabilistic
                               regression model
        y_lfs_train (np.array): Low-fidelity training outputs for probabilistic regression
                                models
        z_train (np.array): Combined low-fidelity training vector for probabilistic
                            regression model
        eigenfunc_random_fields (np.array):
        eigenvals (np.array):
        f_mean_train (np.array): Mean values of probabilistic regression model at
                                 training points
        noise_var (float): Noise variance in the multi-fidelity likelihood
        noise_var_lst (lst): List with noise variance per iteration

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
        nugget_noise_var,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        likelihood_noise_type,
        fixed_likelihood_noise_value,
        output_label,
        coord_labels,
        settings_probab_mapping,
        mf_interface,
        bmfia_subiterator,
        noise_upper_bound,
        x_train,
        y_hf_train,
        y_lfs_train,
        z_train,
        eigenfunc_random_fields,
        eigenvals,
        f_mean_train,
        noise_var,
        noise_var_lst,
    ):
        """Instanciate the multi-fidelity likelihood class."""
        super().__init__(
            model_name,
            forward_model,
            coords_mat,
            time_vec,
            y_obs_vec,
            output_label,
            coord_labels,
        )

        self.mf_interface = mf_interface
        self.settings_probab_mapping = settings_probab_mapping
        self.x_train = x_train
        self.y_hf_train = y_hf_train
        self.y_lfs_train = y_lfs_train
        self.z_train = z_train
        self.eigenfunc_random_fields = eigenfunc_random_fields
        self.eigenvals = eigenvals
        self.f_mean_train = f_mean_train
        self.bmfia_subiterator = bmfia_subiterator
        self.noise_var = noise_var
        self.nugget_noise_var = nugget_noise_var
        self.likelihood_noise_type = likelihood_noise_type
        self.fixed_likelihood_noise_value = fixed_likelihood_noise_value
        self.noise_upper_bound = noise_upper_bound
        self.noise_var_lst = noise_var_lst

    @classmethod
    def from_config_create_likelihood(
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

        # TODO the unlabeled treatment of raw data for eigenfunc_random_fields and input vars and
        #  random fields is prone to errors and should be changed! The implementation should
        #  rather use the variable module and reconstruct the eigenfunctions of the random fields
        #  if not provided in the data field

        # get model options
        model_options = config[model_name]

        # get specifics of gaussian likelihood model
        likelihood_noise_type = model_options["likelihood_noise_type"]
        fixed_likelihood_noise_value = model_options.get("fixed_likelihood_noise_value")
        nugget_noise_var = model_options.get("nugget_noise_var", 1e-9)
        noise_upper_bound = model_options.get("noise_upper_bound")

        # ---------- multi-fidelity settings ---------------------------------------------------
        settings_probab_mapping = {"mf_approx_settings": model_options.get("mf_approx_settings")}
        approximation_settings_name = "mf_approx_settings"
        num_processors_multi_processing = settings_probab_mapping['mf_approx_settings'].get(
            "num_processors_multi_processing"
        )
        mf_interface = BmfiaInterface(
            settings_probab_mapping, approximation_settings_name, num_processors_multi_processing
        )

        # ----------------------- create subordinate bmfia iterator ------------------------------
        bmfia_iterator_name = model_options["mf_approx_settings"]["mf_subiterator"]
        bmfia_subiterator = from_config_create_iterator(config, bmfia_iterator_name)

        # ----------------------- create visualization object(s) ---------------------------------
        qvis.from_config_create(config, model_name=model_name)

        # ----------------------  Initialize some attributes -----------------------------------
        x_train = None
        y_hf_train = None
        y_lfs_train = None
        z_train = None
        eigenfunc_random_fields = None  # TODO this should be moved to the variable class!
        eigenvals = None
        f_mean_train = None
        noise_var = None
        noise_var_lst = []

        return cls(
            model_name,
            nugget_noise_var,
            forward_model,
            coords_mat,
            time_vec,
            y_obs,
            likelihood_noise_type,
            fixed_likelihood_noise_value,
            output_label,
            coord_labels,
            settings_probab_mapping,
            mf_interface,
            bmfia_subiterator,
            noise_upper_bound,
            x_train,
            y_hf_train,
            y_lfs_train,
            z_train,
            eigenfunc_random_fields,
            eigenvals,
            f_mean_train,
            noise_var,
            noise_var_lst,
        )

    def evaluate(self, samples, gradient_bool=False):
        """Evaluate multi-fidelity likelihood.

        Evaluation with current set of variables
        which are an attribute of the underlying low-fidelity simulation model.

        Args:
            samples (np.ndarray): Evaluated samples
            gradient_bool (bool, optional): Boolean to determine whether the gradient of the
                                            likelihood should be evaluated (if set to True)

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        if gradient_bool:
            raise NotImplementedError(
                "The gradient response is not implemented, yet for the multi-fidelity likelihood."
            )
        # Initialize underlying models in the first call
        if self.z_train is None:
            self._initialize()

        # reshape the model output according to the number of coordinates
        num_coordinates = self.coords_mat.shape[0]
        num_samples = samples.shape[0]

        # we explicitly cut the array at the variable size as within one batch several chains
        # e.g., in MCMC might be calculated; we only want the last chain here
        Y_LF_mat = self.forward_model.evaluate(samples)['mean'].reshape(-1, num_coordinates)[
            :num_samples, :
        ]
        # evaluate the modified multi-fidelity likelihood expression with LF model response
        mf_log_likelihood = self._evaluate_mf_likelihood(Y_LF_mat, samples)
        return mf_log_likelihood

    def _evaluate_mf_likelihood(self, y_lf_mat, x_batch):
        """Evaluate the Bayesian multi-fidelity likelihood as described in [1].

        Args:
            y_lf_mat (np.array): Response matrix of the low-fidelity model; Row-wise corresponding
                                 to rows in x_batch input batch matrix. Different coordinate
                                 locations along the columns
            x_batch (np.array): Input batch matrix; rows correspond to one input vector;
                                different dimensions along columns

        Returns:
            log_lik_mf_vec (np.array): Column-vector of multi-fidelity log-likelihoods with
                                       one entry per sample

        References:
            [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
                "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
                Propagation in Complex Physical Simulations", arXiv:2001.02892
        """
        diff_mat, var_y_mat = self._calculate_distance_vector_and_var_y(y_lf_mat, x_batch)

        self._calculate_likelihood_noise_var(diff_mat)

        # iterate here over all GPs simultaneously such that the new m is a vector of all e.g. first
        # entries in all GPs
        log_lik_mf = np.empty((0, 1))
        for diff_vec, variance_vec in zip(diff_mat, var_y_mat):
            log_lik_mf_entry = self._log_likelihood_fun(
                np.atleast_2d(variance_vec) + self.noise_var, np.atleast_2d(diff_vec)
            )
            log_lik_mf = np.vstack((log_lik_mf, log_lik_mf_entry))

        return log_lik_mf

    def _calculate_distance_vector_and_var_y(self, y_lf_mat, x_batch):
        """Calculate the distance vectors.

        Distance is calcualted between the observation vector and
        the batch (at x_batch) of simulation vectors. The observation vector is
        a row vector and the simulation vectors are row-wise collected in
        z_mat, respectively m_f_mat and var_y_mat. The resulting difference
        matrix contains the element-wise differences of the observation vector
        with the row-wise simulation vectors.

        Args:
            y_lf_mat (np.array): Matrix containing row-wise vectors of individual low-fidelity
                                 responses.
            x_batch (np.array):  Matrix containing row-wise input vectors for the lf and hf
                                 simulations.

        Returns:
            diff_mat (np.array): Matrix containing row-wise difference vectors between the output
                                 observations and the batch of row-wise simulation outputs.
            var_y_mat (np.array): Matrix containing row-wise variance values for the row-wise
                                  high-fidelity predictions.
        """
        # construct LF feature matrix
        z_mat = self.bmfia_subiterator._set_feature_strategy(
            y_lf_mat, x_batch, self.coords_mat[: y_lf_mat.shape[0]]
        )
        # Get the response matrices of the multi-fidelity mapping
        m_f_mat, var_y_mat = self.mf_interface.evaluate(z_mat)

        assert np.array_equal(
            m_f_mat.shape[1], np.atleast_2d(self.y_obs).shape[1]
        ), "Column dimension of the probab. regression output and y_obs do not agree! Abort..."

        diff_mat = m_f_mat - np.atleast_2d(self.y_obs) * np.ones(m_f_mat.shape)

        return diff_mat, var_y_mat

    def _calculate_likelihood_noise_var(self, diff_mat):
        """Calculate noise variance in likelihood function.

        Based on the chosen likelihood noise type, calculate the current
        noise variance in the Gaussian likelihood model.

        Args:
            diff_mat (np.array): Matrix containing row-wise difference vectors between the output
                                 observations and the batch of row-wise simulation outputs.

        Returns:
            None
        """
        # choose likelihood noise type
        if self.likelihood_noise_type == "fixed":
            self.noise_var = max(self.fixed_likelihood_noise_value, self.nugget_noise_var)
        elif self.likelihood_noise_type == "jeffreys_prior":
            self.noise_var = np.sum(diff_mat**2) / (1 + (diff_mat.shape[0] * diff_mat.shape[1]))
            self.noise_var = max(self.noise_var, self.nugget_noise_var)
            _logger.info(f"Calculated ML-estimate for likelihood noise: {self.noise_var}")
        else:
            raise ValueError(
                f'You provided the likelihood noise type "{self.likelihood_noise_type}",'
                'but the only valid options are "fixed" or "jeffreys_prior". Abort...'
            )

    def _calculate_rkhs_inner_prod(self, diff_vec, inv_k_mf_mat):
        """Calculate the inner product in the reproducing kernel Hilbert space.

        Args:
            diff_vec (np.array): Row difference vector between the observation vector and one
                                 simulation vector. Should be a row vector.
            inv_k_mf_mat (np.array): Inverse covariance matrix (precision matrix) of the
                                     multi-fidelity prediction output vector,
                                     should be scalar or row vector

        Returns:
            inner_prod_rkhs (np.array): Inner product of the difference vector between the data
                                        observation vector and the mean prediction vector
                                        of the multi-fidelity model in the reproducing kernel
                                        Hilbert space with kernel given by the precision matrix
                                        of the multi-fidelity model's output.
        """
        # Check for valid inputs
        assert (diff_vec.ndim == 2) and (
            diff_vec.shape[0] == 1
        ), "Dimension of the difference vector seem off! Abort..."

        assert inv_k_mf_mat.ndim == 2, "Dimension of inv_k_mf_mat is not 2D! Abort..."

        # case of only diagonal covariance represented as a vector
        if inv_k_mf_mat.shape[0] == 1:
            inner_prod_rkhs = np.dot(np.multiply(inv_k_mf_mat, diff_vec), diff_vec.T)

        # case of full covariance matrix
        elif inv_k_mf_mat.shape[0] == inv_k_mf_mat.shape[1]:
            inner_prod_rkhs = np.dot(np.dot(diff_vec, inv_k_mf_mat), diff_vec.T)

        # catch non-valid kernels
        else:
            raise ValueError('The provided kernel is not symmetric nor a vector! Abort...')

        return inner_prod_rkhs

    def _log_likelihood_fun(self, mf_variance_vec, diff_vec):
        """Multi-fidelity log-likelihood function.

        Args:
            mf_variance_vec (np.array): Vector of predicted posterior variance values of
                                        the probabilistic multi-fidelity regression model.
                                        This should be a row vecor.
            diff_vec (np.array): Differences between observation and simulations.
                                 This is a two-dimension

        Retruns:
            log_lik_mf (np.array): Value of log-likelihood function for difference vector.
                                   A two-dimensional vector containing one scalar.
        """
        # Check dimensions of incoming variables
        assert mf_variance_vec.ndim == 2, "Dimension of mf_variance_vec must be two! Abort..."
        assert diff_vec.ndim == 2, "Dimension of diff_vec must be two! Abort..."
        assert (
            diff_vec.size == mf_variance_vec.size
        ), "Size of diff_vec and mf_variance_vec must be the same! Abort..."
        assert mf_variance_vec.shape[0] == 1, (
            "The variable 'mf_variance_vec' must be a row-vector, "
            f"but you provided the shape {mf_variance_vec.shape}! Abort..."
        )
        assert diff_vec.shape[0] == 1, (
            "The variable 'diff_vec' must be a row-vector but you "
            f"provided the shape {diff_vec.shape}. Abort..."
        )

        num_obs = self.y_obs.size  # number of observations
        # Note we assume here only a diagonal covariance matrix, can be generalized in the future
        inv_mf_variance_vec = (
            1 / mf_variance_vec
        )  # note: full covariance matrices not supported, yet

        # calculate the log determinate of the inverse covariance matrix (for now diag vector)
        # note: we use the sum log trick here for better numerical behavior
        log_det_k_mf = np.sum(np.log(mf_variance_vec))
        inner_prod_rkhs = self._calculate_rkhs_inner_prod(diff_vec, inv_mf_variance_vec)

        if self.likelihood_noise_type == "fixed" or self.likelihood_noise_type == "jeffreys_prior":
            log_lik_mf = -1 / 2 * (num_obs * np.log(2 * np.pi) + log_det_k_mf + inner_prod_rkhs)

            # potentially extend likelihood by Jeffreys prior
            if self.likelihood_noise_type == "jeffreys_prior":
                log_lik_mf = log_lik_mf + (0.5 * np.log(2) - 0.5 * np.log(self.noise_var))

        else:
            raise ValueError(
                "Likelihood noise type must be 'fixed' or 'jeffreys_prior', "
                f"but you provided {self.likelihood_noise_type}! Abort ..."
            )

        return np.array(log_lik_mf)

    def _initialize(self):
        """Initialize the multi-fidelity likelihood model.

        Returns:
            None
        """
        _logger.info("---------------------------------------------------------------------")
        _logger.info("Speed-up through Bayesian multi-fidelity inverse analysis (BMFIA)!")
        _logger.info("---------------------------------------------------------------------")
        print_bmfia_acceleration()

        self.bmfia_subiterator.coords_experimental_data = self.coords_mat
        self.bmfia_subiterator.time_vec = self.time_vec
        self.bmfia_subiterator.y_obs = self.y_obs
        self._build_approximation()

    def _build_approximation(self):
        """Construct the probabilistic surrogate / mapping.

        Surrogate is calculated based on the provided training-data
        and optimize the hyper-parameters by maximizing
        the data's evidence or its lower bound (ELBO).

        Returns:
            None
        """
        # Start the bmfia (sub)iterator to create the training data for the probabilistic mapping
        self.z_train, self.y_hf_train = self.bmfia_subiterator.core_run()
        # ----- train regression model on the data ----------------------------------------
        self.mf_interface.build_approximation(self.z_train, self.y_hf_train)
        _logger.info("---------------------------------------------------------------------")
        _logger.info('Probabilistic model was built successfully!')
        _logger.info("---------------------------------------------------------------------")

        # plot the surrogate model
        qvis.bmfia_visualization_instance.plot(
            self.z_train, self.y_hf_train, self.mf_interface.probabilistic_mapping_obj_lst
        )

    # ------- TODO: below not needed atm but something similar might be of interest lateron -----
    def input_dim_red(self):
        """Compression of the input array of the simulation.

        Returns:
            None
        """
        self.get_random_fields()
        # TODO: more to come...

    def get_random_fields_and_truncated_basis(self):
        """Get the random fields and their description from the data files.

        Data is stored in pickle-files and return their truncated basis.
        The truncation is determined based on the explained
        variance threshold (explained_var).

        Args:
            explained_var (float): Threshold for truncation in percent.

        Returns:
            random_fields_trunc_dict (dict): Dictionary containing samples of the random fields
                                             as well as their truncated basis.
            x_uncorr (np.array): Array containing the samples of remaining uncorrelated random
                                 variables
        """
        raise NotImplementedError(
            "Implementation of the method "
            "'get_random_fields_and_truncated_basis' is not finished."
            "The method cannot be used at the moment. Abort..."
        )

    # --------------------------- functions ------------------------------------------------------
    @staticmethod
    def _project_samples_on_truncated_basis(truncated_basis_dict, num_samples):
        """Conduct low-dimensional projection of random fields.

        Project the high-dimensional samples of the random field on the
        truncated bases to yield the projection coefficients of the series
        expansion that serve as a new reduced representation of the random
        fields.

        Args:
            truncated_basis_dict (dic): Dictionary containing random field samples and truncated
                                        bases
            num_samples (int): Number of Monte-Carlo samples

        Returns:
            coefs_mat (np.array): Matrix containing the reduced representation of all random fields
                                stacked together along the columns
        """
        # TODO: not yet used at the moment but will follow soon
        coefs_mat = np.empty((num_samples, 0))

        # iterate over random fields
        for basis in truncated_basis_dict.items():
            coefs_mat = np.hstack(
                (coefs_mat, np.dot(basis[1]["samples"], basis[1]["trunc_basis"].T))
            )

        return coefs_mat
