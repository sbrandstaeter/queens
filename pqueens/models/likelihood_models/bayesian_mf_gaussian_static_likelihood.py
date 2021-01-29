import numpy as np
from scipy.optimize import minimize
from pqueens.iterators.iterator import Iterator
from .likelihood_model import LikelihoodModel
from pqueens.interfaces.bmfmc_interface import BmfmcInterface


class BMFGaussianStaticModel(LikelihoodModel):
    """
    Returns:
        Instance of BMFGaussianStaticModel. This is a multi-fidelity version of the
        Gaussian static noise likelihood model.

    References:
        [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
            "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
            Propagation in Complex Physical Simulations", arXiv:2001.02892
    """

    def __init__(
        self,
        model_name,
        model_parameters,
        nugget_noise_var,
        forward_model,
        coords_mat,
        y_obs_vec,
        likelihood_noise_type,
        fixed_likelihood_noise_value,
        output_label,
        coord_labels,
        settings_probab_mapping,
        mf_interface,  # TODO maybe this can be deleted
        bmfia_subiterator,
        noise_upper_bound,
    ):
        super(BMFGaussianStaticModel, self).__init__(
            model_name,
            model_parameters,
            forward_model,
            coords_mat,
            y_obs_vec,
            output_label,
            coord_labels,
        )

        self.mf_interface = mf_interface  # TODO check if this can be deleted
        self.settings_probab_mapping = settings_probab_mapping
        self.x_train = None
        self.y_hf_train = None
        self.y_lfs_train = None
        self.gammas_ext_train = None
        self.z_train = None
        self.p_yhf_mean = None
        self.p_yhf_var = None
        self.eigenfunc_random_fields = None  # TODO this should be moved to the variable class!
        self.eigenvals = None
        self.f_mean_train = None
        self.training_indices = None
        self.bmfia_subiterator = bmfia_subiterator
        self.uncertain_parameters = model_parameters
        self.k_mf = None
        self.det_k_mf = None
        self.k_mf_inv = None

        self.noise_var = 1e-4
        self.nugget_noise_var = nugget_noise_var
        self.likelihood_noise_type = likelihood_noise_type
        self.fixed_likelihood_noise_value = fixed_likelihood_noise_value
        self.noise_upper_bound = noise_upper_bound
        self.noise_var_lst = []

    @classmethod
    def from_config_create_likelihood(
        cls,
        model_name,
        config,
        model_parameters,
        forward_model,
        coords_mat,
        y_obs_vec,
        output_label,
        coord_labels,
    ):
        """

        Returns:
            BMFGaussianStaticModel (obj): A BMFGaussianStaticModel object
        """

        # TODO the unlabeled treatment of raw data for eigenfunc_random_fields and input vars and
        #  random fields is prone to errors and should be changed! The implementation should
        #  rather use the variable module and reconstruct the eigenfunctions of the random fields
        #  if not provided in the data field

        # get model options
        model_options = config[model_name]

        # get specifics of gaussian static likelihood model
        likelihood_noise_type = model_options["likelihood_noise_type"]
        fixed_likelihood_noise_value = model_options.get("fixed_likelihood_noise_value")
        nugget_noise_var = model_options.get("nugget_noise_var")
        noise_upper_bound = model_options.get("noise_upper_bound")

        # ---------- multi-fidelity settings ---------------------------------------------------
        settings_probab_mapping = {"mf_approx_settings": model_options.get("mf_approx_settings")}
        approximation_settings_name = "mf_approx_settings"
        mf_interface = BmfmcInterface(settings_probab_mapping, approximation_settings_name)

        # ----------------------- create subordinate bmfia iterator ------------------------------
        bmfia_iterator_name = model_options["mf_approx_settings"]["mf_subiterator"]
        bmfia_subiterator = Iterator.from_config_create_iterator(config, bmfia_iterator_name)

        return cls(
            model_name,
            model_parameters,
            nugget_noise_var,
            forward_model,
            coords_mat,
            y_obs_vec,
            likelihood_noise_type,
            fixed_likelihood_noise_value,
            output_label,
            coord_labels,
            settings_probab_mapping,
            mf_interface,  # TODO check if this is really needed
            bmfia_subiterator,
            noise_upper_bound,
        )

    def evaluate(self):
        """
        Evaluate multi-fidelity likelihood with current set of variables which are an attribute
        of the underlying low-fidelity simulation model

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.

        """
        # Initialize underlying models in the first call
        if self.z_train is None:
            self._initialize()
        # catch one dimensional vectors
        Y_LF_mat = self._update_and_evaluate_forward_model()

        # get x_batch form variable object
        x_batch = []
        for j, _ in enumerate(self.variables):
            x_batch.append(np.array([i[1]['value'] for i in self.variables[j].variables.items()]))

        x_batch = np.atleast_2d(x_batch)

        # evaluate the modified multi-fidelity likelihood expression with LF model response
        mf_log_likelihood = self._evaluate_mf_likelihood(Y_LF_mat, x_batch)

        return mf_log_likelihood

    def _evaluate_mf_likelihood(self, y_lf_mat, x_batch):
        """
        Bayesian multi-fidelity likelihood as described in [1]
        Args:
            y_lf_mat (np.array): Response matrix of the low-fidelity model; Row-wise corresponding
                                 to rows in x_batch input batch matrix. Different coordinate
                                 locations along the columns
            x_batch (np.array): Input batch matrix; rows correspond to one input vector;
                                different dimensions along columns

        Returns:
            log_lik_mf_vec (np.array): Array of multi-fidelity log-likelihoods with one entry per
                                       sample

        References:
            [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
                "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
                Propagation in Complex Physical Simulations", arXiv:2001.02892

        """
        n = self.y_obs_vec.size

        # loop over all simulation runs, respectively x_vecs and y_vecs
        log_lik_mf_lst = []
        for x_vec, y_lf_vec in zip(x_batch, y_lf_mat):
            # construct LF feature matrix
            z_mat = np.atleast_2d(
                self._get_feature_mat(y_lf_vec, x_vec, self.coords_mat[: y_lf_vec.shape[0]])
            ).T

            # output is vector-valued to to z_mat
            m_f_vec, k_y_mat = self.mf_interface.map(z_mat, full_cov=True)
            diff_vec = np.atleast_2d(self.y_obs_vec).T - m_f_vec

            if self.likelihood_noise_type == "fixed":
                self.noise_var = self.fixed_likelihood_noise_value
            elif self.likelihood_noise_type == "jeffreys_prior":
                # optimize log likelihood here
                res = minimize(
                    lambda noise: self._neg_log_likelihood_fun(noise, k_y_mat, n, diff_vec),
                    self.noise_var,
                    method='L-BFGS-B',
                    bounds=((self.nugget_noise_var, self.noise_upper_bound),),  # TODO pull bounds
                    # out
                    jac=lambda noise: self._grad_neg_log_likelihood_fun_noise(
                        noise, k_y_mat, n, diff_vec
                    ),
                    options={'disp': False},
                )
                self.noise_var = res.x
                self.noise_var_lst.append(res.x)
                print(f"Optimized noise: {res.x}")

            log_lik_mf = -self._neg_log_likelihood_fun(self.noise_var, k_y_mat, n, diff_vec)

            log_lik_mf_lst.append(log_lik_mf)

        log_lik_mf_vec = np.array(log_lik_mf_lst).reshape(-1, 1)

        return log_lik_mf_vec

    def _neg_log_likelihood_fun(self, noise_var, k_y_mat, n, diff_vec):

        k_mf = k_y_mat + noise_var * np.eye(k_y_mat.shape[0])

        if not np.array_equal(k_mf, self.k_mf):
            self.k_mf_inv = np.linalg.inv(k_mf)
            self.det_k_mf = np.linalg.det(k_mf)  # TODO get determinante from gauss-seidel
            self.k_mf = k_mf

        log_lik_mf = -(
            -n / 2 * np.log(2 * np.pi)
            - 1 / 2 * np.log(self.det_k_mf)
            - 0.5 * np.dot(np.dot(diff_vec.T, self.k_mf_inv), diff_vec)
        )

        # potentially extent likelihood by Jeffreys prior
        if self.likelihood_noise_type == "jeffreys_prior":
            # note that K is multivariate and dense here such that the Jeffreys prior yields:
            log_lik_mf = log_lik_mf + (n + 2) / 2 * np.log(self.det_k_mf)

        log_lik_mf = log_lik_mf.squeeze()

        return log_lik_mf

    def _grad_neg_log_likelihood_fun_noise(self, noise_var, K_y_mat, n, diff_vec):

        K_mf = K_y_mat + noise_var * np.eye(K_y_mat.shape[0])

        if not np.array_equal(K_mf, self.k_mf):
            self.k_mf_inv = np.linalg.inv(K_mf)
            self.det_k_mf = np.linalg.det(K_mf)  # TODO get determinante from gauss-seidel
            self.k_mf = K_mf

        alpha = np.dot(self.k_mf_inv, diff_vec)
        grad_log = 0.5 * np.trace(np.outer(alpha, alpha.T) - self.k_mf_inv) - (
            n + 2
        ) / 2 * np.trace(self.k_mf_inv)

        return -grad_log

    def _get_feature_mat(
        self, y_LF_vec, x_vec, coordinates
    ):  # TODO make this a compatible to vector coords
        y_LF_vec = np.atleast_2d(y_LF_vec).T
        x_vec = np.atleast_2d(x_vec)
        if self.bmfia_subiterator.settings_probab_mapping['features_config'] == "man_features":
            output_size = int(y_LF_vec.shape[0] / x_vec.shape[0])
            idx_vec = self.bmfia_subiterator.settings_probab_mapping['X_cols']
            if len(idx_vec) < 2:
                gamma_vec = np.atleast_2d(x_vec[:, idx_vec]).T
            else:
                gamma_vec = np.atleast_2d(x_vec[:, idx_vec])

            gamma_vec_rep = np.matlib.repmat(gamma_vec, 1, output_size).reshape(
                -1, gamma_vec.shape[1]
            )
            z_mat = np.hstack([y_LF_vec, gamma_vec_rep, coordinates])
        elif self.bmfia_subiterator.settings_probab_mapping['features_config'] == "opt_features":
            if self.bmfia_subiterator.settings_probab_mapping['num_features'] < 1:
                raise ValueError()
            self.bmfia_subiterator._update_probabilistic_mapping_with_features()
        elif self.bmfia_subiterator.settings_probab_mapping['features_config'] == "coord_features":
            z_mat = np.hstack([y_LF_vec, coordinates])
        elif self.bmfia_subiterator.settings_probab_mapping['features_config'] == "no_features":
            z_mat = y_LF_vec
        else:
            raise IOError("Feature space method specified in input file is unknown!")

        return z_mat

    def _initialize(self):
        print("---------------------------------------------------------------------")
        print("Speed-up through Bayesian multi-fidelity inverse analysis (BMFIA)!")
        print("---------------------------------------------------------------------")
        self.bmfia_subiterator.coords = self.coords_mat
        self.bmfia_subiterator.y_obs_vec = self.y_obs_vec
        self._build_approximation()

    def _build_approximation(self):
        """
        Construct the probabilistic surrogate / mapping based on the provided training-data and
        optimize the hyper-parameters by maximizing the data's evidence or its lower bound (ELBO).

        Returns:
            None

        """
        # Start the bmfia (sub)iterator to create the training data for the probabilistic mapping
        self.z_train, self.y_hf_train = self.bmfia_subiterator.core_run()
        # ----- train regression model on the data ----------------------------------------
        self.mf_interface.build_approximation(self.z_train, self.y_hf_train)
        print('Probabilistic model was built successfully!')

    # ------- TODO below not needed atm but something similar might be of interest lateron -----
    def input_dim_red(self):
        self.get_random_fields()
        # TODO more to come...

    def get_random_fields_and_truncated_basis(self, explained_var=95.0):
        """
        Get the random fields and their description from the data files (pickle-files) and
        return their truncated basis. The truncation is determined based on the explained
        variance threshold (explained_var).

        Args:
            explained_var (float): Threshold for truncation in percent.

        Returns:
            random_fields_trunc_dict (dict): Dictionary containing samples of the random fields
                                             as well as their truncated basis.
            x_uncorr (np.array): Array containing the samples of remaining uncorrelated random
                                 variables

        """
        # determine uncorrelated random variables
        num_random_var = len(self.uncertain_parameters.get("random_variables"))
        x_uncorr = self.X_mc[:, 0:num_random_var]

        # iterate over all random fields
        dim_random_fields = 0

        if self.uncertain_parameters.get("random_fields") is not None:
            # TODO get information of random fields and conduct dim reduction
            pass

    def _update_and_evaluate_forward_model(self):
        """
        Pass the variables update to subordinate simulation model and then evaluate the
        simulation model.

        Returns:
           Y_mat (np.array): Simulation output (row-wise) that corresponds to input batch X_batch

        """
        # Note that the wrapper of the model update needs to called externally such that
        # self.variables is updated
        self.forward_model.variables = self.variables
        Y_mat = self.forward_model.evaluate()['mean']

        return Y_mat


# --------------------------- functions ------------------------------------------------------
def _project_samples_on_truncated_basis(truncated_basis_dict, num_samples):
    """
    Project the high-dimensional samples of the random field on the truncated bases to yield the
    projection coefficients of the series expansion that serve as a new reduced representation of
    the random fields

    Args:
        truncated_basis_dict (dic): Dictionary containing random field samples and truncated bases
        num_samples (int): Number of Monte-Carlo samples

    Returns:
        coefs_mat (np.array): Matrix containing the reduced representation of all random fields
                              stacked together along the columns
    """
    coefs_mat = np.empty((num_samples, 0))

    # iterate over random fields
    for basis in truncated_basis_dict.items():
        coefs_mat = np.hstack((coefs_mat, np.dot(basis[1]["samples"], basis[1]["trunc_basis"].T)))

    return coefs_mat
