"""Random fields module."""

import logging

import numpy as np

from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class RandomField:
    """RandomField class.

    Attributes:
            dimension (int): Dimension of the random field (number of coordinates).
            mean_param (float): Parameter for mean function parameterization of random field.
            mean_type (str): Type of mean function of the random field.
            mean (np.ndarray): Discretized mean function.
            coords (np.ndarray): Coordinates at which the random field is evaluated.
            std_hyperparam_rf (float): Hyperparameter for standard-deviation of random field.
            corr_length (float): Hyperparameter for the correlation length.
            nugget_variance_rf (float): Nugget variance for the random field (lower bound for
                                        diagonal values of the covariance matrix).
            explained_variance (float): Explained variance by the eigen decomposition.
            dim_truncated: TODO_doc
            K_mat (np.ndarray): Covariance matrix of the random field.
            cholesky_decomp_covar_mat (np.ndarray): Cholesky decomposition of the covariance matrix.
            eigen_vals_vec (np.ndarray): Eigenvalues of the covariance matrix.
            eigen_vecs_mat (np.ndarray): Eigenvectors of the covariance matrix.
            weighted_eigen_val_mat_truncated (np.ndarray): Truncated representation of the weighted
                                                           eigenvalue matrix.
    """

    @log_init_args(_logger)
    def __init__(
        self,
        coords,
        std_hyperparam_rf,
        corr_length,
        mean_param=0,
        mean_type="constant",
        explained_variance=0.95,
    ):
        """Initialize random field object.

        Args:
            coords (dict): Dictionary with coordinates of discretized random field and the
                           corresponding keys
            std_hyperparam_rf (float): Hyperparameter for standard-deviation of random field
            corr_length (float): Hyperparameter for the correlation length
            mean_param (float): Parameter for mean function parameterization of random field
            mean_type (str): Type of mean function of the random field
            explained_variance (float): Explained variance of by the eigen decomposition
        """
        self.dimension = len(coords['keys'])
        self.mean_param = mean_param
        self.mean_type = mean_type
        self.mean = self.calculate_mean_fun()
        self.coords = coords
        self.std_hyperparam_rf = std_hyperparam_rf
        self.corr_length = corr_length
        self.nugget_variance_rf = 1e-9
        self.explained_variance = explained_variance
        self.weighted_eigen_val_mat_truncated = None
        self.K_mat = self.compute_covariance_matrix_and_cholseky()
        self.cholesky_decomp_covar_mat = np.linalg.cholesky(self.K_mat)
        # compute eigendecomposition
        self.eigen_vals_vec, self.eigen_vecs_mat = self.compute_eigendecomposition()
        # decompose and truncate the random field
        self.weighted_eigen_val_mat_truncated = self._decompose_and_truncate_random_field()
        self.dim_truncated = self.weighted_eigen_val_mat_truncated.shape[1]

    def draw(self, num_samples):
        """Draw samples from the truncated representation of the random field.

        Args:
            num_samples: TODO_doc
        Returns:
            samples (np.ndarray): Drawn samples
        """
        samples = np.random.normal(0, 1, (num_samples, self.dim_truncated))
        return samples

    def expanded_representation(self, sample):
        """Expand truncated representation of sample.

        Args:
            sample (np.ndarray): Truncated representation of sample

        Returns:
            sample_expanded (np.ndarray): Expanded representation of sample
        """
        if self.mean_type == 'inflow_parabola':
            sample = self.mean * (
                1 + self.std_hyperparam_rf * np.dot(self.cholesky_decomp_covar_mat, sample)
            )
            sample[0] = 0  # BCs
            sample[-1] = 0  # BCs

        elif self.mean_type == 'constant':
            sample = self.mean + np.dot(self.weighted_eigen_val_mat_truncated, sample)

        return sample

    def calculate_mean_fun(self):
        """Calculate the discretized mean function."""
        if self.mean_type == 'inflow_parabola':
            fixed_one_dim_coords_vector = np.linspace(0, 1, self.dimension, endpoint=True)
            # Parabola that has its maximum at x = 0
            mean = 4 * self.mean_param * (-((fixed_one_dim_coords_vector - 0.5) ** 2) + 0.25)
        elif self.mean_type == 'constant':
            mean = self.mean_param * np.ones(self.dimension)
        else:
            raise ValueError(
                f"Unknown mean_type. You entered {self.mean_type}, "
                "but we only accept 'inflow_parabola' and 'constant'."
            )
        return mean

    def compute_covariance_matrix_and_cholseky(self):
        """Compute discretized covariance matrix and cholesky decomposition.

        Based on the kernel description of the random field, build its
        covariance matrix using the external geometry and coordinates.
        Afterwards, calculate the Cholesky decomposition.
        """
        K_mat = np.zeros((self.dimension, self.dimension))
        # here we assume a specific kernel, namely a rbf kernel
        for num1, x_one in enumerate(self.coords['coords']):
            for num2, x_two in enumerate(self.coords['coords']):
                K_mat[num1, num2] = self.std_hyperparam_rf**2 * np.exp(
                    -(np.linalg.norm(x_one - x_two) ** 2) / (2 * self.corr_length**2)
                )

        return K_mat + self.nugget_variance_rf * np.eye(self.dimension)

    def compute_eigendecomposition(self):
        """Compute eigenvalues and eigenvectors of covariance matrix."""
        # TODO we should use the information about the Cholesky decomp
        eig_val, eig_vec = np.linalg.eigh(self.K_mat)
        eigen_vals_vec = np.real(eig_val)
        eigen_vecs_mat = np.real(eig_vec)
        return eigen_vals_vec, eigen_vecs_mat

    def _decompose_and_truncate_random_field(self):
        """Decompose and then truncate the random field.

        According to desired variance fraction that should be
        covered/explained by the truncation.
        """
        sum_val = 0
        sum_eigenval = np.sum(self.eigen_vals_vec)
        # calculate m, which is the truncated length and covers 98% of variance
        for num, eigenval in reversed(list(enumerate(self.eigen_vals_vec))):
            sum_val += eigenval
            variance_fraction = sum_val / sum_eigenval
            num_eigen = num
            if variance_fraction > self.explained_variance:
                break
        # truncated eigenfunction base
        eigen_vec_mat_red = self.eigen_vecs_mat[:, num_eigen:]

        # fix orientation of eigenvectors
        for i in range(self.eigen_vals_vec.shape[0] - num_eigen):
            eigen_vec_mat_red[:, i] *= np.sign(eigen_vec_mat_red[0, i])

        # truncated eigenvalues
        eig_val_vec_red = self.eigen_vals_vec[num_eigen:]

        # truncated diagonal eigenvalue matrix
        eigen_val_red_diag_mat = np.diagflat(eig_val_vec_red)

        # weight the eigenbasis with the eigenvalues
        return np.dot(eigen_vec_mat_red, np.sqrt(eigen_val_red_diag_mat))
