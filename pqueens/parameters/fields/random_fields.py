"""Random fields module."""

import numpy as np


class RandomField:
    """RandomField class.

    Attributes:
            dimension (int): Dimension of the random field (number of coordinates)
            mean_param (float): Parameter for mean function parameterization of random field
            mean_type (str): Type of mean function of the random field
            mean (np.ndarray): Discretized mean function
            coords (np.ndarray): Coordinates at which the random field is evaluated
            std_hyperparam_rf (float): Hyperparameter for standard-deviation of random field
            corr_length (float): Hyperparameter for the correlation length
            nugget_variance_rf (float): Nugget variance for the random field (lower bound for
                                        diagonal values of the covariance matrix)
            explained_variance (float): Explained variance of by the eigen decomposition
    """

    def __init__(
        self,
        dimension,
        coords,
        mean_param,
        mean_type,
        std_hyperparam_rf,
        corr_length,
        explained_variance,
    ):
        """Initialize random field object.

        Args:
            dimension (int): Dimension of the random field (number of coordinates)
            coords (dict): Dictionary with coordinates of discretized random field and the
                           corresponding keys
            mean_param (float): Parameter for mean function parameterization of random field
            mean_type (str): Type of mean function of the random field
            std_hyperparam_rf (float): Hyperparameter for standard-deviation of random field
            corr_length (float): Hyperparameter for the correlation length
            explained_variance (float): Explained variance of by the eigen decomposition
        """
        self.dimension = dimension
        self.mean_param = mean_param
        self.mean_type = mean_type
        self.mean = None
        self.coords = coords
        self.std_hyperparam_rf = std_hyperparam_rf
        self.corr_length = corr_length
        self.nugget_variance_rf = 1e-9
        self.explained_variance = explained_variance
        self.initialize()
        self.dim_truncated = self.weighted_eigen_val_mat_truncated.shape[1]

    def draw(self, num_samples):
        """Draw samples from the truncated representation of the random field.

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
            sample_expanded (np.ndarray) Expanded representation of sample
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

    def initialize(self):
        """Calculate discretized mean and covariance of random field."""
        self.calculate_mean_fun()
        self.calculate_covariance_matrix_and_cholseky()

    def calculate_mean_fun(self):
        """Calculate the discretized mean function."""
        if self.mean_type == 'inflow_parabola':
            fixed_one_dim_coords_vector = np.linspace(0, 1, self.dimension, endpoint=True)
            # Parabola that has its maximum at x = 0
            self.mean = 4 * self.mean_param * (-((fixed_one_dim_coords_vector - 0.5) ** 2) + 0.25)
        elif self.mean_type == 'constant':
            self.mean = self.mean_param * np.ones(self.dimension)

    def calculate_covariance_matrix_and_cholseky(self):
        """Calculate discretized covariance matrix and cholesky decomposition.

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

        self.K_mat = K_mat + self.nugget_variance_rf * np.eye(self.dimension)
        self.cholesky_decomp_covar_mat = np.linalg.cholesky(self.K_mat)

        # decompose and truncate the random field
        self._decompose_and_truncate_random_field()

    def _decompose_and_truncate_random_field(self):
        """Decompose and then truncate the random field.

        According to desired variance fraction that should be
        covered/explained by the truncation.
        """
        # compute eigendecomposition
        # TODO we should use the information about the Cholesky decomp
        eig_val, eig_vec = np.linalg.eigh(self.K_mat)
        self.eigen_vals_vec = np.real(eig_val)
        self.eigen_vecs_mat = np.real(eig_vec)

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
        # will be written to the db externally
        self.weighted_eigen_val_mat_truncated = np.dot(
            eigen_vec_mat_red, np.sqrt(eigen_val_red_diag_mat)
        )
