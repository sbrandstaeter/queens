"""KL Random fields class."""

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
import gpflow
from check_shapes import inherit_check_shapes
from gpflow.base import Parameter, TensorType
from gpflow.config import default_int
from gpflow.functions import Function, MeanFunction

from queens.distributions.mean_field_normal import MeanFieldNormalDistribution
from queens.parameters.fields.random_fields import RandomField

_logger = logging.getLogger(__name__)


class DamagedBeam(MeanFunction, Function):
    def __init__(
        self,
        mu: TensorType = None,
        sigma: TensorType = None,
        relative_peak: TensorType = None,
        offset: TensorType = None,
        jump: TensorType = None,
        scale: TensorType = None,
    ) -> None:
        super().__init__()

        mu = np.zeros(1) if mu is None else mu
        sigma = np.ones(1) if sigma is None else sigma
        relative_peak = np.ones(1) if relative_peak is None else relative_peak
        offset = np.zeros(1) if offset is None else offset
        jump = np.zeros(1) if jump is None else jump
        scale = np.ones(1) if scale is None else scale

        self.mu = Parameter(mu)
        self.sigma = Parameter(sigma)
        self.relative_peak = Parameter(relative_peak)
        self.offset = Parameter(offset)
        self.jump = Parameter(jump)
        self.scale = Parameter(scale)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        x, Y, Z = np.split(X, 3, axis=1)

        reshape_shape_X = tf.concat(
            [tf.ones(shape=(tf.rank(x) - 1), dtype=default_int()), [-1]],
            axis=0,
        )
        reshape_shape_Y = tf.concat(
            [tf.ones(shape=(tf.rank(Y) - 1), dtype=default_int()), [-1]],
            axis=0,
        )
        reshape_shape_Z = tf.concat(
            [tf.ones(shape=(tf.rank(Z) - 1), dtype=default_int()), [-1]],
            axis=0,
        )
        mu = tf.reshape(self.mu, reshape_shape_X)
        sigma = tf.reshape(self.sigma, reshape_shape_X)
        relative_peak = tf.reshape(self.relative_peak, reshape_shape_X)
        offset = tf.reshape(self.offset, reshape_shape_X)

        jump = tf.reshape(self.jump, reshape_shape_Y)
        scale = tf.reshape(self.scale, reshape_shape_Y)

        return (
            tf.exp(-((x - mu) ** 2) / (2 * sigma**2))
            * relative_peak
            * tf.sigmoid(scale * (Y - jump))
            + offset
        )


class GPRRandomField(RandomField):
    """Karhunen Loeve RandomField class.

    Attributes:
            nugget_variance (float): Nugget variance for the random field (lower bound for
                                        diagonal values of the covariance matrix).
            explained_variance (float): Explained variance by the eigen decomposition.
            std (float): Hyperparameter for standard-deviation of random field
            corr_length (float): Hyperparameter for the correlation length
            cut_off (float): Lower value limit of covariance matrix entries
            mean (np.array): Mean at coordinates of random field, can be a single constant
            cov_matrix (np.array): Covariance matrix to compute eigendecomposition on
            eigenbasis (np.array): Eigenvectors of covariance matrix, weighted by the eigenvalues
            eigenvalues (np.array): Eigenvalues of covariance matrix
            eigenvectors (np.array): Eigenvectors of covariance matrix
            dimension (int): Dimension of the latent space
    """

    def __init__(
        self,
        coords,
        mean=0.0,
        std=1.0,
        corr_length=0.3,
        explained_variance=None,
        latent_dimension=None,
        cut_off=0.0,
        kernel="RBF",
        nu=1.5,
        alpha=1.0,
        X=[],
        y=[],
        noise_std=1e-5,
        fit=False,
        mu=0,
        sigma=0,
        relative_peak=0,
        offset=0,
        scale=0,
        jump=0,
    ):
        """Initialize GPR object.

        Args:
            coords (dict): Dictionary with coordinates of discretized random field and the
            mean (np.array): Mean at coordinates of random field, can be a single constant
            std (float): Hyperparameter for standard-deviation of random field
            corr_length (float): Hyperparameter for the correlation length
            explained_variance (float): Explained variance of by the eigen decomposition,
                                        mutually exclusive argument with latent_dimension
            latent_dimension (int): Dimension of the latent space,
                                    mutually exclusive argument with explained_variance
            cut_off (float): Lower value limit of covariance matrix entries
        """
        super().__init__(coords)
        self.nugget_variance = 1e-9
        self.explained_variance = explained_variance
        self.std = std
        self.corr_length = corr_length
        self.cut_off = cut_off
        self.mean = mean
        self.cov_matrix = None
        self.eigenbasis = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.kernel = kernel
        self.nu = nu
        self.alpha = alpha
        self.X = X
        self.y = y
        self.noise_std = noise_std
        self.fit = fit

        if (latent_dimension is None and explained_variance is None) or (
            latent_dimension is not None and explained_variance is not None
        ):
            raise KeyError("Specify either dimension or explained variance")
        if kernel == "RBF":
            self.kernel = gpflow.kernels.RBF(variance=self.std**2, lengthscales=self.corr_length)
        if kernel == "Matern":
            self.kernel = gpflow.kernels.Matern52(
                variance=self.std**2, lengthscales=self.corr_length
            )
        if kernel == "SE":
            self.kernel = gpflow.kernels.SquaredExponential(
                variance=self.std**2, lengthscales=self.corr_length
            )
        if kernel != "RBF" and kernel != "Matern" and kernel != "SE":
            raise KeyError("Kernel must be RBF, Matern, or SE (Squared Exponential)")
        if latent_dimension is not None:
            self.dimension = latent_dimension
        else:
            self.dimension = None

        damaged_beam = DamagedBeam(
            mu=mu,
            sigma=sigma,
            relative_peak=relative_peak,
            offset=offset,
            scale=scale,
            jump=jump,
        )
        _logger.info(gpflow.__version__)
        self.calculate_covariance_matrix()
        self.eigendecomp_cov_matrix()
        if self.fit == True:
            self.distribution = gpflow.models.GPR(
                (X, y), kernel=self.kernel, mean_function=damaged_beam
            )
        else:
            X = np.zeros((0, 3))
            y = np.zeros((0, 1))
            self.distribution = gpflow.models.GPR(
                (X, y), kernel=self.kernel, mean_function=damaged_beam
            )
        # self.distribution = MeanFieldNormalDistribution(
        #     mean=0, variance=1, dimension=self.dimension
        # )

    def draw(self, num_samples):
        """Draw samples from the latent representation of the random field.

        Args:
            num_samples: Number of draws of latent random samples
        Returns:
            samples (np.ndarray): Drawn samples
        """

        mean_distribution = MeanFieldNormalDistribution(
            mean=0, variance=1, dimension=self.dimension
        )
        return mean_distribution.draw(num_samples)
        # return np.zeros(num_samples, self.dimension)

    def logpdf(self, samples):
        """Get joint logpdf of latent space.

        Args:
            samples (np.array): Samples for evaluating the logpdf

        Returns:
            logpdf (np.array): Logpdf of the samples
        """
        logpdf, grad_logpdf = self.distribution.log_marginal_likelihood()
        return logpdf

    def grad_logpdf(self, samples):
        """Get gradient of joint logpdf of latent space.

        Args:
            samples (np.array): Samples for evaluating the gradient of the logpdf

        Returns:
            gradient (np.array): Gradient of the logpdf
        """
        logpdf, grad_logpdf = self.distribution.log_marginal_likelihood()
        return grad_logpdf

    def expanded_representation(self, samples):
        """Expand latent representation of sample.

        Args:
            samples (np.ndarray): Latent representation of sample

        Returns:
            samples_expanded (np.ndarray): Expanded representation of sample
        """
        sample_coords = np.stack(
            (self.coords['coords'][:, 0], self.coords['coords'][:, 1], self.coords['coords'][:, 2]),
            axis=-1,
        ).reshape(-1, 3)
        samples_expanded = np.array(
            self.distribution.predict_f_samples(sample_coords, num_samples=1)
        ).reshape(1, -1)
        _logger.info("Samples:")
        _logger.info(np.size(samples, 0))
        # _logger.info(np.size(samples, 1))
        _logger.info("Samples Expanded")
        _logger.info(np.size(samples_expanded, 0))
        _logger.info(np.size(samples_expanded, 1))
        _logger.info("Eigenbasis")
        _logger.info(np.size(self.eigenbasis, 0))
        _logger.info(np.size(self.eigenbasis, 1))
        return samples_expanded

    def latent_gradient(self, upstream_gradient):
        """Gradient of the field with respect to the latent parameters.

        Args:
            upstream_gradient (np.ndarray): Gradient with respect to all coords of the field

        Returns:
            latent_grad (np.ndarray): Graident of the field with respect to the latent
            parameters
        """
        latent_grad = np.matmul(upstream_gradient, self.eigenbasis)
        return latent_grad

    def calculate_covariance_matrix(self):
        """Calculate discretized covariance matrix.

        Based on the kernel description of the random field, build its
        covariance matrix using the external geometry and coordinates.
        """
        # assume squared exponential kernel
        distance = squareform(pdist(self.coords['coords'], 'sqeuclidean'))
        # covariance = * np.exp(-distance / (2 * self.corr_length**2))
        if self.kernel == "RBF":
            self.kernel = gpflow.kernels.RBF(variance=self.std**2, lengthscales=self.corr_length)
        if self.kernel == "Matern":
            self.kernel = gpflow.kernels.Matern52(
                variance=self.std**2, lengthscales=self.corr_length
            )
        if self.kernel == "SE":
            self.kernel = gpflow.kernels.SquaredExponential(
                variance=self.std**2, lengthscales=self.corr_length
            )
        covariance = np.array(self.kernel(self.coords['coords']))
        covariance[covariance < self.cut_off] = 0
        self.cov_matrix = covariance + self.nugget_variance * np.eye(self.dim_coords)

    def eigendecomp_cov_matrix(self):
        """Decompose and then truncate the random field.

        According to desired variance fraction that should be
        covered/explained by the truncation.
        """
        # compute eigendecomposition
        eig_val, eig_vec = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.flip(eig_val)
        eigenvectors = np.flip(eig_vec, axis=1)

        if self.dimension is None:
            eigenvalues_normed = eigenvalues / np.sum(eigenvalues)
            dimension = (np.cumsum(eigenvalues_normed) < self.explained_variance).argmin() + 1
            if dimension == 1 and eigenvalues_normed[0] <= self.explained_variance:
                raise ValueError("Expansion failed.")

            self.dimension = dimension

        # truncated eigenfunction base
        self.eigenvalues = eigenvalues[: self.dimension]
        self.eigenvectors = eigenvectors[:, : self.dimension]

        if self.explained_variance is None:
            self.explained_variance = np.sum(self.eigenvalues) / np.sum(eigenvalues)
            _logger.info("Explained variance is %f", self.explained_variance)

        # weight the eigenbasis with the eigenvalues
        self.eigenbasis = self.eigenvectors * np.sqrt(self.eigenvalues)
