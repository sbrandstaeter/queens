#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""KL Random fields class."""

import logging
from typing import Iterable, List, Optional, Sequence, Union

import gpflow
import numpy as np
import tensorflow as tf
from check_shapes import check_shape as cs
from check_shapes import check_shapes, inherit_check_shapes
from gpflow.base import Parameter, TensorType
from gpflow.config import default_int
from gpflow.functions import Function, MeanFunction
from gpflow.kernels.base import Combination, Kernel
from gpflow.utilities import positive
from scipy.spatial.distance import pdist, squareform

from queens.distributions.mean_field_normal import MeanFieldNormal
from queens.parameters.random_fields._random_field import RandomField

_logger = logging.getLogger(__name__)


class ChangePoint3D(Combination):
    def __init__(
        self,
        kernels: List[Kernel],
        locations: List[float],
        steepness: Union[float, List[float]] = 1.0,
        switch_dim: int = 0,
        name: Optional[str] = None,
    ):
        """:param kernels: list of kernels defining the different regimes
        :param locations: list of change-point locations in the 1d input space
        :param steepness: the steepness parameter(s) of the sigmoids, this can be
            common between them or decoupled
        :param switch_dim: the (one) dimension of the input space along which
            the change-points are defined
        """
        if len(kernels) != len(locations) + 1:
            raise ValueError(
                "Number of kernels ({nk}) must be one more than the number of "
                "changepoint locations ({nl})".format(nk=len(kernels), nl=len(locations))
            )

        if isinstance(steepness, Iterable) and len(steepness) != len(locations):
            raise ValueError(
                "Dimension of steepness ({ns}) does not match number of changepoint "
                "locations ({nl})".format(ns=len(steepness), nl=len(locations))
            )

        super().__init__(kernels, name=name)

        self.switch_dim = switch_dim
        self.locations = Parameter(locations)
        self.steepness = Parameter(steepness, transform=positive())

    def _set_kernels(self, kernels: List[Kernel]):
        # it is not clear how to flatten out nested change-points
        self.kernels = kernels

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        sig_X = self._sigmoids(X)  # N1 x 1 x Ncp
        sig_X2 = self._sigmoids(X2) if X2 is not None else sig_X  # N2 x 1 x Ncp

        # `starters` are the sigmoids going from 0 -> 1, whilst `stoppers` go
        # from 1 -> 0, dimensions are N1 x N2 x Ncp
        starters = sig_X * tf.transpose(sig_X2, perm=(1, 0, 2))
        stoppers = (1 - sig_X) * tf.transpose((1 - sig_X2), perm=(1, 0, 2))

        # prepend `starters` with ones and append ones to `stoppers` since the
        # first kernel has no start and the last kernel has no end
        N1 = tf.shape(X)[0]
        N2 = tf.shape(X2)[0] if X2 is not None else N1
        ones = tf.ones((N1, N2, 1), dtype=X.dtype)
        starters = tf.concat([ones, starters], axis=2)
        stoppers = tf.concat([stoppers, ones], axis=2)

        # now combine with the underlying kernels
        kernel_stack = tf.stack([k(X, X2) for k in self.kernels], axis=2)
        return tf.reduce_sum(kernel_stack * starters * stoppers, axis=2)

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        N = tf.shape(X)[0]
        sig_X = tf.reshape(self._sigmoids(X), (N, -1))  # N x Ncp

        ones = tf.ones((N, 1), dtype=X.dtype)
        starters = tf.concat([ones, sig_X * sig_X], axis=1)  # N x Ncp
        stoppers = tf.concat([(1 - sig_X) * (1 - sig_X), ones], axis=1)

        kernel_stack = tf.stack([k(X, full_cov=False) for k in self.kernels], axis=1)
        return tf.reduce_sum(kernel_stack * starters * stoppers, axis=1)

    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        locations = tf.sort(self.locations)  # ensure locations are ordered
        locations = tf.reshape(locations, (1, 1, -1))
        steepness = tf.reshape(self.steepness, (1, 1, -1))
        Xslice = tf.reshape(X[:, self.switch_dim], (-1, 1, 1))
        return tf.sigmoid(steepness * (Xslice - locations))


class DamagedBeam(MeanFunction, Function):
    def __init__(
        self,
        mu: TensorType = None,
        sigma: TensorType = None,
        relative_peak: TensorType = None,
        offset: TensorType = None,
        jump: TensorType = None,
        scale: TensorType = None,
        width: TensorType = None,
    ) -> None:
        super().__init__()

        mu = np.zeros(1) if mu is None else mu
        sigma = np.ones(1) if sigma is None else sigma
        relative_peak = np.ones(1) if relative_peak is None else relative_peak
        offset = np.zeros(1) if offset is None else offset
        jump = np.zeros(1) if jump is None else jump
        scale = np.ones(1) if scale is None else scale
        width = np.zeros(1) if width is None else width

        self.mu = Parameter(mu)
        self.sigma = Parameter(sigma)
        self.relative_peak = Parameter(relative_peak)
        self.offset = Parameter(offset)
        self.jump = Parameter(jump)
        self.scale = Parameter(scale)
        self.width = Parameter(width)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:

        reshape_shape_X = tf.concat(
            [tf.ones(shape=(tf.rank(X) - 1), dtype=default_int()), [-1]],
            axis=0,
        )

        sigma = tf.reshape(self.sigma, reshape_shape_X)
        relative_peak = tf.reshape(self.relative_peak, reshape_shape_X)
        offset = tf.reshape(self.offset, reshape_shape_X)
        width = tf.reshape(self.width, reshape_shape_X)
        prior = offset
        for i in self.mu:
            mu_i = tf.reshape(i, reshape_shape_X)
            prior = (
                prior
                + (
                    tf.sigmoid(sigma * (X - mu_i + (width / 2)))
                    * tf.sigmoid(-sigma * (X - mu_i - (width / 2)))
                )
                * relative_peak
            )
        return prior


class GPRRandomField1D(RandomField):
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
        std_peak=10,
        corr_length=0.3,
        explained_variance=None,
        latent_dimension=None,
        cut_off=0.0,
        kernel="RBF",
        customkernel=None,
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
        width=0,
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
        self.std_peak = std_peak
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
        if customkernel != None:
            self.kernel = customkernel

        if kernel != "RBF" and kernel != "Matern" and kernel != "SE" and customkernel == None:
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
            width=width,
        )
        self.calculate_covariance_matrix()
        self.eigendecomp_cov_matrix()
        if self.fit == True:
            self.distribution = gpflow.models.GPR(
                (X, y), kernel=self.kernel, mean_function=damaged_beam
            )
        else:
            X = np.zeros((0, 1))
            y = np.zeros((0, 1))
            self.distribution = gpflow.models.GPR(
                (X, y), kernel=self.kernel, mean_function=damaged_beam
            )
        # self.distribution = MeanFieldNormal(
        #     mean=0, variance=1, dimension=self.dimension
        # )

    def draw(self, num_samples):
        """Draw samples from the latent representation of the random field.

        Args:
            num_samples: Number of draws of latent random samples
        Returns:
            samples (np.ndarray): Drawn samples
        """
        mean_distribution = MeanFieldNormal(mean=0, variance=1, dimension=self.dimension)
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
            (self.coords["coords"][:, 0]),
            axis=-1,
        ).reshape(
            -1, 1
        )[:, None]
        samples_expanded = np.array(
            self.distribution.predict_f_samples(sample_coords, num_samples=1)
        ).reshape(1, -1)
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
        distance = squareform(pdist(self.coords["coords"], "sqeuclidean"))
        # covariance = * np.exp(-distance / (2 * self.corr_length**2))
        covariance = np.array(self.kernel(self.coords["coords"]))
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
