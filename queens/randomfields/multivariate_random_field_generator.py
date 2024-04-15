"""TODO_doc."""

import sys

import numpy as np
import scipy

from queens.randomfields.univariate_random_field_factory import create_univariate_random_field


class MultiVariateRandomFieldGenerator:
    """Generator of samples of multivariate cross-correlated random fields.

    Class for the generation of samples from Gaussian and
    non-Gaussian multi variate random fields using translation process
    theory. The generation of the cross-correlated random fields is based on
    the approach described in:

    Vorechovsky, Miroslav. "Simulation of simply cross correlated random fields
    by series expansion methods." Structural safety 30.4 (2008): 337-363.

    Attributes:
        var_dim (int):      How many variates do we have?
        my_univ_random_fields (list): Univariate random field generators.
        stoch_dim (int):    Stochastic dimension, i.e. number of random variable
                            needed to generate samples.
        pre_factor (linalg.block_diag):  Factor to multiply independent random
                                         phase angles with (product of eigenvector
                                         and values of cross-correlation matrix).
    """

    def __init__(
        self,
        marginal_distributions=None,
        num_fields=None,
        crosscorr=None,
        spatial_dimension=None,
        corr_struct=None,
        corr_length=None,
        energy_frac=None,
        field_bbox=None,
        num_terms_per_dim=None,
        total_terms=None,
    ):
        """TODO_doc.

        Args:
            marginal_distributions (list): probability distributions of
                                           individual fields
            num_fields (int):              number of fields
            crosscorr (np.array):          cross correlation matrix
            spatial_dimension (int):       spatial dimension of fields
                                           ( the same for all components )
            corr_struct (string):          correlation structure of fields
            corr_length (float):           correlation length of fields
            energy_frac (float):           energy fraction to be retained by
                                           series expansion
            field_bbox (np.array):         bounding box of fields
            num_terms_per_dim (int):       number of expansion terms per spatial
                                           dimension
            total_terms (int):             number of total expansion terms
                                           (per field)
        """
        # do some sanity checks
        if num_fields == len(crosscorr[:, 0]) and num_fields == len(crosscorr[0, :]):
            self.var_dim = num_fields
        else:
            raise RuntimeError('Size of cross-correlation matrix must match number of fields')

        if num_fields != len(marginal_distributions):
            raise RuntimeError(
                'Number of input distributions does not match number of uni-variate fields'
            )

        temp = 0
        self.my_univ_random_fields = []
        for i in range(self.var_dim):
            self.my_univ_random_fields.append(
                create_univariate_random_field(
                    marg_pdf=marginal_distributions[i],
                    spatial_dimension=spatial_dimension,
                    corrstruct=corr_struct,
                    corr_length=corr_length,
                    energy_frac=energy_frac,
                    field_bbox=field_bbox,
                    num_terms_per_dim=num_terms_per_dim,
                    total_terms=total_terms,
                )
            )

            temp = temp + self.my_univ_random_fields[i].stoch_dim

        self.stoch_dim = temp

        # calculate pre_factor_ matrix to obtain block correlated random
        # phase angles/amplitude according to Vorechovsky 2008

        # do this only to check whether cross correlation matrix is
        # semi-positive definite
        np.linalg.cholesky(crosscorr)

        # eigenvalue decomposition of cross-correlation matrix
        lambda_c, phi_c = np.linalg.eig(crosscorr)

        # use dimension of first field for now
        temp = np.diag(np.ones((self.my_univ_random_fields[0].stoch_dim,)))

        phi_d = np.kron(phi_c, temp)
        lambda_c3 = lambda_c.reshape((lambda_c.shape[0], 1, 1)) * temp

        # make sparse at some point
        lambda_d = scipy.linalg.block_diag(*lambda_c3)

        # self.pre_factor=sparse(phi_d*(lambda_d**(1/2)))
        self.pre_factor = np.dot(phi_d, (lambda_d ** (1 / 2)))

    def get_stoch_dim(self):
        """Return stochastic dimension of multi-variate field.

        Returns:
            int: Stochastic dimension
        """
        return self.stoch_dim

    def evaluate_field_at_location(self, x, xi):
        """Generate realization of random field and evaluate it at *x*.

        Args:
            x (np.array):  Locations at which to evaluate random field
            xi (np.array): Random phase angles to compute realization of
                           random field

        Returns:
            np.array: Value of realization of random field at specified
            locations
        """
        new_vals = np.zeros((x.shape[0], self.var_dim))
        helper = np.dot(self.pre_factor, xi)
        np.set_printoptions(threshold=sys.maxsize)

        my_xi = helper.reshape((-1, self.var_dim), order='F')

        for i in range(self.var_dim):
            new_vals[:, i] = self.my_univ_random_fields[i].evaluate_field_at_location(
                x, my_xi[:, i]
            )
        return new_vals
