from pqueens.randomfields.univariate_random_field_generator import UnivariateRandomFieldSimulator
import numpy as np


class RandomFieldGenFourier(UnivariateRandomFieldSimulator):
    """Fourier series based random field generator

    Random field generator for univariate random fields based on a  Fourier
    series expansion as described in [#f2]_

    .. rubric:: Footnotes
    .. [#f2] Tamellini, L. (2012). Polynomial approximation of PDEs with
        stochastic coefficients.

    Attributes:

        m (int):                number of terms in expansion in each direction
        trunc_thres (int):      truncation threshold for Fourier series
        kb (np.array):          array to store indices for Fourier expansion
        largest_length (float): length of random field (for now equal in all
                                dimensions based on largest dimension of bounding box)
        corr_length  (float):   correlation length of field (so far only isotropic fields)

    """

    def __init__(
        self,
        marginal_distribution,
        corr_length,
        energy_frac,
        field_bbox,
        dimension,
        num_ex_term_per_dim,
        num_terms,
    ):
        """

        Args:
            m (int):                number of terms in expansion in each
                                    direction
            trunc_thres (int):      truncation threshold for Fourier series
            kb (np.array):          array to store indices for Fourier expansion
            largest_length (float): length of random field (for now equal in all
                                    dimensions based on largest dimension of
                                    bounding box)
            corr_length  (float):   correlation length of field
                                    (so far only isotropic fields)

        """
        self.m = None
        self.trunc_thres = None
        self.kb = None
        self.largest_length = None
        self.corr_length = None

        # call superclass  first
        super().__init__(marginal_distribution)

        # sanity checks are done in factory
        self.spatial_dim = dimension

        san_check_bbox = field_bbox.shape
        if san_check_bbox[0] is not self.spatial_dim * 2:
            raise ValueError(
                'field bounding box must be size {} and not {}'.format(
                    self.spatial_dim * 2, san_check_bbox[0]
                )
            )

        self.bounding_box = field_bbox

        # compute largest length and size of random field for now.
        # reshape bounding box so that each dimension is in new row
        bbox = np.reshape(self.bounding_box, (self.spatial_dim, 2))

        # compute the maximum
        self.largest_length = bbox.max(0).max(0)

        if energy_frac < 0 or energy_frac > 1:
            raise ValueError('energy fraction must be between 0 and 1.')

        self.des_energy_frac = energy_frac
        self.m = num_ex_term_per_dim
        self.trunc_thres = num_terms

        if corr_length <= 0:
            raise ValueError('Error: correlation length must be positive')

        if corr_length > 0.35 * self.largest_length:
            raise ValueError(
                'correlation length must smaller than '
                '0.35*largest dimension, please increase size '
                'of bounding box.'
            )

        self.corr_length = corr_length

    def compute_expansion_coefficient(self, k, length_of_field, corr_length):
        """ Compute expansion coeffiecients of Fourier series.

        Args:
            k (int):
            length_of_field (float): periodicity of simulated field
            corr_length (float): correlation length of simulated field

        Returns:
            float: expansion coeffiecient

        """
        if k == 0:
            coeff = corr_length * np.sqrt(np.pi) / (2 * length_of_field)
        else:
            coeff = (
                corr_length
                * np.sqrt(np.pi)
                / (length_of_field)
                * np.exp(-((k * np.pi * corr_length) ** 2 / (4 * length_of_field ** 2)))
            )
        return coeff
