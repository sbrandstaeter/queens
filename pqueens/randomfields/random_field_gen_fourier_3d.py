"""TODO_doc."""

import numpy as np

from pqueens.randomfields.random_field_gen_fourier import RandomFieldGenFourier


class RandomFieldGenFourier3D(RandomFieldGenFourier):
    """Fourier series based 3D random field generator.

    Generator for 3-dimensional random fields based on a Fourier series
    expansion.


    Attributes:
        kb: TODO_doc
        act_energy_frac: TODO_doc
        stoch_dim: TODO_doc
    """

    def __init__(
        self,
        marginal_distribution,
        corr_length,
        energy_frac,
        field_bbox,
        num_ex_term_per_dim,
        num_terms,
    ):
        """TODO_doc.

        Args:
            marginal_distribution (scipy.stats.rv_continuous): marginal
                distribution of random field
            corr_length (float): correlation length of random field
            energy_frac (float): energy fraction retianed by Fourier series
                expansion
            field_bbox (np.array): bouding box for field
            num_ex_term_per_dim (int): number of expansion term per dimension
            num_terms (int): number of terms in expansion
        """
        # call superclass constructor first
        super().__init__(
            marginal_distribution,
            corr_length,
            energy_frac,
            field_bbox,
            3,
            num_ex_term_per_dim,
            num_terms,
        )

        # setup truncation of fourier expansion
        num_ck = 0
        sum_ck = 0

        temp = []
        for k1 in range(0, self.m):
            for k2 in range(0, self.m):
                for k3 in range(0, self.m):
                    if (k1**2 + k2**2 + k3**2) <= self.trunc_thres:
                        sum_ck = sum_ck + (
                            self.compute_expansion_coefficient(
                                k1, self.largest_length, self.corr_length
                            )
                            * self.compute_expansion_coefficient(
                                k2, self.largest_length, self.corr_length
                            )
                            * self.compute_expansion_coefficient(
                                k3, self.largest_length, self.corr_length
                            )
                        )
                        num_ck = num_ck + 1
                        temp.append(np.array((k1, k2, k3)))

        self.kb = np.array(temp)

        if sum_ck < energy_frac:
            raise RuntimeError('Error: not converged try again')

        self.act_energy_frac = sum_ck

        # commpute stochastic dimension based on kb
        self.stoch_dim = self.kb.shape[0] * 8

    def gen_sample_gauss_field(self, loc, phase_angles):
        """Generate sample of Gaussian field.

        Compute realization of standard Gaussian field based on passed phase
        angles *phase_angles* and return values of the realization at *loc*.

        Args:
            loc (np.array): Location at which field is evaluated
            phase_angles (np.array): Pseudo random phase angles for field
                generation
        Returns:
            np.array: Vector containing values of realization at specified
            locations
        """
        if len(phase_angles) != self.stoch_dim:
            raise RuntimeError(
                f'Number of random phase angles {len(phase_angles)} does not match '
                f'stochastic dimension {self.stoch_dim} of the field!'
            )

        if len(loc[0, :]) != 3:
            raise RuntimeError(
                f'Dimension of location vector must be two, not {len(loc[0, :])}'
            )

        # reorder phase angles in matrix
        xi = np.reshape(phase_angles, (-1, 8))
        tempgp = 0
        for i in range(self.kb.shape[0]):
            wk1 = self.kb[i, 0] * np.pi / self.largest_length
            wk2 = self.kb[i, 1] * np.pi / self.largest_length
            wk3 = self.kb[i, 2] * np.pi / self.largest_length
            tempgp = tempgp + np.sqrt(
                (
                    super().compute_expansion_coefficient(
                        self.kb[i, 0], self.largest_length, self.corr_length
                    )
                    * super().compute_expansion_coefficient(
                        self.kb[i, 1], self.largest_length, self.corr_length
                    )
                    * super().compute_expansion_coefficient(
                        self.kb[i, 2], self.largest_length, self.corr_length
                    )
                )
            ) * (
                xi[i, 0]
                * np.cos(wk1 * loc[:, 0])
                * np.cos(wk2 * loc[:, 1])
                * np.cos(wk3 * loc[:, 2])
                + xi[i, 1]
                * np.sin(wk1 * loc[:, 0])
                * np.sin(wk2 * loc[:, 1])
                * np.cos(wk3 * loc[:, 2])
                + xi[i, 2]
                * np.cos(wk1 * loc[:, 0])
                * np.sin(wk2 * loc[:, 1])
                * np.cos(wk3 * loc[:, 2])
                + xi[i, 3]
                * np.sin(wk1 * loc[:, 0])
                * np.cos(wk2 * loc[:, 1])
                * np.cos(wk3 * loc[:, 2])
                + xi[i, 4]
                * np.cos(wk1 * loc[:, 0])
                * np.cos(wk2 * loc[:, 1])
                * np.sin(wk3 * loc[:, 2])
                + xi[i, 5]
                * np.sin(wk1 * loc[:, 0])
                * np.sin(wk2 * loc[:, 1])
                * np.sin(wk3 * loc[:, 2])
                + xi[i, 6]
                * np.cos(wk1 * loc[:, 0])
                * np.sin(wk2 * loc[:, 1])
                * np.sin(wk3 * loc[:, 2])
                + xi[i, 7]
                * np.sin(wk1 * loc[:, 0])
                * np.cos(wk2 * loc[:, 1])
                * np.sin(wk3 * loc[:, 2])
            )

        return tempgp
