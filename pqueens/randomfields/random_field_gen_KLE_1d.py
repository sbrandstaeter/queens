import numpy as np
from pqueens.randomfields.random_field_gen_KLE import RandomFieldGenKLE


class RandomFieldGenKLE1D(RandomFieldGenKLE):
    """Generator for 1d random fields based on KLE expansion

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
        # RandomFieldGenKLE1D standard constructor
        # call superclass constructor first
        super().__init__(
            marginal_distribution,
            corr_length,
            energy_frac,
            field_bbox,
            dimension,
            num_ex_term_per_dim,
            num_terms,
        )

        # sanity checks are done in superclass
        w_n = np.zeros((self.m, self.spatial_dim))
        # Compute roots of characteristic function for each dimension
        w_n[:] = self.compute_roots_of_characteristic_equation()
        self.w_n = w_n

        # sum of eigenvalues
        sum_lambda = 0
        # compute eigenvalues
        self.lambda_n = np.zeros((self.trunc_thres, 2))
        for k in range(self.trunc_thres):
            self.lambda_n[k, :] = [
                2 * self.corr_length / ((self.corr_length ** 2 * self.w_n[k] ** 2 + 1)),
                k,
            ]
            sum_lambda = sum_lambda + self.lambda_n[k, 0]

        if sum_lambda / self.largest_length < self.des_energy_frac:
            raise RuntimeError(
                'Energy fraction retained by KLE expansion is '
                ' only {}, not {}'.format(sum_lambda / self.largest_length, self.des_energy_frac)
            )

        self.act_energy_frac = sum_lambda / self.largest_length

    def gen_sample_gauss_field(self, loc, phase_angles):
        """ Generate sample of standard Gaussian field

        Compute realization of standard Gaussian field based on passed phase
        angles phase_angles and location

        Arguments:
            loc (np.array):             Locations at which to evaluate generated
                                        field
            phase_angles (np.array):    Realization of standard normal random
                                        phase angles
        Returns:
            np.array: values of the random field realization at loc
        """

        if len(phase_angles) is not self.stoch_dim:
            raise RuntimeError(
                'Number of random phase angles does not match ' 'stochastic dimension of the field!'
            )

        # if length(loc(1,:))~=1
        #    error('Error: Location vector must have one dimensions!')
        # end
        # use KLE expansion to compute random field values
        coeff = np.array(np.sqrt(self.lambda_n[:, 0]) * np.transpose(phase_angles))
        values = np.dot(self.compute_eigen_function_vec(loc, 0), coeff.T)

        return values
