"""TODO_doc."""

from scipy.stats import norm


class UnivariateRandomFieldSimulator:
    """Generator for samples of uni-variate random fields.

    Class for the generation of samples form Gaussian and
    non-Gaussian random fields using translation process theory.

    Attributes:
        spatial_dim (int): Spatial dimension of the field, i.e. 1,2, or 3.
        prob_dist (scipy.stats.rv_continuous): Marginal probability
            distribution of the field.
        autocov_type (): Autocovariance function of the field.
        truncated (bool): Is the field truncated?
        lower_bound (float): Lower cutoff value of the field.
        upper_bound (float): Upper cutoff value of the field.
        space_discretized (bool): Is the generation based on a spatial
            discretization, i.e. a grid?
        bounding_box (np.array): Domain over which samples of the
            field are generated.
        des_energy_frac (float): Desired percentage of variance retained
            by series approximation of (underlying) Gaussian Field.
        act_energy_frac (float): Actual percentage of variance retained by
            series approximation.
        stoch_dim (int): Stochastic dimension, i.e. number of random variables
            needed to generate samples.
    """

    def __init__(self, marginal_distribution):
        """TODO_doc.

        Args:
            marginal_distribution: TODO_doc
        """
        self.spatial_dim = None
        self.prob_dist = None
        self.autocov_type = None
        self.truncated = False
        self.lower_bound = 0
        self.upper_bound = 0
        self.space_discretized = False
        self.bounding_box = None
        self.des_energy_frac = None
        self.act_energy_frac = None
        self.stoch_dim = None

        # check whether we have a normal or lognormal distributon
        if (
            marginal_distribution.dist.name != 'norm'
            and marginal_distribution.dist.name != 'lognorm'
        ):
            raise RuntimeError('Error: marginal_distribution must be either Normal or Lognormal')
        self.prob_dist = marginal_distribution

    def gen_sample_gauss_field(self, loc, phase_angles):
        """TODO_doc: add a one-line explanation.

        This method generates sample of standard Gaussian random field and evaluates it
        at *loc*. The actual generation based on series expansion methods is
        implemented in the subclasses.

        *gen_sample_gauss_field(loc, phase_angles)* inputs:

        * *loc* location(s) at which realization of random field is evaluated
        * *phase_angles* random phase angles or amplitudes used to generate realization of random
        field.

        Args:
            loc (np.array): Location at which the field is evaluated
            phase_angles (np.array): Pseudo random phase angles for field generation

        Returns:
            np.array: Value of random field at locations *x*
        """
        raise NotImplementedError()

    def get_stoch_dim(self):
        """Return stochastic dimension of the field.

        Returns:
            int: Stochastic dimension of the field
        """
        return self.stoch_dim

    def evaluate_field_at_location(self, loc, phase_angles):
        """TODO_doc: add a one-line explanation.

        Generate sample of random field based on *phase_angles* and evaluate it at
        *loc*.

        Args:
            loc (np.array): Location at which the field is evaluated
            phase_angles (np.array): Pseudo random phase angles for field generation

        Returns:
            np.array: Value of random field at locations *loc*
        """
        values = self.gen_sample_gauss_field(loc, phase_angles)

        # translate field
        return self.prob_dist.ppf(norm.cdf(values, 0, 1))
