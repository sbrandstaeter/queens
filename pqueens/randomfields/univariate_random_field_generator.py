from scipy.stats import norm


class UnivariateRandomFieldSimulator(object):
    """Generator for samples of univariate random fields.

    Class for the generation of samples form Gaussian and
    and non-Gaussian random fields using thranslation process theory

    Attributes:
        spatial_dim (int): spatial dimension of the field, i.e., 1,2, or 3
        prob_dist (scipy.stats.rv_continuous): marginal probability
            distribution of the field
        autocov_type (): autocovariance function of the field
        truncated (bool): is the field truncated
        lower_bound (float) lower cutoff value of the field
        upper_bound (float) upper cutoff value of the field
        space_discretized (bool): is the generation based on a spatial
            discretization, i.e., a grid
        bounding_box (np.array): domain over which samples of the
            field are generated
        des_energy_frac (float): desired percentage of variance retained
            by series approximation of (underlying) Gaussian Field
        act_energy_frac (float): actual percentage of variance retained by
            series approximation
        stoch_dim (int): stochastic dimension, i.e., number of random variable
            needed to generate samples

    """
    def __init__(self, marginal_distribution):
        """
        Args:
            spatial_dim (int): spatial dimension of the field, i.e., 1,2, or 3
            prob_dist (): marginal probability distribution of the field
            autocov_type (): autocovariance function of the field
            truncated (bool): is the field truncated
            lower_bound (float) lower cutoff value of the field
            upper_bound (float) upper cutoff value of the field
            space_discretized (bool): is the generation based on a spatial
                discretization, i.e., a grid
            bounding_box (np.array): domain over which samples of the
                field are generated
            des_energy_frac (float): desired percentage of variance retained
                by series approximation of (underlying) Gaussian Field
            act_energy_frac (float): actual percentage of variance retained by
                series approximation
            stoch_dim (int): stochastic dimension, i.e., number of random
                variable needed to generate samples

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
        if (marginal_distribution.dist.name is not 'norm' and
            marginal_distribution.dist.name is not 'lognorm'):
            raise RuntimeError('Error: marginal_distribution must be either '
                               'Normal or Lognormal')
        self.prob_dist = marginal_distribution

    def gen_sample_gauss_field(self, x, xi):
        """
        gen_sample_gauss_field(x, xi)
        input x location(s) at which realization of random field is evaluated
        xi random phase angles or amplitudes used to generate realization of
        random field. This mehtod generates sample of standard Gaussian random
        field and evaluates it at x. The actual generation based on series
        expansion methods is implemented in the subclasses.

        Args:
            x (np.array): location at which field is evaluated
            xi (np.array): pseudo random phase angles for field generation

        Returns:
            np.array: value of random field at locations x

        """
        raise NotImplementedError()

    def get_stoch_dim(self):
        """return stochastic dimension of the field

        Returns:
            int: stochastic dimension of the field
        """
        return self.stoch_dim

    def evaluate_field_at_location(self, x, xi):
        """Generate sample of random field based on xi and evaluate it at x

        Args:
            x (np.array): location at which field is evaluated
            xi (np.array): pseudo random phase angles for field generation

        Returns:
            np.array: value of random field at locations x
        """
        values = self.gen_sample_gauss_field(x, xi)

        # translate field
        return self.prob_dist.ppf(norm.cdf(values, 0, 1))
