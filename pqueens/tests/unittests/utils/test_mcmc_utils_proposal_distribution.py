"""
Test-module for distributions of mcmc_utils module

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest
import scipy.stats

from pqueens.utils.mcmc_utils import create_proposal_distribution
from pqueens.utils.mcmc_utils import BetaDistribution
from pqueens.utils.mcmc_utils import LogNormal
from pqueens.utils.mcmc_utils import NormalProposal
from pqueens.utils.mcmc_utils import Uniform


########################### UNIFORM ####################################
@pytest.fixture(params=[-2.0, -1.0, 0.0, 1.0, 2.0])
def sample_pos(request):
    """ sample position to be evaluated """
    return request.param


@pytest.fixture(scope='module')
def a():
    """ A possible left bound of interval. """

    return -1.0


@pytest.fixture(scope='module')
def b():
    """ A possible right bound of interval. """

    return 1.0


@pytest.fixture(scope='module')
def uniform_distr(a, b):
    """ A uniform distribution. """

    return Uniform(a=a, b=b)


############################ NORMAL ####################################
#### univariate ####
@pytest.fixture(scope='module')
def valid_mean_value():
    """ A possible scalar mean value. """

    return np.array([1.0])


@pytest.fixture(scope='module')
def valid_var_value():
    """ A possible scalar variance value. """

    return np.array([2.0])


@pytest.fixture(scope='module')
def univariate_normal(valid_mean_value, valid_var_value):
    """ A valid normal distribution. """

    return NormalProposal(mean=valid_mean_value, covariance=valid_var_value)


#### multivariate ####
@pytest.fixture(scope='module')
def valid_mean_vector():
    """ A possible mean vector. """

    return np.array([0.0, 1.0, 2.0])


@pytest.fixture(scope='module')
def valid_lower_cholesky():
    """ Lower triangular matrix of a Cholesky decomposition. """

    return np.array([[1.0, 0.0, 0.0], [0.1, 2.0, 0.0], [1.0, 0.8, 3.0]])


@pytest.fixture(scope='module')
def valid_covariance_matrix(valid_lower_cholesky):
    """ Recompose matrix based on given Cholesky decomposition. """

    return np.dot(valid_lower_cholesky, valid_lower_cholesky.T)


@pytest.fixture(scope='module')
def multivariate_normal(valid_mean_vector, valid_covariance_matrix):
    """ A multivariate normal distribution. """
    return NormalProposal(mean=valid_mean_vector, covariance=valid_covariance_matrix)


@pytest.fixture(scope='module')
def invalid_dimension_covariance_matrix():
    """
    A numpy array of dimension 3.

    valid covariance is either a scalar (dimension 1)
    or a matrix (dimension 2)
    """

    return np.array([[[1.0, 0.1], [1.0, 0.1]], [[0.2, 2.0], [0.2, 2.0]]])


@pytest.fixture(scope='module')
def invalid_rectangular_covariance_matrix():
    """
    Rectangular matrix to test ValueError of covariance matrix.

    a valid covariance matrix has to be quadratic
    """

    return np.array([[1.0, 0.1], [0.2, 2.0], [3.0, 0.3]])


@pytest.fixture(scope='module')
def invalid_nonsymmetric_covariance_matrix():
    """
    A non-symmetric matrix.

    valid covariance matrix has to be symmetric
    """
    return np.array([[1.0, 0.1], [0.2, 2.0]])


@pytest.fixture(scope='module', params=[1, 4])
def num_draws(request):
    """ number of samples to draw from distribution. """
    return request.param


@pytest.fixture(scope='module')
def uncorrelated_vector(num_draws):
    """
    A vector of uncorrelated samples from standard normal distribution.

    as expected by a call to a Gaussian random number generator
    """
    vec = [[1.0], [2.0], [3.0]]
    return np.tile(vec, num_draws)


########################### LOGNORMAL ##################################
# use values of the univariate normal
@pytest.fixture(scope='module')
def lognormal_distr(valid_mean_value, valid_var_value):
    """ A lognormal distribution. """
    return LogNormal(mu=valid_mean_value, sigma=np.sqrt(valid_var_value))


# -------- some fixtures for the beta distribution
@pytest.fixture()
def default_beta_distr():
    a = 2.0
    b = 2.0
    loc = 1
    scale = 2
    return BetaDistribution(a, b, loc, scale)


########################################################################
############################# TESTS ####################################
########################################################################


############################## Uniform #################################
@pytest.mark.unit_tests
def test_init_Uniform(uniform_distr, a, b):
    """ Test init method of Uniform class. """

    width = b - a
    mean_ref = scipy.stats.uniform.mean(loc=a, scale=width)
    var_ref = scipy.stats.uniform.var(loc=a, scale=width)

    assert uniform_distr.dimension == 1
    np.testing.assert_allclose(uniform_distr.mean, mean_ref)
    np.testing.assert_allclose(uniform_distr.covariance, var_ref)
    np.testing.assert_allclose(uniform_distr.a, a)
    np.testing.assert_allclose(uniform_distr.b, b)
    np.testing.assert_allclose(uniform_distr.width, width)


@pytest.mark.unit_tests
def test_init_Uniform_wrong_interval(a):
    """ Test init method of Uniform class. """

    with pytest.raises(ValueError, match=r'Invalid interval.*'):
        Uniform(a, a - np.abs(a))


@pytest.mark.unit_tests
def test_cdf_Uniform(uniform_distr, a, b):
    """ Test cdf method of Uniform distribution class. """

    sample_pos = 0.5 * (a + b)
    ref_sol = 0.5

    np.testing.assert_allclose(uniform_distr.cdf(sample_pos), ref_sol)


@pytest.mark.unit_tests
def test_draw_Uniform(uniform_distr, a, b, mocker):
    """ Test the draw method of uniform distribution. """

    sample = np.asarray(0.5 * (a + b))
    mocker.patch('numpy.random.uniform', return_value=sample)
    draw = uniform_distr.draw()

    np.testing.assert_allclose(draw, sample)


@pytest.mark.unit_tests
def test_logpdf_Uniform(uniform_distr, a, b, sample_pos):
    """ Test pdf method of Uniform distribution class. """

    # in QUEENS we define the support of the uniform distribution as [a, b)
    # However, in scipy.stats it is [a, b]
    # we circumvent this by setting the width of the interval as one float smaller
    width = b - a
    ref_sol = scipy.stats.uniform.logpdf(sample_pos, loc=a, scale=np.nextafter(width, -1))

    np.testing.assert_allclose(uniform_distr.logpdf(sample_pos), ref_sol)


@pytest.mark.unit_tests
def test_pdf_Uniform(uniform_distr, a, b, sample_pos):
    """ Test pdf method of Uniform distribution class. """

    # in QUEENS we define the support of the uniform distribution as [a, b)
    # However, in scipy.stats it is [a, b]
    # we circumvent this by setting the width of the interval as one float smaller
    width = b - a
    ref_sol = scipy.stats.uniform.pdf(sample_pos, loc=a, scale=np.nextafter(width, -1))

    np.testing.assert_allclose(uniform_distr.pdf(sample_pos), ref_sol)


@pytest.mark.unit_tests
def test_ppf_Uniform(uniform_distr, a, b):
    """ Test ppf method of Uniform distribution class. """

    quantile = 0.5
    ref_sol = 0.5 * (a + b)

    np.testing.assert_allclose(uniform_distr.ppf(quantile), ref_sol)


############################ LogNormal #################################
@pytest.mark.unit_tests
def test_init_LogNormal(lognormal_distr, valid_mean_value, valid_var_value):
    """ Test init method of LogNormal class. """

    sigma = np.sqrt(valid_var_value)

    mean_ref = scipy.stats.lognorm.mean(s=sigma, scale=np.exp(valid_mean_value))
    var_ref = scipy.stats.lognorm.var(s=sigma, scale=np.exp(valid_mean_value))

    assert lognormal_distr.dimension == 1
    np.testing.assert_allclose(lognormal_distr.mu, valid_mean_value)
    np.testing.assert_allclose(lognormal_distr.sigma, sigma)
    np.testing.assert_allclose(lognormal_distr.mean, mean_ref)
    np.testing.assert_allclose(lognormal_distr.covariance, var_ref)


@pytest.mark.unit_tests
def test_init_LogNormal_invalid(valid_mean_value, valid_var_value):
    """ Test init method of LogNormal class. """
    with pytest.raises(ValueError, match=r'Sigma has to be positiv.*'):
        LogNormal(valid_mean_value, -valid_var_value)


@pytest.mark.unit_tests
def test_cdf_Lognormal(valid_mean_value, valid_var_value):
    """
    Test cdf method of Lognormal class (univariate case).

    We know the analytical value for a lognormal with mu=0.0
    (see e.g. Wikipedia).
    """

    valid_std = np.sqrt(valid_var_value)
    lognormal_distr = LogNormal(mu=0.0, sigma=valid_std)

    # sample to be evaluated
    sample_pos = 1.0
    ref_sol = 0.5

    np.testing.assert_allclose(lognormal_distr.cdf(sample_pos), ref_sol)


@pytest.mark.unit_tests
def test_draw_LogNormal(lognormal_distr, mocker):
    """ Test the draw method of uniform distribution. """

    sample = np.asarray(0.5)
    mocker.patch('numpy.random.lognormal', return_value=sample)
    draw = lognormal_distr.draw()

    np.testing.assert_allclose(draw, sample)


@pytest.mark.unit_tests
def test_pdf_Lognormal(lognormal_distr, valid_mean_value, valid_var_value):
    """ Test pdf method of Lognormal class (univariate case). """

    valid_std = np.sqrt(valid_var_value)

    # sample to be evaluated
    x = np.ones(1)
    reference_solution = scipy.stats.lognorm.pdf(x, s=valid_std, scale=np.exp(valid_mean_value))

    np.testing.assert_allclose(lognormal_distr.pdf(x), reference_solution)


@pytest.mark.unit_tests
def test_logpdf_Lognormal(lognormal_distr, valid_mean_value, valid_var_value):
    """ Test logpdf method of Lognormal class (univariate case). """

    valid_std = np.sqrt(valid_var_value)

    # sample to be evaluated
    x = np.ones(1)
    reference_solution = scipy.stats.lognorm.logpdf(x, s=valid_std, scale=np.exp(valid_mean_value))

    np.testing.assert_allclose(lognormal_distr.logpdf(x), reference_solution)


@pytest.mark.unit_tests
def test_ppf_Lognormal(valid_mean_value, valid_var_value):
    """
    Test ppf method of Lognormal class (univariate case).

    We know the analytical value for a lognormal with mu=0.0
    (see e.g. Wikipedia).
    """

    valid_std = np.sqrt(valid_var_value)
    lognormal_distr = LogNormal(mu=0.0, sigma=valid_std)

    # sample to be evaluated
    quantile = 0.5
    ref_sol = 1.0

    np.testing.assert_allclose(lognormal_distr.ppf(quantile), ref_sol)


############################## Normal ##################################
# univariate
@pytest.mark.unit_tests
def test_init_NormalProposal_univariate(univariate_normal, valid_mean_value, valid_var_value):
    """ Test init method of NormalProposal class (univariate case). """

    # cholesky_decomp_covar_mat decomposition of a scalar is root of scalar
    lower_cholesky = np.atleast_2d(np.sqrt(valid_var_value))

    assert univariate_normal.dimension == 1
    np.testing.assert_allclose(univariate_normal.mean, valid_mean_value)
    np.testing.assert_allclose(univariate_normal.covariance, valid_var_value)
    np.testing.assert_allclose(univariate_normal.low_chol, lower_cholesky)


@pytest.mark.unit_tests
def test_cdf_NormalProposal_univariate(valid_mean_value, univariate_normal):
    """ Test cdf method of NormalProposal class (univariate case). """

    # sample to be evaluated
    sample_pos = valid_mean_value
    ref_sol = 0.5

    np.testing.assert_allclose(univariate_normal.cdf(sample_pos), ref_sol)


@pytest.mark.unit_tests
def test_draw_NormalProposal_univariate(
    univariate_normal, valid_mean_value, valid_var_value, num_draws, mocker
):
    """ Test the draw method of normal distribution. """

    standard_normal_sample = np.array([[0.1]] * num_draws)
    mocker.patch('numpy.random.randn', return_value=standard_normal_sample)
    normal_sample = valid_mean_value + np.sqrt(valid_var_value) * standard_normal_sample
    univariate_draw = univariate_normal.draw(num_draws=num_draws)

    np.testing.assert_allclose(univariate_draw, normal_sample)


@pytest.mark.unit_tests
def test_logpdf_NormalProposal_univariate(univariate_normal, valid_mean_value, valid_var_value):
    """ Test logpdf method of NormalProposal class (univariate case). """

    # sample to be evaluated
    x = np.ones(1)
    ref_sol = scipy.stats.norm.logpdf(x, loc=valid_mean_value, scale=np.sqrt(valid_var_value))

    np.testing.assert_allclose(univariate_normal.logpdf(x), ref_sol)


@pytest.mark.unit_tests
def test_pdf_NormalProposal_univariate(univariate_normal, valid_mean_value, valid_var_value):
    """ Test pdf method of NormalProposal class (univariate case). """

    # sample to be evaluated
    x = np.ones(1)
    ref_sol = scipy.stats.norm.pdf(x, loc=valid_mean_value, scale=np.sqrt(valid_var_value))

    np.testing.assert_allclose(univariate_normal.pdf(x), ref_sol)


@pytest.mark.unit_tests
def test_ppf_NormalProposal_univariate(univariate_normal, valid_mean_value, valid_var_value):
    """ Test ppf method of NormalProposal class (univariate case). """
    quantile = 0.5
    ref_sol = valid_mean_value

    np.testing.assert_allclose(univariate_normal.ppf(quantile), ref_sol)


# multivariate
@pytest.mark.unit_tests
def test_init_NormalProposal_wrong_dimension(
    valid_mean_vector, invalid_dimension_covariance_matrix
):
    """ Test ValueError of init method of NormalProposal class. """

    with pytest.raises(ValueError, match=r'Covariance.*wrong dimension.'):
        NormalProposal(valid_mean_vector, invalid_dimension_covariance_matrix)


@pytest.mark.unit_tests
def test_init_NormalProposal_not_quadratic(
    valid_mean_vector, invalid_rectangular_covariance_matrix
):
    """ Test ValueError of init method of NormalProposal class. """

    with pytest.raises(ValueError, match=r'.*Covariance.*not quadratic.*'):
        NormalProposal(valid_mean_vector, invalid_rectangular_covariance_matrix)


@pytest.mark.unit_tests
def test_init_NormalProposal_not_symmetric(
    valid_mean_vector, invalid_nonsymmetric_covariance_matrix
):
    """ Test ValueError of init method of NormalProposal class. """

    with pytest.raises(ValueError, match=r'.*Covariance.*not symmetric.*'):
        NormalProposal(valid_mean_vector, invalid_nonsymmetric_covariance_matrix)


@pytest.mark.unit_tests
def test_init_NormalProposal_multivariate(
    multivariate_normal, valid_mean_vector, valid_covariance_matrix, valid_lower_cholesky
):
    """ Test init method of NormalProposal class (multivariate case). """

    assert multivariate_normal.dimension is valid_covariance_matrix.shape[0]
    np.testing.assert_allclose(multivariate_normal.mean, valid_mean_vector)
    np.testing.assert_allclose(multivariate_normal.covariance, valid_covariance_matrix)
    np.testing.assert_allclose(multivariate_normal.low_chol, valid_lower_cholesky)


@pytest.mark.unit_tests
def test_draw_NormalProposal_multivariate(
    multivariate_normal,
    valid_mean_vector,
    valid_lower_cholesky,
    uncorrelated_vector,
    num_draws,
    mocker,
):
    """ Test the draw method of normal proposal distribution. """

    mocker.patch('numpy.random.randn', return_value=uncorrelated_vector)
    correlated_vector = (
        valid_mean_vector.reshape(valid_mean_vector.shape[0], 1)
        + np.dot(valid_lower_cholesky, uncorrelated_vector)
    ).T
    multivariate_draw = multivariate_normal.draw(num_draws=num_draws)

    np.testing.assert_allclose(multivariate_draw, correlated_vector)


@pytest.mark.unit_tests
def test_cdf_NormalProposal_multivariate(
    multivariate_normal, valid_mean_vector, valid_covariance_matrix
):
    """ Test cdf method of NormalProposal class (multivariate case). """
    sample_pos = valid_mean_vector
    ref_sol = scipy.stats.multivariate_normal.cdf(
        sample_pos, mean=valid_mean_vector, cov=valid_covariance_matrix
    )

    np.testing.assert_allclose(multivariate_normal.cdf(sample_pos), ref_sol, rtol=1e-4)


@pytest.mark.unit_tests
def test_pdf_NormalProposal_multivariate(
    multivariate_normal, valid_mean_vector, valid_covariance_matrix
):
    """ Test pdf method of NormalProposal class (multivariate case). """

    # sample to be evaluated
    x = np.ones(3)
    reference_solution = scipy.stats.multivariate_normal.pdf(
        x, mean=valid_mean_vector, cov=valid_covariance_matrix
    )

    np.testing.assert_allclose(multivariate_normal.pdf(x), reference_solution)


@pytest.mark.unit_tests
def test_logpdf_NormalProposal_multivariate(
    multivariate_normal, valid_mean_vector, valid_covariance_matrix
):
    """ Test logpdf method of NormalProposal class (multivariate case). """

    # sample to be evaluated
    x = np.ones(3)
    reference_solution = scipy.stats.multivariate_normal.logpdf(
        x, mean=valid_mean_vector, cov=valid_covariance_matrix
    )

    np.testing.assert_allclose(multivariate_normal.logpdf(x), reference_solution)


@pytest.mark.unit_tests
def test_ppf_NormalProposal_multivariate(multivariate_normal):
    """ Test ppf method of NormalProposal class (multivariate case). """
    with pytest.raises(RuntimeError):
        multivariate_normal.ppf(q=0.5)


# ---- some quick analytic tests for the beta distribution
@pytest.mark.unit_tests
def test_pdf_beta(default_beta_distr):
    x = 2.0
    pdf_value = default_beta_distr.pdf(x)
    expected_pdf_value = 0.75
    np.testing.assert_almost_equal(pdf_value, expected_pdf_value, decimal=4)


@pytest.mark.unit_tests
def test_cdf_beta(default_beta_distr):
    x = 2.0
    cdf_value = default_beta_distr.cdf(x)
    expected_cdf_value = 0.5
    np.testing.assert_almost_equal(cdf_value, expected_cdf_value, decimal=4)


####################### Create Distribution ############################
@pytest.mark.unit_tests
def test_create_proposal_distribution_normal(valid_mean_vector, valid_covariance_matrix):
    """ Test creation routine of proposal distribution objects. """

    normal_options = {
        'distribution': 'normal',
        'distribution_parameter': [valid_mean_vector, valid_covariance_matrix],
    }

    normal_proposal = create_proposal_distribution(normal_options)

    assert isinstance(normal_proposal, NormalProposal)


@pytest.mark.unit_tests
def test_create_proposal_distribution_invalid():
    """ Test creation routine of proposal distribution objects. """

    invalid_options = {'distribution': 'UnsupportedType', 'distribution_parameter': [0.0, 1.0]}

    with pytest.raises(ValueError, match=r'.*type not supported.*'):
        create_proposal_distribution(invalid_options)
