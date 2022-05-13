"""Test-module for normal distribution."""

import numpy as np
import pytest
import scipy.stats

from pqueens.distributions import from_config_create_distribution


# ------------- univariate --------------
@pytest.fixture(params=[-2.0, [-1.0, 0.0, 1.0, 2.0]])
def sample_pos_1d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(scope='module')
def mean_1d():
    """A possible scalar mean value."""
    return 1.0


@pytest.fixture(scope='module')
def covariance_1d():
    """A possible scalar variance value."""
    return 2.0


@pytest.fixture(scope='module')
def normal_1d(mean_1d, covariance_1d):
    """A valid normal distribution."""
    distribution_options = {
        'distribution': 'normal',
        'mean': mean_1d,
        'covariance': covariance_1d,
    }
    return from_config_create_distribution(distribution_options)


@pytest.fixture(scope='module')
def uncorrelated_vector_1d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution.

    as expected by a call to a Gaussian random number generator
    """
    vec = [[1.0]]
    return np.tile(vec, num_draws)


# ------------- multivariate --------------
@pytest.fixture(params=[[-2.0, -1.0, 0.0], [[-1.0, -3.0, 1.0], [-1.0, -1.0, -1.0]]])
def sample_pos_3d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(scope='module')
def mean_3d():
    """A possible mean vector."""
    return np.array([0.0, -1.0, 2.0])


@pytest.fixture(scope='module')
def low_chol_3d():
    """Lower triangular matrix of a Cholesky decomposition."""
    return np.array([[1.0, 0.0, 0.0], [0.1, 2.0, 0.0], [1.0, 0.8, 3.0]])


@pytest.fixture(scope='module')
def covariance_3d(low_chol_3d):
    """Recompose matrix based on given Cholesky decomposition."""
    return np.dot(low_chol_3d, low_chol_3d.T)


@pytest.fixture(scope='module')
def normal_3d(mean_3d, covariance_3d):
    """A multivariate normal distribution."""
    distribution_options = {
        'distribution': 'normal',
        'mean': mean_3d,
        'covariance': covariance_3d,
    }
    return from_config_create_distribution(distribution_options)


@pytest.fixture(scope='module', params=[1, 4])
def num_draws(request):
    """Number of samples to draw from distribution."""
    return request.param


@pytest.fixture(scope='module')
def uncorrelated_vector_3d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution.

    as expected by a call to a Gaussian random number generator
    """
    vec = [[1.0], [-2.0], [3.0]]
    return np.tile(vec, num_draws)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------

# ------------- univariate --------------
@pytest.mark.unit_tests
def test_init_normal_1d(normal_1d, mean_1d, covariance_1d):
    """Test init method of Normal Distribution class."""
    assert normal_1d.dimension == 1
    np.testing.assert_equal(normal_1d.mean, np.array(mean_1d).reshape(1))
    np.testing.assert_equal(normal_1d.covariance, np.array(covariance_1d).reshape(1, 1))


@pytest.mark.unit_tests
def test_init_normal_1d_incovariance(mean_1d, covariance_1d):
    """Test init method of Normal Distribution class."""
    with pytest.raises(np.linalg.LinAlgError, match=r'Cholesky decomposition failed *'):
        distribution_options = {
            'distribution': 'normal',
            'mean': mean_1d,
            'covariance': -covariance_1d,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_cdf_normal_1d(normal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test cdf method of Normal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.norm.cdf(sample_pos_1d, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.cdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_draw_normal_1d(normal_1d, mean_1d, covariance_1d, uncorrelated_vector_1d, mocker):
    """Test the draw method of normal distribution."""
    mocker.patch('numpy.random.randn', return_value=uncorrelated_vector_1d)
    draw = normal_1d.draw()
    ref_sol = mean_1d + covariance_1d ** (1 / 2) * uncorrelated_vector_1d.T
    np.testing.assert_equal(draw, ref_sol)


@pytest.mark.unit_tests
def test_logpdf_normal_1d(normal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test pdf method of Normal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.norm.logpdf(sample_pos_1d, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.logpdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_pdf_normal_1d(normal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test pdf method of Normal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.norm.pdf(sample_pos_1d, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.pdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_ppf_normal_1d(normal_1d, mean_1d, covariance_1d):
    """Test ppf method of Normal Distribution distribution class."""
    std = np.sqrt(covariance_1d)
    quantile = 0.5
    ref_sol = scipy.stats.norm.ppf(quantile, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.ppf(quantile), ref_sol)


# ------------- multivariate --------------
@pytest.mark.unit_tests
def test_init_normal_3d(normal_3d, mean_3d, covariance_3d):
    """Test init method of Normal Distribution class."""
    assert normal_3d.dimension == 3
    np.testing.assert_equal(normal_3d.mean, mean_3d.reshape(3))
    np.testing.assert_equal(normal_3d.covariance, covariance_3d.reshape(3, 3))


@pytest.mark.unit_tests
def test_cdf_normal_3d(normal_3d, mean_3d, covariance_3d, sample_pos_3d):
    """Test cdf method of Normal Distribution distribution class."""
    sample_pos_3d = sample_pos_3d.reshape(-1, 3)
    ref_sol = scipy.stats.multivariate_normal.cdf(
        sample_pos_3d, mean=mean_3d, cov=covariance_3d
    ).reshape(-1)
    # high tolerance due to numerical errors
    np.testing.assert_allclose(normal_3d.cdf(sample_pos_3d), ref_sol, rtol=1e-3)


@pytest.mark.unit_tests
def test_draw_normal_3d(normal_3d, mean_3d, low_chol_3d, uncorrelated_vector_3d, mocker):
    """Test the draw method of normal distribution."""
    mocker.patch('numpy.random.randn', return_value=uncorrelated_vector_3d)
    draw = normal_3d.draw()
    ref_sol = mean_3d + np.dot(low_chol_3d, uncorrelated_vector_3d).T
    np.testing.assert_equal(draw, ref_sol)


@pytest.mark.unit_tests
def test_logpdf_normal_3d(normal_3d, mean_3d, covariance_3d, sample_pos_3d):
    """Test pdf method of Normal Distribution distribution class."""
    sample_pos_3d = sample_pos_3d.reshape(-1, 3)
    ref_sol = scipy.stats.multivariate_normal.logpdf(sample_pos_3d, mean=mean_3d, cov=covariance_3d)
    np.testing.assert_allclose(normal_3d.logpdf(sample_pos_3d), ref_sol)


@pytest.mark.unit_tests
def test_pdf_normal_3d(normal_3d, mean_3d, covariance_3d, sample_pos_3d):
    """Test pdf method of Normal Distribution distribution class."""
    sample_pos_3d = sample_pos_3d.reshape(-1, 3)
    ref_sol = scipy.stats.multivariate_normal.pdf(sample_pos_3d, mean=mean_3d, cov=covariance_3d)
    np.testing.assert_allclose(normal_3d.pdf(sample_pos_3d), ref_sol)


@pytest.mark.unit_tests
def test_ppf_normal_3d(normal_3d, mean_3d, covariance_3d):
    """Test ppf method of Normal Distribution distribution class."""
    with pytest.raises(ValueError, match='Method does not support multivariate distributions!'):
        normal_3d.ppf(np.zeros(2))


@pytest.mark.unit_tests
def test_init_normal_wrong_dimension(mean_3d):
    """Test ValueError of init method of Normal Distribution class."""
    covariance = np.array([[[1.0, 0.1], [1.0, 0.1]], [[0.2, 2.0], [0.2, 2.0]]])
    with pytest.raises(ValueError, match=r'Provided covariance is not a matrix.*'):
        distribution_options = {
            'distribution': 'normal',
            'mean': mean_3d,
            'covariance': covariance,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_init_normal_not_quadratic(mean_3d):
    """Test ValueError of init method of Normal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0], [3.0, 0.3]])
    with pytest.raises(ValueError, match=r'Provided covariance matrix is not quadratic.*'):
        distribution_options = {
            'distribution': 'normal',
            'mean': mean_3d,
            'covariance': covariance,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_init_normal_not_symmetric(mean_3d):
    """Test ValueError of init method of Normal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0]])
    with pytest.raises(ValueError, match=r'Provided covariance matrix is not symmetric.*'):
        distribution_options = {
            'distribution': 'normal',
            'mean': mean_3d,
            'covariance': covariance,
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_init_lognormal_not_symmetric():
    """Test ValueError of init method of Normal Distribution class."""
    covariance = np.array([[1.0, 0.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match=r'Dimension of mean vector and covariance matrix do not*'):
        distribution_options = {
            'distribution': 'lognormal',
            'normal_mean': np.zeros(3),
            'normal_covariance': covariance,
        }
        from_config_create_distribution(distribution_options)
