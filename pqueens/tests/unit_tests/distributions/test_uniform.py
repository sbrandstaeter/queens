"""Test-module for uniform distribution."""

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
def lower_bound_1d():
    """A possible left bound of interval."""
    return -1.0


@pytest.fixture(scope='module')
def upper_bound_1d():
    """A possible right bound of interval."""
    return 1.0


@pytest.fixture(scope='module')
def uniform_1d(lower_bound_1d, upper_bound_1d):
    """A uniform distribution."""
    distribution_options = {
        'distribution': 'uniform',
        'lower_bound': lower_bound_1d,
        'upper_bound': upper_bound_1d,
    }
    return from_config_create_distribution(distribution_options)


# ------------- multivariate --------------
@pytest.fixture(
    params=[
        [-2.0, -1.0],
        [[-1.0, -3.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 2.0], [1.0, 2.0], [1.0, 3.0], [2.0, 3.0]],
    ]
)
def sample_pos_2d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(scope='module')
def lower_bound_2d():
    """A possible left bound of interval."""
    return np.array([-1.0, -3.0])


@pytest.fixture(scope='module')
def upper_bound_2d():
    """A possible right bound of interval."""
    return np.array([1.0, 2.0])


@pytest.fixture(scope='module')
def uniform_2d(lower_bound_2d, upper_bound_2d):
    """A uniform distribution."""
    distribution_options = {
        'distribution': 'uniform',
        'lower_bound': lower_bound_2d,
        'upper_bound': upper_bound_2d,
    }
    return from_config_create_distribution(distribution_options)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------

# ------------- univariate --------------
@pytest.mark.unit_tests
def test_init_uniform_1d(uniform_1d, lower_bound_1d, upper_bound_1d):
    """Test init method of Uniform Distribution class."""
    width = np.array(upper_bound_1d - lower_bound_1d).reshape(1)
    lower_bound_1d = np.array(lower_bound_1d).reshape(1)
    upper_bound_1d = np.array(upper_bound_1d).reshape(1)
    mean_ref = scipy.stats.uniform.mean(loc=lower_bound_1d, scale=width).reshape(1)
    var_ref = scipy.stats.uniform.var(loc=lower_bound_1d, scale=width).reshape(1, 1)

    assert uniform_1d.dimension == 1
    np.testing.assert_allclose(uniform_1d.mean, mean_ref)
    np.testing.assert_allclose(uniform_1d.covariance, var_ref)
    np.testing.assert_equal(uniform_1d.lower_bound, lower_bound_1d)
    np.testing.assert_equal(uniform_1d.upper_bound, upper_bound_1d)
    np.testing.assert_equal(uniform_1d.width, width)


@pytest.mark.unit_tests
def test_init_uniform_1d_wrong_interval(lower_bound_1d):
    """Test init method of Uniform Distribution class."""
    with pytest.raises(ValueError, match=r'Lower bound must be smaller than upper bound*'):
        distribution_options = {
            'distribution': 'uniform',
            'lower_bound': lower_bound_1d,
            'upper_bound': lower_bound_1d - np.abs(lower_bound_1d),
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_cdf_uniform_1d(uniform_1d, lower_bound_1d, upper_bound_1d, sample_pos_1d):
    """Test cdf method of Uniform distribution class."""
    width = upper_bound_1d - lower_bound_1d
    ref_sol = scipy.stats.uniform.cdf(sample_pos_1d, loc=lower_bound_1d, scale=width).reshape(-1)
    np.testing.assert_allclose(uniform_1d.cdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_draw_uniform_1d(uniform_1d, lower_bound_1d, upper_bound_1d, mocker):
    """Test the draw method of uniform distribution."""
    sample = np.asarray(0.5 * (lower_bound_1d + upper_bound_1d)).reshape(1, 1)
    mocker.patch('numpy.random.uniform', return_value=sample)
    draw = uniform_1d.draw()
    np.testing.assert_equal(draw, sample)


@pytest.mark.unit_tests
def test_logpdf_uniform_1d(uniform_1d, lower_bound_1d, upper_bound_1d, sample_pos_1d):
    """Test pdf method of Uniform distribution class."""
    width = upper_bound_1d - lower_bound_1d
    ref_sol = scipy.stats.uniform.logpdf(sample_pos_1d, loc=lower_bound_1d, scale=width).reshape(-1)
    np.testing.assert_allclose(uniform_1d.logpdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_grad_logpdf_uniform_1d(uniform_1d, sample_pos_1d):
    """Test grad_logpdf method of uniform distribution class."""
    sample_pos_1d = sample_pos_1d.reshape(-1, 1)
    ref_sol = np.zeros(sample_pos_1d.shape)
    np.testing.assert_allclose(uniform_1d.grad_logpdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_pdf_uniform_1d(uniform_1d, lower_bound_1d, upper_bound_1d, sample_pos_1d):
    """Test pdf method of Uniform distribution class."""
    width = upper_bound_1d - lower_bound_1d
    ref_sol = scipy.stats.uniform.pdf(sample_pos_1d, loc=lower_bound_1d, scale=width).reshape(-1)
    np.testing.assert_allclose(uniform_1d.pdf(sample_pos_1d), ref_sol)


@pytest.mark.unit_tests
def test_ppf_uniform_1d(uniform_1d, lower_bound_1d, upper_bound_1d):
    """Test ppf method of Uniform distribution class."""
    quantile = 0.5
    width = upper_bound_1d - lower_bound_1d
    ref_sol = scipy.stats.uniform.ppf(quantile, loc=lower_bound_1d, scale=width).reshape(-1)
    np.testing.assert_allclose(uniform_1d.ppf(quantile), ref_sol)


# ------------- multivariate --------------
@pytest.mark.unit_tests
def test_init_uniform_2d(uniform_2d, lower_bound_2d, upper_bound_2d):
    """Test init method of Uniform Distribution class."""
    width = upper_bound_2d - lower_bound_2d
    mean_ref = np.array(
        [
            scipy.stats.uniform.mean(loc=lower_bound_2d[0], scale=width[0]),
            scipy.stats.uniform.mean(loc=lower_bound_2d[1], scale=width[1]),
        ]
    )
    var_ref = np.diag(
        [
            scipy.stats.uniform.var(loc=lower_bound_2d[0], scale=width[0]),
            scipy.stats.uniform.var(loc=lower_bound_2d[1], scale=width[1]),
        ]
    )

    assert uniform_2d.dimension == 2
    np.testing.assert_allclose(uniform_2d.mean, mean_ref)
    np.testing.assert_allclose(uniform_2d.covariance, var_ref)
    np.testing.assert_equal(uniform_2d.lower_bound, lower_bound_2d)
    np.testing.assert_equal(uniform_2d.upper_bound, upper_bound_2d)
    np.testing.assert_equal(uniform_2d.width, width)


@pytest.mark.unit_tests
def test_init_uniform_2d_wrong_interval(lower_bound_2d):
    """Test init method of Uniform Distribution class."""
    with pytest.raises(ValueError, match=r'Lower bound must be smaller than upper bound*'):
        distribution_options = {
            'distribution': 'uniform',
            'lower_bound': lower_bound_2d,
            'upper_bound': lower_bound_2d + np.array([0.0, -0.1]),
        }
        from_config_create_distribution(distribution_options)


@pytest.mark.unit_tests
def test_cdf_uniform_2d(uniform_2d, lower_bound_2d, upper_bound_2d, sample_pos_2d):
    """Test cdf method of Uniform distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    width = upper_bound_2d - lower_bound_2d
    ref_sol = scipy.stats.uniform.cdf(
        sample_pos_2d[:, 0], loc=lower_bound_2d[0], scale=width[0]
    ) * scipy.stats.uniform.cdf(sample_pos_2d[:, 1], loc=lower_bound_2d[1], scale=width[1])
    np.testing.assert_allclose(uniform_2d.cdf(sample_pos_2d), ref_sol)


@pytest.mark.unit_tests
def test_draw_uniform_2d(uniform_2d, lower_bound_2d, upper_bound_2d, mocker):
    """Test the draw method of uniform distribution."""
    sample = np.asarray(0.5 * (lower_bound_2d + upper_bound_2d)).reshape(1, 2)
    mocker.patch('numpy.random.uniform', return_value=sample)
    draw = uniform_2d.draw()
    np.testing.assert_equal(draw, sample)


@pytest.mark.unit_tests
def test_logpdf_uniform_2d(uniform_2d, lower_bound_2d, upper_bound_2d, sample_pos_2d):
    """Test pdf method of Uniform distribution class."""
    width = upper_bound_2d - lower_bound_2d
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    ref_sol = scipy.stats.uniform.logpdf(
        sample_pos_2d[:, 0], loc=lower_bound_2d[0], scale=width[0]
    ) + scipy.stats.uniform.logpdf(sample_pos_2d[:, 1], loc=lower_bound_2d[1], scale=width[1])
    np.testing.assert_allclose(uniform_2d.logpdf(sample_pos_2d), ref_sol)


@pytest.mark.unit_tests
def test_grad_logpdf_uniform_2d(uniform_2d, sample_pos_2d):
    """Test grad_logpdf method of uniform distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    ref_sol = np.zeros(sample_pos_2d.shape)
    np.testing.assert_allclose(uniform_2d.grad_logpdf(sample_pos_2d), ref_sol)


@pytest.mark.unit_tests
def test_pdf_uniform_2d(uniform_2d, lower_bound_2d, upper_bound_2d, sample_pos_2d):
    """Test pdf method of Uniform distribution class."""
    width = upper_bound_2d - lower_bound_2d
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    ref_sol = scipy.stats.uniform.pdf(
        sample_pos_2d[:, 0], loc=lower_bound_2d[0], scale=width[0]
    ) * scipy.stats.uniform.pdf(sample_pos_2d[:, 1], loc=lower_bound_2d[1], scale=width[1])
    np.testing.assert_allclose(uniform_2d.pdf(sample_pos_2d), ref_sol)


@pytest.mark.unit_tests
def test_ppf_uniform_2d(uniform_2d, lower_bound_2d, upper_bound_2d):
    """Test ppf method of Uniform distribution class."""
    with pytest.raises(ValueError, match='Method does not support multivariate distributions!'):
        uniform_2d.ppf(np.zeros(2))
