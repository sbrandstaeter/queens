"""Test-module for exponential distribution."""

import numpy as np
import pytest
import scipy.stats

from pqueens.distributions.exponential import ExponentialDistribution


# ------------- univariate --------------
@pytest.fixture(params=[-2.0, [-1.0, 0.0, 1.0, 2.0]])
def sample_pos_1d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="rate_1d", scope='module')
def fixture_rate_1d():
    """A possible rate."""
    return 2.0


@pytest.fixture(name="exponential_1d", scope='module')
def fixture_exponential_1d(rate_1d):
    """An exponential distribution."""
    return ExponentialDistribution(rate=rate_1d)


# ------------- multivariate --------------
@pytest.fixture(
    params=[
        [3.0, 1.0],
        [[-1.0, -3.0], [-1.0, 0.0], [-1.0, 1.0], [0.0, 0.0], [0.0, 2.0], [1.0, 2.0]],
    ]
)
def sample_pos_2d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="rate_2d", scope='module')
def fixture_rate_2d():
    """A possible rate."""
    return np.array([1.0, 0.5])


@pytest.fixture(name="exponential_2d", scope='module')
def fixture_exponential_2d(rate_2d):
    """An exponential distribution."""
    return ExponentialDistribution(rate=rate_2d)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------


# ------------- univariate --------------
def test_init_exponential_1d(exponential_1d, rate_1d):
    """Test init method of Exponential Distribution class."""
    rate_1d = np.array(rate_1d).reshape(1)
    scale_1d = np.array(1 / rate_1d).reshape(1)
    mean_ref = scipy.stats.expon.mean(loc=0, scale=scale_1d).reshape(1)
    var_ref = scipy.stats.expon.var(loc=0, scale=scale_1d).reshape(1, 1)

    assert exponential_1d.dimension == 1
    np.testing.assert_allclose(exponential_1d.mean, mean_ref)
    np.testing.assert_allclose(exponential_1d.covariance, var_ref)
    np.testing.assert_equal(exponential_1d.rate, rate_1d)
    np.testing.assert_equal(exponential_1d.scale, scale_1d)


def test_init_exponential_1d_wrong_rate(rate_1d):
    """Test init method of Exponential Distribution class."""
    with pytest.raises(ValueError, match=r'The parameter \'rate\' has to be positive.*'):
        ExponentialDistribution(rate=-rate_1d)


def test_cdf_exponential_1d(exponential_1d, rate_1d, sample_pos_1d):
    """Test cdf method of Exponential distribution class."""
    scale_1d = 1 / rate_1d
    ref_sol = scipy.stats.expon.cdf(sample_pos_1d, loc=0, scale=scale_1d).reshape(-1)
    np.testing.assert_allclose(exponential_1d.cdf(sample_pos_1d), ref_sol)


def test_draw_exponential_1d(exponential_1d, rate_1d):
    """Test the draw method of exponential distribution."""
    np.random.seed(1)
    sample = np.random.exponential(scale=1 / rate_1d, size=1).reshape(-1, 1)
    np.random.seed(1)
    draw = exponential_1d.draw()
    np.testing.assert_equal(draw, sample)


def test_logpdf_exponential_1d(exponential_1d, rate_1d, sample_pos_1d):
    """Test pdf method of Exponential distribution class."""
    scale_1d = 1 / rate_1d
    ref_sol = scipy.stats.expon.logpdf(sample_pos_1d, loc=0, scale=scale_1d).reshape(-1)
    np.testing.assert_allclose(exponential_1d.logpdf(sample_pos_1d), ref_sol)


def test_grad_logpdf_exponential_1d(exponential_1d, sample_pos_1d):
    """Test *grad_logpdf* method of exponential distribution class."""
    sample_pos_1d = sample_pos_1d.reshape(-1, 1)
    condition = (sample_pos_1d >= 0).all(axis=1).reshape(-1, 1)
    ref_sol = np.where(condition, -exponential_1d.rate, np.nan)
    np.testing.assert_allclose(exponential_1d.grad_logpdf(sample_pos_1d), ref_sol)


def test_pdf_exponential_1d(exponential_1d, rate_1d, sample_pos_1d):
    """Test pdf method of Exponential distribution class."""
    scale_1d = 1 / rate_1d
    ref_sol = scipy.stats.expon.pdf(sample_pos_1d, loc=0, scale=scale_1d).reshape(-1)
    np.testing.assert_allclose(exponential_1d.pdf(sample_pos_1d), ref_sol)


def test_ppf_exponential_1d(exponential_1d, rate_1d):
    """Test ppf method of Exponential distribution class."""
    quantile = 0.5
    scale_1d = 1 / rate_1d
    ref_sol = scipy.stats.expon.ppf(quantile, loc=0, scale=scale_1d).reshape(-1)
    np.testing.assert_allclose(exponential_1d.ppf(quantile), ref_sol)


# ------------- multivariate --------------
def test_init_exponential_2d(exponential_2d, rate_2d):
    """Test init method of Exponential Distribution class."""
    scale_2d = 1 / rate_2d
    mean_ref = np.array(
        [
            scipy.stats.expon.mean(loc=0, scale=scale_2d[0]),
            scipy.stats.expon.mean(loc=0, scale=scale_2d[1]),
        ]
    )
    var_ref = np.diag(
        [
            scipy.stats.expon.var(loc=0, scale=scale_2d[0]),
            scipy.stats.expon.var(loc=0, scale=scale_2d[1]),
        ]
    )

    assert exponential_2d.dimension == 2
    np.testing.assert_allclose(exponential_2d.mean, mean_ref)
    np.testing.assert_allclose(exponential_2d.covariance, var_ref)
    np.testing.assert_equal(exponential_2d.rate, rate_2d)
    np.testing.assert_equal(exponential_2d.scale, scale_2d)


def test_init_exponential_2d_wrong_rate():
    """Test init method of Exponential Distribution class."""
    with pytest.raises(ValueError, match=r'The parameter \'rate\' has to be positive.*'):
        ExponentialDistribution(rate=np.array([-1, 1]))


def test_cdf_exponential_2d(exponential_2d, rate_2d, sample_pos_2d):
    """Test cdf method of Exponential distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    scale_2d = 1 / rate_2d
    ref_sol = scipy.stats.expon.cdf(
        sample_pos_2d[:, 0], loc=0, scale=scale_2d[0]
    ) * scipy.stats.expon.cdf(sample_pos_2d[:, 1], loc=0, scale=scale_2d[1])
    np.testing.assert_allclose(exponential_2d.cdf(sample_pos_2d), ref_sol)


def test_draw_exponential_2d(exponential_2d, rate_2d):
    """Test the draw method of exponential distribution."""
    np.random.seed(1)
    sample = np.random.exponential(scale=1 / rate_2d, size=(3, 2))
    np.random.seed(1)
    draw = exponential_2d.draw(num_draws=3)
    np.testing.assert_equal(draw, sample)


def test_logpdf_exponential_2d(exponential_2d, rate_2d, sample_pos_2d):
    """Test pdf method of Exponential distribution class."""
    scale_2d = 1 / rate_2d
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    ref_sol = scipy.stats.expon.logpdf(
        sample_pos_2d[:, 0], loc=0, scale=scale_2d[0]
    ) + scipy.stats.expon.logpdf(sample_pos_2d[:, 1], loc=0, scale=scale_2d[1])
    np.testing.assert_allclose(exponential_2d.logpdf(sample_pos_2d), ref_sol)


def test_grad_logpdf_exponential_2d(exponential_2d, sample_pos_2d):
    """Test *grad_logpdf* method of exponential distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    condition = (sample_pos_2d >= 0).all(axis=1).reshape(-1, 1)
    ref_sol = np.where(condition, -exponential_2d.rate, np.nan)
    np.testing.assert_allclose(exponential_2d.grad_logpdf(sample_pos_2d), ref_sol)


def test_pdf_exponential_2d(exponential_2d, rate_2d, sample_pos_2d):
    """Test pdf method of Exponential distribution class."""
    scale_2d = 1 / rate_2d
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    ref_sol = scipy.stats.expon.pdf(
        sample_pos_2d[:, 0], loc=0, scale=scale_2d[0]
    ) * scipy.stats.expon.pdf(sample_pos_2d[:, 1], loc=0, scale=scale_2d[1])
    np.testing.assert_allclose(exponential_2d.pdf(sample_pos_2d), ref_sol)


def test_ppf_exponential_2d(exponential_2d, rate_2d):
    """Test ppf method of Exponential distribution class."""
    with pytest.raises(ValueError, match='Method does not support multivariate distributions!'):
        exponential_2d.ppf(np.zeros(2))
