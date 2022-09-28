"""Test-module for beta distribution."""

import numpy as np
import pytest
import scipy.stats

from pqueens.distributions import from_config_create_distribution


@pytest.fixture(params=[0.5, [-1.0, 0.0, 1.0, 2.0]])
def sample_pos(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(scope='module')
def lower_bound():
    """A possible left bound of interval."""
    return -1.0


@pytest.fixture(scope='module')
def upper_bound():
    """A possible right bound of interval."""
    return 2.0


@pytest.fixture(scope='module')
def shape_a():
    """A possible shape parameter a."""
    return 3.0


@pytest.fixture(scope='module')
def shape_b():
    """A possible shape parameter b."""
    return 0.5


@pytest.fixture(scope='module')
def beta(lower_bound, upper_bound, shape_a, shape_b):
    """A beta distribution."""
    distribution_options = {
        'distribution': 'beta',
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'a': shape_a,
        'b': shape_b,
    }
    return from_config_create_distribution(distribution_options)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------


def test_init_beta(beta, lower_bound, upper_bound, shape_a, shape_b):
    """Test init method of Beta Distribution class."""
    lower_bound = np.array(lower_bound).reshape(1)
    upper_bound = np.array(upper_bound).reshape(1)
    width = upper_bound - lower_bound
    mean_ref = scipy.stats.beta.mean(a=shape_a, b=shape_b, loc=lower_bound, scale=width).reshape(1)
    var_ref = scipy.stats.beta.var(a=shape_a, b=shape_b, loc=lower_bound, scale=width).reshape(1, 1)

    assert beta.dimension == 1
    np.testing.assert_equal(beta.mean, mean_ref)
    np.testing.assert_equal(beta.covariance, var_ref)
    np.testing.assert_equal(beta.lower_bound, lower_bound)
    np.testing.assert_equal(beta.upper_bound, upper_bound)
    np.testing.assert_equal(beta.a, shape_a)
    np.testing.assert_equal(beta.b, shape_b)


def test_init_beta_wrong_interval(lower_bound, shape_a, shape_b):
    """Test init method of Beta Distribution class."""
    with pytest.raises(ValueError, match=r'Lower bound must be smaller than upper bound*'):
        distribution_options = {
            'distribution': 'beta',
            'lower_bound': lower_bound,
            'upper_bound': lower_bound - np.abs(lower_bound),
            'a': shape_a,
            'b': shape_b,
        }
        from_config_create_distribution(distribution_options)


def test_init_beta_negative_shape(lower_bound, shape_a, shape_b):
    """Test init method of Beta Distribution class."""
    with pytest.raises(ValueError, match=r'The parameter \'b\' has to be positive.*'):
        distribution_options = {
            'distribution': 'beta',
            'lower_bound': lower_bound,
            'upper_bound': lower_bound - np.abs(lower_bound),
            'a': shape_a,
            'b': -shape_b,
        }
        from_config_create_distribution(distribution_options)


def test_cdf_beta(beta, lower_bound, upper_bound, sample_pos, shape_a, shape_b):
    """Test cdf method of beta distribution class."""
    width = upper_bound - lower_bound
    ref_sol = scipy.stats.beta.cdf(
        sample_pos, a=shape_a, b=shape_b, loc=lower_bound, scale=width
    ).reshape(-1)
    np.testing.assert_equal(beta.cdf(sample_pos), ref_sol)


def test_draw_beta(beta, lower_bound, upper_bound, mocker):
    """Test the draw method of beta distribution."""
    sample = np.asarray(0.5 * (lower_bound + upper_bound)).reshape(1, 1)
    mocker.patch('scipy.stats._distn_infrastructure.rv_frozen.rvs', return_value=sample)
    draw = beta.draw()
    np.testing.assert_equal(draw, sample)


def test_logpdf_beta(beta, lower_bound, upper_bound, sample_pos, shape_a, shape_b):
    """Test pdf method of beta distribution class."""
    width = upper_bound - lower_bound
    ref_sol = scipy.stats.beta.logpdf(
        sample_pos, a=shape_a, b=shape_b, loc=lower_bound, scale=width
    ).reshape(-1)
    np.testing.assert_equal(beta.logpdf(sample_pos), ref_sol)


def test_grad_logpdf_beta(beta, sample_pos):
    """Test grad_logpdf method of beta distribution class."""
    with pytest.raises(
        NotImplementedError,
        match=r'This method is currently not implemented for ' r'the beta distribution.',
    ):
        beta.grad_logpdf(sample_pos)


def test_pdf_beta(beta, lower_bound, upper_bound, sample_pos, shape_a, shape_b):
    """Test pdf method of beta distribution class."""
    width = upper_bound - lower_bound
    ref_sol = scipy.stats.beta.pdf(
        sample_pos, a=shape_a, b=shape_b, loc=lower_bound, scale=width
    ).reshape(-1)
    np.testing.assert_equal(beta.pdf(sample_pos), ref_sol)


def test_ppf_beta(beta, lower_bound, upper_bound, shape_a, shape_b):
    """Test ppf method of beta distribution class."""
    quantile = 0.5
    width = upper_bound - lower_bound
    ref_sol = scipy.stats.beta.ppf(
        quantile, a=shape_a, b=shape_b, loc=lower_bound, scale=width
    ).reshape(-1)
    np.testing.assert_equal(beta.ppf(quantile), ref_sol)
