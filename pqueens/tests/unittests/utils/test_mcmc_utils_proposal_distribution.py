import pytest
import numpy as np
from pqueens.utils.mcmc_utils import NormalProposal, create_proposal_distribution


@pytest.fixture(scope='module')
def valid_lower_cholesky():
    return np.array([[1.0, 0.0, 0.],
                     [0.1, 2.0, 0.],
                     [1.0, 0.8, 3.]])


@pytest.fixture(scope='module')
def valid_covariance_matrix(valid_lower_cholesky):
    return np.dot(valid_lower_cholesky, valid_lower_cholesky.T)


@pytest.fixture(scope='module')
def invalid_dimension_covariance_matrix():
    return np.array([[[1.0, 0.1], [1.0, 0.1]],
                     [[0.2, 2.0], [0.2, 2.0]]])


@pytest.fixture(scope='module')
def invalid_rectangular_covariance_matrix():
    return np.array([[1.0, 0.1],
                     [0.2, 2.0],
                     [3.0, 0.3]])


@pytest.fixture(scope='module')
def invalid_nonsymmetric_covariance_matrix():
    return np.array([[1.0, 0.1],
                     [0.2, 2.0]])


@pytest.fixture(scope='module')
def uncorrelated_vector():
    return np.array([1.0, 2.0, 3.0])


def test_init_NormalProposal_wrong_dimension(invalid_dimension_covariance_matrix):
    """ test ValueError of init method of NormalProposal class"""

    with pytest.raises(ValueError, match=r'.*Wrong dimension.*'):
        NormalProposal(invalid_dimension_covariance_matrix)


def test_init_NormalProposal_not_quadratic(invalid_rectangular_covariance_matrix):
    """ test ValueError of init method of NormalProposal class"""

    with pytest.raises(ValueError, match=r'.*not quadratic.*'):
        NormalProposal(invalid_rectangular_covariance_matrix)


def test_init_NormalProposal_not_symmetric(invalid_nonsymmetric_covariance_matrix):
    """ test ValueError of init method of NormalProposal class"""

    with pytest.raises(ValueError, match=r'.*not symmetric.*'):
        NormalProposal(invalid_nonsymmetric_covariance_matrix)


def test_init_NormalProposal_univariate():
    """ test init method of NormalProposal class (univariate case)"""

    covariance = np.asarray(2.0)
    # cholesky decomposition of a scalar is root of scalar
    lower_cholesky = np.sqrt(covariance)
    multivariate_normal_proposal = NormalProposal(covariance)

    assert multivariate_normal_proposal.dimension is 1
    np.testing.assert_allclose(multivariate_normal_proposal.covariance, covariance)
    np.testing.assert_allclose(multivariate_normal_proposal.low_chol, lower_cholesky)


def test_init_NormalProposal_multivariate(valid_covariance_matrix, valid_lower_cholesky):
    """ test init method of NormalProposal class (multivariate case)"""

    multivariate_normal_proposal = NormalProposal(valid_covariance_matrix)

    assert multivariate_normal_proposal.dimension is valid_covariance_matrix.shape[0]
    np.testing.assert_allclose(multivariate_normal_proposal.covariance, valid_covariance_matrix)
    np.testing.assert_allclose(multivariate_normal_proposal.low_chol, valid_lower_cholesky)


def test_draw_NormalProposal(valid_covariance_matrix, valid_lower_cholesky, uncorrelated_vector, mocker):
    """ test the draw method of normal proposal distribution"""

    # univariate case
    standard_normal_sample = np.asarray(0.1)
    mocker.patch('numpy.random.randn', return_value=standard_normal_sample)
    variance = np.asarray(2.0)
    univariate_normal_proposal = NormalProposal(variance)
    normal_sample = np.sqrt(variance) * standard_normal_sample
    univariate_draw = univariate_normal_proposal.draw()


    # multivariate case
    mocker.patch('numpy.random.randn', return_value=uncorrelated_vector)
    multivariate_normal_proposal = NormalProposal(valid_covariance_matrix)
    correlated_vector = np.dot(valid_lower_cholesky, uncorrelated_vector)
    multivariate_draw = multivariate_normal_proposal.draw()

    np.testing.assert_allclose(univariate_draw, normal_sample)
    np.testing.assert_allclose(multivariate_draw, correlated_vector)


def test_create_proposal_distribution_normal(valid_covariance_matrix):
    """ test creation routine of proposal distribution objects"""

    normal_options = {'type': 'normal',
                                   'proposal_covariance': valid_covariance_matrix}

    normal_proposal = create_proposal_distribution(normal_options)

    assert isinstance(normal_proposal, NormalProposal)


def test_create_proposal_distribution_invalid():
    """ test creation routine of proposal distribution objects"""

    invalid_options = {'proposal_covariance': 1.0}

    with pytest.raises(ValueError, match=r'.*type not supported.*'):
        create_proposal_distribution(invalid_options)
