import pytest
import numpy as np
from pqueens.utils.mcmc_utils import mh_select


@pytest.fixture(scope='module')
def log_acceptance_probability():
    acceptance_probability = 0.3
    log_acceptance_probability = np.log(acceptance_probability)
    return log_acceptance_probability


@pytest.fixture(scope='module')
def current_sample():
    return 3.0


@pytest.fixture(scope='module')
def proposed_sample(current_sample):
    return 2.0*current_sample


def test_mh_select_reject(log_acceptance_probability, current_sample, proposed_sample, mocker):
    """ Test rejection of proposal in Metropolis Hastings selection
    based on given acceptance probability"""

    mocker.patch('numpy.random.uniform', return_value=0.5)

    selected_sample, accepted = mh_select(log_acceptance_probability, current_sample, proposed_sample)

    assert selected_sample is current_sample
    assert accepted is False


def test_mh_select_accept(log_acceptance_probability, current_sample, proposed_sample, mocker):
    """ Test acceptance of proposal in Metropolis Hastings selection
    based on given acceptance probability"""

    mocker.patch('numpy.random.uniform', return_value=0.25)

    selected_sample, accepted = mh_select(log_acceptance_probability, current_sample, proposed_sample)

    assert selected_sample is proposed_sample
    assert accepted is True


def test_mh_select_accept_prob_1(current_sample, proposed_sample):
    """ Test acceptance of proposal in Metropolis Hastings selection
    based on acceptance probability >= 1.0"""

    acceptance_probability = 1.
    log_acceptance_probability = np.log(acceptance_probability)
    selected_sample, accepted = mh_select(log_acceptance_probability, current_sample, proposed_sample)

    assert selected_sample is proposed_sample
    assert accepted is True


def test_mh_select_accept_prob_0(current_sample, proposed_sample):
    """ Test rejection of proposal in Metropolis Hastings selection
    based on acceptance probability = 0.0"""

    acceptance_probability = 0.
    log_acceptance_probability = np.log(acceptance_probability)
    selected_sample, accepted = mh_select(log_acceptance_probability, current_sample, proposed_sample)

    assert selected_sample is current_sample
    assert accepted is False
