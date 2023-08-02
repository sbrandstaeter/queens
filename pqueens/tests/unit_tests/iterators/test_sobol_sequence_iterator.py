"""TODO_doc."""

import numpy as np
import pytest

from pqueens.distributions.lognormal import LogNormalDistribution
from pqueens.distributions.normal import NormalDistribution
from pqueens.distributions.uniform import UniformDistribution
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.sobol_sequence_iterator import SobolSequenceIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.parameters.parameters import Parameters


@pytest.fixture()
def global_settings():
    """TODO_doc."""
    global_settings = {"experiment_name": "test"}
    return global_settings


@pytest.fixture()
def default_parameters():
    """Default parameters."""
    x1 = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = NormalDistribution(mean=0, covariance=4)
    x3 = LogNormalDistribution(normal_mean=0.3, normal_covariance=1)
    return Parameters(x1=x1, x2=x2, x3=x3)


@pytest.fixture()
def default_model(default_parameters):
    """TODO_doc."""
    # create interface
    interface = DirectPythonInterface(
        parameters=default_parameters, function="ishigami90", num_workers=1
    )

    # create mock model
    model = SimulationModel(interface)

    return model


@pytest.fixture()
def default_qmc_iterator(default_model, global_settings, default_parameters):
    """TODO_doc."""
    my_iterator = SobolSequenceIterator(
        default_model,
        parameters=default_parameters,
        seed=42,
        number_of_samples=100,
        randomize=True,
        result_description=None,
        global_settings=global_settings,
    )
    return my_iterator


def test_correct_sampling(default_qmc_iterator):
    """Test if we get correct samples."""
    default_qmc_iterator.pre_run()

    # check if mean and std match
    means_ref = np.array([0.0204326276, -0.0072869057, 2.2047842442])

    np.testing.assert_allclose(
        np.mean(default_qmc_iterator.samples, axis=0), means_ref, 1e-09, 1e-09
    )

    std_ref = np.array([1.8154208424, 1.9440692556, 2.5261052422])
    np.testing.assert_allclose(np.std(default_qmc_iterator.samples, axis=0), std_ref, 1e-09, 1e-09)

    # check if samples are identical too
    ref_sample_first_row = np.array([3.1259685949, -2.5141151734, 3.4102209094])

    np.testing.assert_allclose(
        default_qmc_iterator.samples[0, :], ref_sample_first_row, 1e-07, 1e-07
    )


def test_correct_results(default_qmc_iterator):
    """Test if we get correct results."""
    default_qmc_iterator.pre_run()
    default_qmc_iterator.core_run()

    # check if results are identical too
    ref_results = np.array(
        [
            2.6397695522,
            5.1992267219,
            2.9953908199,
            7.8633899617,
            0.5600099301,
            -55.9005701034,
            6.6225412593,
            5.0542526964,
            6.4044981383,
            -0.9481326093,
        ]
    )

    np.testing.assert_allclose(
        default_qmc_iterator.output["mean"][0:10].flatten(), ref_results, 1e-09, 1e-09
    )
