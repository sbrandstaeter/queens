import numpy as np
import pytest

from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.iterators.sobol_sequence_iterator import SobolSequenceIterator
from pqueens.models.simulation_model import SimulationModel
from pqueens.variables.variables import Variables


@pytest.fixture()
def global_settings():
    global_settings = {"experiment_name": "test"}
    return global_settings


@pytest.fixture()
def default_model():
    uncertain_parameters = {}
    uncertain_parameter1 = {
        "size": 1,
        "distribution": "uniform",
        "lower_bound": -3.14159265359,
        "upper_bound": 3.14159265359,
    }

    uncertain_parameter2 = {
        "type": "FLOAT",
        "size": 1,
        "distribution": "normal",
        "mean": 0,
        "covariance": 4,
    }

    uncertain_parameter3 = {
        "type": "FLOAT",
        "size": 1,
        "distribution": "lognormal",
        "mu": 0.3,
        "sigma": 1,
    }

    random_variables = {
        'x1': uncertain_parameter1,
        'x2': uncertain_parameter2,
        'x3': uncertain_parameter3,
    }
    uncertain_parameters["random_variables"] = random_variables

    variables = Variables.from_uncertain_parameters_create(uncertain_parameters)

    # create interface
    interface = DirectPythonInterface('test_interface', 'ishigami.py', variables)

    # create mock model
    model = SimulationModel("my_model", interface, uncertain_parameters)

    return model


@pytest.fixture()
def default_qmc_iterator(default_model, global_settings):
    my_iterator = SobolSequenceIterator(
        default_model,
        seed=42,
        number_of_samples=100,
        randomize=True,
        result_description=None,
        global_settings=global_settings,
    )
    return my_iterator


@pytest.mark.unit_tests
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


@pytest.mark.unit_tests
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
