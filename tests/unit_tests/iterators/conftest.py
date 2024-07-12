"""Fixtures needed across unit_tests for iterators."""

from copy import deepcopy

import pytest
from mock import Mock

from queens.distributions.lognormal import LogNormalDistribution
from queens.distributions.normal import NormalDistribution
from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters


@pytest.fixture(name="default_simulation_model")
def fixture_default_simulation_model():
    """Default simulation model."""
    interface = DirectPythonInterface(parameters=Mock(), function="ishigami90", num_workers=1)
    model = SimulationModel(interface)
    return model


@pytest.fixture(name="default_parameters_uniform_2d")
def fixture_default_parameters_uniform_2d():
    """Parameters with 2 uniform distributions."""
    x1 = UniformDistribution(lower_bound=-2, upper_bound=2)
    x2 = UniformDistribution(lower_bound=-2, upper_bound=2)
    return Parameters(x1=x1, x2=x2)


@pytest.fixture(name="default_parameters_uniform_3d")
def fixture_default_parameters_uniform_3d():
    """Parameters with 3 uniform distributions."""
    random_variable = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    return Parameters(
        x1=random_variable, x2=deepcopy(random_variable), x3=deepcopy(random_variable)
    )


@pytest.fixture(name="default_parameters_mixed")
def fixture_default_parameters_mixed():
    """Parameters with different distributions."""
    x1 = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = NormalDistribution(mean=0, covariance=4)
    x3 = LogNormalDistribution(normal_mean=0.3, normal_covariance=1)
    return Parameters(x1=x1, x2=x2, x3=x3)
