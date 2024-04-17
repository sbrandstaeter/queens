"""Test chaospy wrapper."""

import pytest

from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.polynomial_chaos_iterator import PolynomialChaosIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_polynomial_chaos_pseudo_spectral_borehole(tmp_path, _initialize_global_settings):
    """Test case for the PC iterator using a pseudo spectral approach."""
    # Parameters
    rw = UniformDistribution(lower_bound=0.05, upper_bound=0.15)
    r = UniformDistribution(lower_bound=100, upper_bound=50000)
    tu = UniformDistribution(lower_bound=63070, upper_bound=115600)
    hu = UniformDistribution(lower_bound=990, upper_bound=1110)
    tl = UniformDistribution(lower_bound=63.1, upper_bound=116)
    hl = UniformDistribution(lower_bound=700, upper_bound=820)
    l = UniformDistribution(lower_bound=1120, upper_bound=1680)
    kw = UniformDistribution(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="borehole83_lofi", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = PolynomialChaosIterator(
        approach="pseudo_spectral",
        seed=42,
        num_collocation_points=50,
        sampling_rule="gaussian",
        sparse=True,
        polynomial_order=2,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)
    assert results["mean"] == pytest.approx(61.78966587)
    assert results["covariance"] == pytest.approx([1312.23414971])


def test_polynomial_chaos_collocation_borehole(inputdir, tmp_path, _initialize_global_settings):
    """Test for the PC iterator using a collocation approach."""
    # Parameters
    rw = UniformDistribution(lower_bound=0.05, upper_bound=0.15)
    r = UniformDistribution(lower_bound=100, upper_bound=50000)
    tu = UniformDistribution(lower_bound=63070, upper_bound=115600)
    hu = UniformDistribution(lower_bound=990, upper_bound=1110)
    tl = UniformDistribution(lower_bound=63.1, upper_bound=116)
    hl = UniformDistribution(lower_bound=700, upper_bound=820)
    l = UniformDistribution(lower_bound=1120, upper_bound=1680)
    kw = UniformDistribution(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="borehole83_lofi", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = PolynomialChaosIterator(
        approach="collocation",
        seed=42,
        num_collocation_points=50,
        sampling_rule="sobol",
        polynomial_order=2,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)
    assert results["mean"] == pytest.approx(62.05018243)
    assert results["covariance"] == pytest.approx([1273.81372103])
