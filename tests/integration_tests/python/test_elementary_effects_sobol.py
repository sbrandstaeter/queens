"""TODO_doc."""

import numpy as np

from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.elementary_effects_iterator import ElementaryEffectsIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_elementary_effects_sobol(
    expected_result_mu,
    expected_result_mu_star,
    expected_result_sigma,
    global_settings,
):
    """Test case for elementary effects on Sobol's G-function."""
    # Parameters
    x1 = UniformDistribution(lower_bound=0, upper_bound=1)
    x2 = UniformDistribution(lower_bound=0, upper_bound=1)
    x3 = UniformDistribution(lower_bound=0, upper_bound=1)
    x4 = UniformDistribution(lower_bound=0, upper_bound=1)
    x5 = UniformDistribution(lower_bound=0, upper_bound=1)
    x6 = UniformDistribution(lower_bound=0, upper_bound=1)
    x7 = UniformDistribution(lower_bound=0, upper_bound=1)
    x8 = UniformDistribution(lower_bound=0, upper_bound=1)
    x9 = UniformDistribution(lower_bound=0, upper_bound=1)
    x10 = UniformDistribution(lower_bound=0, upper_bound=1)
    parameters = Parameters(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, x8=x8, x9=x9, x10=x10)

    # Setup iterator
    interface = DirectPythonInterface(function="sobol_g_function", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = ElementaryEffectsIterator(
        seed=2,
        num_trajectories=100,
        num_optimal_trajectories=4,
        number_of_levels=10,
        confidence_level=0.95,
        local_optimization=False,
        num_bootstrap_samples=1000,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_booleans": [False, False],
                "plotting_dir": "dummy",
                "plot_names": ["bars", "scatter"],
                "save_bool": [False, False],
            },
        },
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )
    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results["sensitivity_indices"]["mu"], expected_result_mu)
    np.testing.assert_allclose(results["sensitivity_indices"]["mu_star"], expected_result_mu_star)
    np.testing.assert_allclose(results["sensitivity_indices"]["sigma"], expected_result_sigma)
