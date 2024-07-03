"""TODO_doc."""
import logging

import pytest

from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.elementary_effects_iterator import ElementaryEffectsIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result

_logger = logging.getLogger(__name__)


def test_elementary_effects_ishigami(global_settings):
    """Test case for elementary effects iterator."""
    # Parameters
    x1 = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = UniformDistribution(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="ishigami90", parameters=parameters)
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
    _logger.info(results)

    assert results["sensitivity_indices"]['mu'][0] == pytest.approx(15.46038594, abs=1e-7)
    assert results["sensitivity_indices"]['mu'][1] == pytest.approx(0.0, abs=1e-7)
    assert results["sensitivity_indices"]['mu'][2] == pytest.approx(0.0, abs=1e-7)

    assert results["sensitivity_indices"]['mu_star'][0] == pytest.approx(15.460385940, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star'][1] == pytest.approx(1.47392000, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star'][2] == pytest.approx(5.63434321, abs=1e-7)

    assert results["sensitivity_indices"]['sigma'][0] == pytest.approx(15.85512257, abs=1e-7)
    assert results["sensitivity_indices"]['sigma'][1] == pytest.approx(1.70193622, abs=1e-7)
    assert results["sensitivity_indices"]['sigma'][2] == pytest.approx(9.20084394, abs=1e-7)

    assert results["sensitivity_indices"]['mu_star_conf'][0] == pytest.approx(13.53414548, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star_conf'][1] == pytest.approx(0.0, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star_conf'][2] == pytest.approx(5.51108773, abs=1e-7)
