"""TODO_doc."""
import numpy as np
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.global_settings import GlobalSettings
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.elementary_effects_iterator import ElementaryEffectsIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


@pytest.fixture(name="expected_result_mu")
def fixture_expected_result_mu():
    """Mu result fixture."""
    expected_result_mu = np.array(
        [
            25.8299150077341,
            19.28297176050532,
            -14.092164789704626,
            5.333475971922498,
            -11.385141403296364,
            13.970208961715421,
            -3.0950202483238303,
            0.6672725255532903,
            7.2385092339309445,
            -7.7664016980947075,
        ]
    )
    return expected_result_mu


@pytest.fixture(name="expected_result_mu_star")
def fixture_expected_result_mu_star():
    """Mu star result fixture."""
    expected_result_mu_star = np.array(
        [
            29.84594504725642,
            21.098173537614855,
            16.4727722348437,
            26.266876218598668,
            16.216603266281044,
            18.051629859410895,
            3.488313966697564,
            2.7128638920479147,
            7.671230484535577,
            10.299932289624746,
        ]
    )
    return expected_result_mu_star


@pytest.fixture(name="expected_result_sigma")
def fixture_expected_result_sigma():
    """Sigma result fixture."""
    expected_result_sigma = np.array(
        [
            53.88783786787971,
            41.02192670857979,
            29.841807478998156,
            43.33349033575829,
            29.407676882180404,
            31.679653142831512,
            5.241491105224932,
            4.252334015139214,
            10.38274186974731,
            18.83046700807382,
        ]
    )
    return expected_result_sigma


def test_elementary_effects_sobol(
    tmp_path, expected_result_mu, expected_result_mu_star, expected_result_sigma
):
    """Test case for elementary effects on Sobol's G-function."""
    # Global settings
    experiment_name = "elementary_effects_sobol"
    output_dir = tmp_path

    with GlobalSettings(experiment_name=experiment_name, output_dir=output_dir, debug=False) as gs:
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
        parameters = Parameters(
            x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, x8=x8, x9=x9, x10=x10
        )

        # Setup QUEENS stuff
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
        )

        # Actual analysis
        run_iterator(iterator)

        # Load results
        result_file = gs.output_dir / f"{gs.experiment_name}.pickle"
        results = load_result(result_file)

    np.testing.assert_allclose(results["sensitivity_indices"]['mu'], expected_result_mu)
    np.testing.assert_allclose(results["sensitivity_indices"]['mu_star'], expected_result_mu_star)
    np.testing.assert_allclose(results["sensitivity_indices"]['sigma'], expected_result_sigma)
