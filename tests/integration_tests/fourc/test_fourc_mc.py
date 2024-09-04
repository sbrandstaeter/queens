"""Test fourc run."""

import numpy as np

from queens.data_processor.data_processor_pvd import DataProcessorPvd
from queens.distributions.uniform import UniformDistribution
from queens.drivers.fourc_driver import FourcDriver
from queens.interfaces.job_interface import JobInterface
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.local_scheduler import LocalScheduler
from queens.utils.io_utils import load_result


def test_fourc_mc(
    third_party_inputs,
    fourc_link_paths,
    fourc_example_expected_output,
    global_settings,
):
    """Test simple fourc run."""
    # generate json input file from template
    fourc_input_file_template = third_party_inputs / "fourc" / "solid_runtime_hex8.dat"
    fourc_executable, _, _ = fourc_link_paths

    # Parameters
    parameter_1 = UniformDistribution(lower_bound=0.0, upper_bound=1.0)
    parameter_2 = UniformDistribution(lower_bound=0.0, upper_bound=1.0)
    parameters = Parameters(parameter_1=parameter_1, parameter_2=parameter_2)

    data_processor = DataProcessorPvd(
        field_name="displacement",
        file_name_identifier=f"{global_settings.experiment_name}_*.pvd",
        file_options_dict={},
    )

    scheduler = LocalScheduler(
        experiment_name=global_settings.experiment_name,
        num_procs=2,
        num_jobs=2,
    )
    driver = FourcDriver(
        input_template=fourc_input_file_template,
        executable=fourc_executable,
        data_processor=data_processor,
    )
    interface = JobInterface(scheduler=scheduler, driver=driver, parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = MonteCarloIterator(
        seed=42,
        num_samples=2,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # assert statements
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], fourc_example_expected_output, decimal=6
    )
