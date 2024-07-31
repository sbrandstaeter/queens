"""Test fourc run."""

import numpy as np

from queens.data_processor.data_processor_ensight import DataProcessorEnsight
from queens.distributions.uniform import UniformDistribution
from queens.drivers.mpi_driver import MpiDriver
from queens.external_geometry.fourc_dat_geometry import FourcDatExternalGeometry
from queens.interfaces.job_interface import JobInterface
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.local_scheduler import LocalScheduler
from queens.utils.io_utils import load_result


def test_fourc_mc_ensight(
    third_party_inputs,
    fourc_link_paths,
    fourc_example_expected_mean,
    fourc_example_expected_var,
    fourc_example_expected_output,
    global_settings,
):
    """Test simple fourc run."""
    # generate json input file from template
    third_party_input_file = (
        third_party_inputs / "fourc" / "meshtying3D_patch_lin_duallagr_new_struct.dat"
    )
    fourc_executable, post_ensight, _ = fourc_link_paths

    # Parameters
    nue = UniformDistribution(lower_bound=0.4, upper_bound=0.49)
    young = UniformDistribution(lower_bound=500, upper_bound=1000)
    parameters = Parameters(nue=nue, young=young)

    # Setup iterator
    external_geometry = FourcDatExternalGeometry(
        list_geometric_sets=["DSURFACE 1"],
        input_template=third_party_input_file,
    )
    data_processor = DataProcessorEnsight(
        file_name_identifier="fourc_mc_ensight_*structure.case",
        file_options_dict={
            "delete_field_data": False,
            "geometric_target": ["geometric_set", "DSURFACE 1"],
            "physical_field_dict": {
                "vtk_field_type": "structure",
                "vtk_array_type": "point_array",
                "vtk_field_label": "displacement",
                "field_components": [0, 1, 2],
            },
            "target_time_lst": ["last"],
        },
        external_geometry=external_geometry,
    )
    scheduler = LocalScheduler(
        experiment_name=global_settings.experiment_name,
        num_procs=2,
        num_procs_post=1,
        max_concurrent=2,
    )
    driver = MpiDriver(
        input_template=third_party_input_file,
        path_to_executable=fourc_executable,
        path_to_postprocessor=post_ensight,
        post_file_prefix="fourc_mc_ensight",
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
    np.testing.assert_array_almost_equal(results["mean"], fourc_example_expected_mean, decimal=6)
    np.testing.assert_array_almost_equal(results["var"], fourc_example_expected_var, decimal=6)
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], fourc_example_expected_output, decimal=6
    )
